"""
Tests the Ray observation source maps scraped gauges and replica counts onto per-pool signals.
"""

from __future__ import annotations

from spatial_ray.control.ray_source import (
    CPU,
    GPU,
    MetricsView,
    RayObservationSource,
    _Ewma,
    build_observation,
)


def _view() -> MetricsView:
    # two transform nodes and one inference node
    return MetricsView(
        node_cpu={"t1": 40.0, "t2": 60.0, "g1": 5.0},
        node_gpu={"t1": 0.0, "t2": 0.0, "g1": 80.0},
        work={"decode": 2048.0, "inference": 12.0},
        queue={"inference": 7.0, "transform": 1.0},
        roles={"t1": "transform", "t2": "transform", "g1": "inference"},
        mean_bytes={"decode": 512.0},
    )


def _kinds() -> dict[str, str | None]:
    return {"decode": None, "transform": CPU, "inference": GPU}


def test_ewma():
    """The first sample sets the average and later samples pull it over."""
    ewma = _Ewma(alpha=0.5)
    assert ewma.update(1.0) == 1.0
    assert ewma.update(0.0) == 0.5
    assert ewma.update(0.0) == 0.25


def test_ewma_holds_across_a_missed_sample():
    """A missed sample holds the average rather than decaying it toward an idle reading."""
    ewma = _Ewma(alpha=0.5)
    assert ewma.update(1.0) == 1.0
    assert ewma.update(None) == 1.0
    assert ewma.update(None) == 1.0


def test_build_observation_util():
    """Each pool reads mean util over its nodes as a fraction."""
    ewmas = {"transform": _Ewma(alpha=1.0), "inference": _Ewma(alpha=1.0)}
    observation = build_observation(9.0, _view(), {"transform": 2, "inference": 1}, _kinds(), ewmas)
    assert observation.t_s == 9.0
    assert observation.pools["transform"].utilization == 0.50
    assert observation.pools["inference"].utilization == 0.80
    assert observation.pools["inference"].queue_depth == 7.0
    assert observation.pools["inference"].work_in_flight == 12.0


def test_util_per_replica_ignores_idle_nodes():
    """Dividing by replicas keeps an idle provisioned node from diluting utilization."""
    view = MetricsView(
        node_cpu={},
        node_gpu={"g1": 90.0, "g2": 0.0},
        work={},
        queue={},
        roles={"g1": "inference", "g2": "inference"},
    )
    ewmas = {"inference": _Ewma(alpha=1.0)}
    observation = build_observation(0.0, view, {"inference": 1}, {"inference": GPU}, ewmas)
    assert observation.pools["inference"].utilization == 0.90


def test_pool_without_util_kind_reports_absent_utilization():
    """A pool with no util kind reports its work gauges but no utilization at all."""
    observation = build_observation(0.0, _view(), {"decode": 4}, _kinds(), {})
    decode = observation.pools["decode"]
    assert decode.utilization is None
    assert decode.work_in_flight == 2048.0
    assert decode.mean_decoded_bytes == 512.0


def test_unreported_gauge_reads_absent_not_zero():
    """A gauge no endpoint reported reads None while a reported zero keeps its real value."""
    observation = build_observation(0.0, _view(), {"decode": 4}, _kinds(), {})
    decode = observation.pools["decode"]
    assert decode.queue_depth is None
    assert decode.queued_depth is None
    idle = MetricsView(node_cpu={}, node_gpu={}, work={}, queue={"transform": 0.0}, roles={})
    transform = build_observation(0.0, idle, {"transform": 1}, {"transform": None}, {}).pools[
        "transform"
    ]
    assert transform.queue_depth == 0.0


def test_replica_queue_scales_to_the_census():
    """The per-replica queue mean times the replica census estimates the pool's whole backlog."""
    view = MetricsView(node_cpu={}, node_gpu={}, work={}, queue={"decode": 8.0}, roles={})
    observation = build_observation(0.0, view, {"decode": 6}, {"decode": None}, {})
    assert observation.pools["decode"].queue_depth == 48.0


def _queued(per_router: float, handles: int) -> MetricsView:
    # a scrape where only the given number of routers exported their queued gauge
    return MetricsView(
        node_cpu={},
        node_gpu={},
        work={},
        queue={},
        roles={},
        queued={"decode": per_router},
        queued_handles={"decode": handles},
    )


def test_router_queue_scales_to_the_peak_handle_count():
    """A router dropping out keeps the queued estimate on the most handles ever seen reporting."""
    peaks: dict[str, int] = {}
    both = build_observation(
        0.0, _queued(10.0, 2), {"decode": 1}, {"decode": None}, {}, peak_handles=peaks
    )
    assert both.pools["decode"].queued_depth == 20.0
    one = build_observation(
        1.0, _queued(10.0, 1), {"decode": 1}, {"decode": None}, {}, peak_handles=peaks
    )
    assert one.pools["decode"].queued_depth == 20.0


def test_source_smooths():
    """The source keeps its EWMA state between ticks."""
    hot = MetricsView(
        node_cpu={"t1": 100.0}, node_gpu={}, work={}, queue={}, roles={"t1": "transform"}
    )
    source = RayObservationSource(
        read_metrics=lambda: hot,
        read_replicas=lambda: {"transform": 1},
        util_kinds={"transform": CPU},
        ewma_alpha=0.5,
    )
    first = source.observe().pools["transform"].utilization
    second = source.observe().pools["transform"].utilization
    assert first == 1.0
    assert second == 1.0


def test_first_scrape_without_gauges_reads_absent():
    """With nothing seen yet a gauge reads absent and ages nothing, leaving the pool to be held."""
    empty = MetricsView(node_cpu={}, node_gpu={}, work={}, queue={}, roles={"t1": "decode"})
    source = RayObservationSource(
        read_metrics=lambda: empty,
        read_replicas=lambda: {"decode": 3},
        util_kinds={"decode": None},
    )
    pool = source.observe().pools["decode"]
    assert pool.queue_depth is None
    assert pool.stale_s == 0.0


def test_whole_scrape_missing_gauges_holds_last_good():
    """A scrape answering everywhere but omitting a gauge carries it forward instead of wiping."""
    whole = MetricsView(
        node_cpu={}, node_gpu={}, work={}, queue={"decode": 30.0}, roles={}, queued={"decode": 4.0}
    )
    silent = MetricsView(node_cpu={}, node_gpu={}, work={}, queue={}, roles={})
    views = iter([whole, silent])
    source = RayObservationSource(
        read_metrics=lambda: next(views),
        read_replicas=lambda: {"decode": 1},
        util_kinds={"decode": None},
    )
    assert source.observe().pools["decode"].queue_depth == 30.0
    held = source.observe().pools["decode"]
    assert (held.queue_depth, held.queued_depth) == (30.0, 4.0)
    assert held.stale_s > 0.0


def test_partial_scrape_holds_last_good():
    """An unanswered endpoint carries the last good gauges forward instead of reading zero load."""
    whole = MetricsView(
        node_cpu={"t1": 100.0},
        node_gpu={},
        work={},
        queue={"decode": 30.0},
        roles={"t1": "decode"},
        queued={"decode": 50.0},
    )
    partial = MetricsView(node_cpu={}, node_gpu={}, work={}, queue={}, roles={"t1": "decode"})
    views = iter([whole, partial, partial])
    source = RayObservationSource(
        read_metrics=lambda: next(views),
        read_replicas=lambda: {"decode": 1},
        util_kinds={"decode": CPU},
        ewma_alpha=0.5,
    )
    first = source.observe().pools["decode"]
    assert (first.queue_depth, first.queued_depth, first.stale_s) == (30.0, 50.0, 0.0)
    held = source.observe().pools["decode"]
    assert (held.queue_depth, held.queued_depth) == (30.0, 50.0)
    assert held.utilization == 1.0
    assert source.observe().pools["decode"].stale_s > held.stale_s


def test_source_ramps():
    """A cold start jumping to full util reports a damped midpoint first."""
    views = iter(
        [
            MetricsView(
                node_cpu={"t1": 0.0}, node_gpu={}, work={}, queue={}, roles={"t1": "transform"}
            ),
            MetricsView(
                node_cpu={"t1": 100.0}, node_gpu={}, work={}, queue={}, roles={"t1": "transform"}
            ),
        ]
    )
    source = RayObservationSource(
        read_metrics=lambda: next(views),
        read_replicas=lambda: {"transform": 1},
        util_kinds={"transform": CPU},
        ewma_alpha=0.25,
    )
    assert source.observe().pools["transform"].utilization == 0.0
    assert source.observe().pools["transform"].utilization == 0.25

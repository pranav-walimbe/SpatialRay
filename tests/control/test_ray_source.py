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
    )


def _kinds() -> dict[str, str | None]:
    return {"decode": None, "transform": CPU, "inference": GPU}


def test_ewma():
    """The first sample sets the average and later samples pull it over."""
    ewma = _Ewma(alpha=0.5)
    assert ewma.update(1.0) == 1.0
    assert ewma.update(0.0) == 0.5
    assert ewma.update(0.0) == 0.25


def test_build_observation_util():
    """Each pool reads mean util over its nodes as a fraction."""
    ewmas = {"transform": _Ewma(alpha=1.0), "inference": _Ewma(alpha=1.0)}
    observation = build_observation(9.0, _view(), {"transform": 2, "inference": 1}, _kinds(), ewmas)
    assert observation.t_s == 9.0
    assert observation.pools["transform"].utilization == 0.50
    assert observation.pools["inference"].utilization == 0.80
    assert observation.pools["inference"].queue_depth == 7.0
    assert observation.pools["inference"].work_in_flight == 12.0


def test_build_observation_backlog_only():
    """A pool with no util kind reports work but zero utilization."""
    observation = build_observation(0.0, _view(), {"decode": 4}, _kinds(), {})
    decode = observation.pools["decode"]
    assert decode.utilization == 0.0
    assert decode.work_in_flight == 2048.0
    assert decode.queue_depth == 0.0


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

"""
Tests the dynamic policy sizes each pool from its own signal and skips unconfigured pools.
"""

from __future__ import annotations

from spatial_ray.policy.dynamic import DynamicPolicy, disaggregated_dynamic_policy
from spatial_ray.policy.signals import AdaptiveBacklog, Backlog, MaxOf, TotalBacklog, Utilization
from spatial_ray.policy.types import Observation, PoolObservation


def _pool(
    name: str,
    *,
    replicas: int = 2,
    queue_depth: float | None = 0.0,
    work_in_flight: float | None = 0.0,
    utilization: float | None = 0.0,
    mean_decoded_bytes: float | None = 0.0,
    queued_depth: float | None = 0.0,
) -> PoolObservation:
    # one pool's signals defaulting to a reported zero so a test opts into absence with None
    return PoolObservation(
        name=name,
        replicas=replicas,
        queue_depth=queue_depth,
        work_in_flight=work_in_flight,
        utilization=utilization,
        mean_decoded_bytes=mean_decoded_bytes,
        queued_depth=queued_depth,
    )


def test_inference_backlog_can_exceed_its_replicas():
    """Router-queued requests let inference ask for more replicas than it already runs."""
    policy = disaggregated_dynamic_policy(
        decode_target_ongoing_requests=8.0, inference_target_ongoing_requests=4.0
    )
    swamped = _pool("inference", replicas=1, queue_depth=5.0, queued_depth=30.0, utilization=0.24)
    assert policy.signals["inference"].demand(swamped) == 8.75 > swamped.replicas


def test_utilization_signal():
    """At the setpoint replicas hold, above it demand rises proportionally."""
    signal = Utilization(target=0.5)
    assert signal.demand(_pool("t", replicas=4, utilization=0.5)) == 4.0
    assert signal.demand(_pool("t", replicas=4, utilization=1.0)) == 8.0


def test_adaptive_backlog_signal():
    """Divides bytes in flight by concurrency times mean size, zero before the mean is set."""
    signal = AdaptiveBacklog(max_ongoing_requests=4)
    assert signal.demand(_pool("d", work_in_flight=800.0, mean_decoded_bytes=100.0)) == 2.0
    assert signal.demand(_pool("d", work_in_flight=800.0)) == 0.0


def test_backlog_signal():
    """Reads its named metric and divides by per-replica capacity."""
    bytes_signal = Backlog(per_replica=100.0, metric="work_in_flight")
    queue_signal = Backlog(per_replica=4.0, metric="queue_depth")
    pool = _pool("d", work_in_flight=250.0, queue_depth=10.0)
    assert bytes_signal.demand(pool) == 2.5
    assert queue_signal.demand(pool) == 2.5


def test_max_signal():
    """MaxOf sizes to whichever component demands more."""
    signal = MaxOf((Utilization(target=0.8), Backlog(per_replica=5.0, metric="queue_depth")))
    queue_bound = _pool("i", replicas=1, utilization=0.4, queue_depth=20.0)
    util_bound = _pool("i", replicas=4, utilization=0.8, queue_depth=1.0)
    assert signal.demand(queue_bound) == 4.0
    assert signal.demand(util_bound) == 4.0


def test_decide_ceils_and_filters():
    """Demand rounds up and pools with no signal are skipped."""
    policy = DynamicPolicy(signals={"transform": Utilization(target=0.5)})
    observation = Observation(
        t_s=0.0,
        arrival_rate=1.0,
        pools={
            "transform": _pool("transform", replicas=3, utilization=0.6),
            "decode": _pool("decode", replicas=8, work_in_flight=999.0),
        },
    )
    action = policy.decide(observation)
    assert action.targets == {"transform": 4}


def test_total_backlog_signal():
    """Running and router-queued requests both count toward the per-replica target."""
    signal = TotalBacklog(target_ongoing_requests=8.0)
    assert signal.demand(_pool("d", queue_depth=16.0, queued_depth=0.0)) == 2.0
    assert signal.demand(_pool("d", queue_depth=16.0, queued_depth=48.0)) == 8.0


def test_total_backlog_on_one_count_can_only_raise():
    """One missing count leaves a lower bound that grows the pool but never shrinks it."""
    signal = TotalBacklog(target_ongoing_requests=8.0)
    loaded = _pool("decode", replicas=3, queue_depth=36.0, queued_depth=None)
    assert signal.demand(loaded) == 4.5
    quiet = _pool("decode", replicas=3, queue_depth=8.0, queued_depth=None)
    assert signal.demand(quiet) == 3.0


def test_total_backlog_exceeds_current_replicas():
    """Router-queued demand lets decode ask for more replicas than it already runs."""
    signal = TotalBacklog(target_ongoing_requests=8.0)
    pool = _pool("decode", replicas=1, queue_depth=8.0, queued_depth=56.0)
    assert signal.demand(pool) == 8.0 > pool.replicas


def test_signals_have_no_opinion_on_an_absent_gauge():
    """Every signal returns None when a gauge it reads went unreported this scrape."""
    assert Utilization(target=0.5).demand(_pool("t", utilization=None)) is None
    assert (
        Backlog(per_replica=4.0, metric="queue_depth").demand(_pool("d", queue_depth=None)) is None
    )
    absent = _pool("d", queue_depth=None, queued_depth=None)
    assert TotalBacklog(target_ongoing_requests=8.0).demand(absent) is None
    assert AdaptiveBacklog(max_ongoing_requests=4).demand(_pool("d", work_in_flight=None)) is None


def test_max_of_ignores_absent_components():
    """MaxOf sizes on the components that reported and goes silent only when none did."""
    signal = MaxOf((Utilization(target=0.8), Backlog(per_replica=5.0, metric="queue_depth")))
    partial = _pool("i", replicas=1, utilization=None, queue_depth=20.0)
    assert signal.demand(partial) == 4.0
    assert signal.demand(_pool("i", utilization=None, queue_depth=None)) is None


def test_absent_gauge_holds_the_pool_instead_of_scaling_it_to_zero():
    """An unreported decode gauge leaves the pool out of targets rather than demanding zero."""
    policy = DynamicPolicy(signals={"decode": TotalBacklog(target_ongoing_requests=8.0)})
    absent = Observation(
        t_s=0.0,
        arrival_rate=1.0,
        pools={"decode": _pool("decode", replicas=3, queue_depth=None, queued_depth=None)},
    )
    assert policy.decide(absent).targets == {}
    idle = Observation(
        t_s=0.0,
        arrival_rate=1.0,
        pools={"decode": _pool("decode", replicas=3, queue_depth=0.0, queued_depth=0.0)},
    )
    assert policy.decide(idle).targets == {"decode": 0}


def test_decide_skips_absent_pool():
    """A configured pool missing from the observation is left out."""
    policy = disaggregated_dynamic_policy(
        decode_target_ongoing_requests=8.0, inference_target_ongoing_requests=4.0
    )
    observation = Observation(
        t_s=0.0,
        arrival_rate=1.0,
        pools={"decode": _pool("decode", replicas=2, queue_depth=16.0, queued_depth=12.0)},
    )
    action = policy.decide(observation)
    assert action.targets == {"decode": 4}

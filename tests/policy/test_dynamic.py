"""
Tests the dynamic policy sizes each pool from its own signal and skips unconfigured pools.
"""

from __future__ import annotations

from spatial_ray.policy.dynamic import DynamicPolicy, disaggregated_dynamic_policy
from spatial_ray.policy.signals import Backlog, MaxOf, Utilization
from spatial_ray.policy.types import Observation, PoolObservation


def _pool(
    name: str,
    *,
    replicas: int = 2,
    queue_depth: float = 0.0,
    work_in_flight: float = 0.0,
    utilization: float = 0.0,
) -> PoolObservation:
    # a single pool's signals with every field defaulted so each test sets only what it exercises
    return PoolObservation(
        name=name,
        replicas=replicas,
        queue_depth=queue_depth,
        work_in_flight=work_in_flight,
        utilization=utilization,
    )


def test_utilization_signal_scales_replicas_by_the_saturation_ratio():
    """At the setpoint demand holds current replicas, and above it demand rises proportionally."""
    signal = Utilization(target=0.5)
    assert signal.demand(_pool("t", replicas=4, utilization=0.5)) == 4.0
    assert signal.demand(_pool("t", replicas=4, utilization=1.0)) == 8.0


def test_backlog_signal_divides_the_named_metric_by_the_per_replica_capacity():
    """The backlog signal reads its configured field by name and ignores the other metrics."""
    bytes_signal = Backlog(per_replica=100.0, metric="work_in_flight")
    queue_signal = Backlog(per_replica=4.0, metric="queue_depth")
    pool = _pool("d", work_in_flight=250.0, queue_depth=10.0)
    assert bytes_signal.demand(pool) == 2.5
    assert queue_signal.demand(pool) == 2.5


def test_inference_max_signal_tracks_the_hotter_of_gpu_util_and_queue():
    """MaxOf sizes to whichever component demands more, so the busier bottleneck wins."""
    signal = MaxOf((Utilization(target=0.8), Backlog(per_replica=5.0, metric="queue_depth")))
    queue_bound = _pool("i", replicas=1, utilization=0.4, queue_depth=20.0)
    util_bound = _pool("i", replicas=4, utilization=0.8, queue_depth=1.0)
    assert signal.demand(queue_bound) == 4.0
    assert signal.demand(util_bound) == 4.0


def test_decide_ceils_demand_and_targets_only_configured_present_pools():
    """Each demand rounds up to whole replicas and pools with no signal are skipped."""
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


def test_decide_skips_a_configured_pool_absent_from_the_observation():
    """A pool with a signal but no observation is left out rather than defaulted."""
    policy = disaggregated_dynamic_policy(
        decode_bytes_per_replica=100.0, inference_queue_per_replica=4.0
    )
    observation = Observation(
        t_s=0.0,
        arrival_rate=1.0,
        pools={"decode": _pool("decode", replicas=2, work_in_flight=350.0)},
    )
    action = policy.decide(observation)
    assert action.targets == {"decode": 4}

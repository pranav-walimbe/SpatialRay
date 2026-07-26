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
    # one pool's signals with every field defaulted
    return PoolObservation(
        name=name,
        replicas=replicas,
        queue_depth=queue_depth,
        work_in_flight=work_in_flight,
        utilization=utilization,
    )


def test_utilization_signal():
    """At the setpoint replicas hold, above it demand rises proportionally."""
    signal = Utilization(target=0.5)
    assert signal.demand(_pool("t", replicas=4, utilization=0.5)) == 4.0
    assert signal.demand(_pool("t", replicas=4, utilization=1.0)) == 8.0


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


def test_decide_skips_absent_pool():
    """A configured pool missing from the observation is left out."""
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

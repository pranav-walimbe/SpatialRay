"""
Tests the cloud launcher maps its config onto the core controller bounds and dynamic policy.
"""

from __future__ import annotations

from perf.cloud.controller import build_policy, pool_bounds
from spatial_ray.control.bounds import PoolBounds
from spatial_ray.policy.signals import Backlog, MaxOf, Utilization


def test_pool_bounds_reads_min_and_max_replicas_per_pool():
    """Each pool's config budget becomes its clamp bounds keyed by pool name."""
    bounds = pool_bounds(
        {
            "decode": {"min_replicas": 1, "max_replicas": 8},
            "inference": {"min_replicas": 1, "max_replicas": 3},
        }
    )
    assert bounds["decode"] == PoolBounds(min_replicas=1, max_replicas=8)
    assert bounds["inference"] == PoolBounds(min_replicas=1, max_replicas=3)


def test_build_policy_wires_each_pool_to_its_configured_bottleneck_signal():
    """Decode reads work-in-flight, transform reads its util target, and inference maxes both."""
    policy = build_policy(
        {
            "decode_bytes_per_replica": 1000.0,
            "inference_queue_per_replica": 8.0,
            "transform_util_target": 0.7,
            "inference_util_target": 0.6,
        }
    )
    decode, transform, inference = (
        policy.signals[name] for name in ("decode", "transform", "inference")
    )
    assert isinstance(decode, Backlog) and decode.metric == "work_in_flight"
    assert isinstance(transform, Utilization) and transform.target == 0.7
    assert isinstance(inference, MaxOf)

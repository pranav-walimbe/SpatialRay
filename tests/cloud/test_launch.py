"""
Tests the cloud launcher maps its config onto the core controller bounds and dynamic policy.
"""

from __future__ import annotations

from perf.cloud.controller import build_policy, pool_bounds
from spatial_ray.control.bounds import PoolBounds
from spatial_ray.policy.signals import MaxOf, TotalBacklog, Utilization


def test_pool_bounds():
    """Each pool's config budget becomes its clamp bounds."""
    bounds = pool_bounds(
        {
            "decode": {"min_replicas": 1, "max_replicas": 8},
            "inference": {"min_replicas": 1, "max_replicas": 3},
        }
    )
    assert bounds["decode"] == PoolBounds(min_replicas=1, max_replicas=8)
    assert bounds["inference"] == PoolBounds(min_replicas=1, max_replicas=3)


def test_build_policy_signals():
    """Decode reads total backlog, transform reads util, inference maxes util and total backlog."""
    policy = build_policy(
        {
            "transform_util_target": 0.7,
            "inference_util_target": 0.6,
            "decode_target_ongoing_requests": 8.0,
            "inference_target_ongoing_requests": 4.0,
        }
    )
    decode, transform, inference_signal = (
        policy.inner.signals[name] for name in ("decode", "transform", "inference")
    )
    assert isinstance(decode, TotalBacklog) and decode.target_ongoing_requests == 8.0
    assert isinstance(transform, Utilization) and transform.target == 0.7
    assert isinstance(inference_signal, MaxOf)
    backlog_signal = inference_signal.signals[1]
    assert isinstance(backlog_signal, TotalBacklog)
    assert backlog_signal.target_ongoing_requests == 4.0

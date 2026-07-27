"""
Tests the cloud launcher maps its config onto the core controller bounds and dynamic policy.
"""

from __future__ import annotations

from perf.cloud.controller import build_policy, pool_bounds
from spatial_ray.control.bounds import PoolBounds
from spatial_ray.policy.signals import AdaptiveBacklog, Backlog, MaxOf, Utilization
from spatial_ray.serve.application import Application
from spatial_ray.serve.graph import DISAGGREGATED, InferenceSpec


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
    """Decode sizes on its request cap, transform reads util, inference maxes util and queue."""
    inference = InferenceSpec(model_factory=lambda: None, max_ongoing_requests=16)
    application = Application(DISAGGREGATED, inference, import_path="pkg.mod:app")
    policy = build_policy(
        {"transform_util_target": 0.7, "inference_util_target": 0.6},
        application,
    )
    decode, transform, inference_signal = (
        policy.signals[name] for name in ("decode", "transform", "inference")
    )
    assert isinstance(decode, AdaptiveBacklog) and decode.max_ongoing_requests == 32
    assert isinstance(transform, Utilization) and transform.target == 0.7
    assert isinstance(inference_signal, MaxOf)
    queue_signal = inference_signal.signals[1]
    assert isinstance(queue_signal, Backlog) and queue_signal.per_replica == 32

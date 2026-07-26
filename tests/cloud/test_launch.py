"""
Tests the cloud launcher maps its config onto the core controller bounds and dynamic policy.
"""

from __future__ import annotations

from perf.cloud.controller import build_policy, pool_bounds
from spatial_ray.control.bounds import PoolBounds
from spatial_ray.policy.signals import Backlog, MaxOf, Utilization
from spatial_ray.serve.graph import InferenceSpec


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
    """Decode reads backlog, transform reads util, inference maxes util and queue."""
    inference_spec = InferenceSpec(model_factory=lambda: None, max_ongoing_requests=16)
    policy = build_policy(
        {
            "decode_bytes_per_replica": 1000.0,
            "transform_util_target": 0.7,
            "inference_util_target": 0.6,
        },
        inference_spec,
    )
    decode, transform, inference = (
        policy.signals[name] for name in ("decode", "transform", "inference")
    )
    assert isinstance(decode, Backlog) and decode.metric == "work_in_flight"
    assert isinstance(transform, Utilization) and transform.target == 0.7
    assert isinstance(inference, MaxOf)
    queue_signal = inference.signals[1]
    assert isinstance(queue_signal, Backlog) and queue_signal.per_replica == 32

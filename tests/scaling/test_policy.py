"""
Tests workload capacity decisions and the Ray Serve application adapter.
"""

from __future__ import annotations

from types import SimpleNamespace

from spatial_ray.scaling.metrics import LEDGER_SNAPSHOT_AGE, pending_work_metric, work_rate_metric
from spatial_ray.scaling.policy import CapacityPolicy, PoolCapacity, PoolLoad
from spatial_ray.scaling.ray import (
    RayPoolScalingConfig,
    RayWorkloadAutoscalingPolicy,
    application_policy_config,
)


def test_capacity_policy_takes_the_larger_rate_or_queue_demand():
    """Steady work and pending work independently request enough replicas."""
    policy = CapacityPolicy(
        capacities={
            "inference": PoolCapacity(work_per_s=100.0, target_utilization=0.5, target_queue_s=2.0)
        }
    )
    assert policy.demand("inference", PoolLoad(work_per_s=120.0)) == 3
    assert policy.demand("inference", PoolLoad(work_per_s=10.0, pending_work=240.0)) == 3


def test_application_config_is_json_safe():
    """Capacity profiles render into Ray policy kwargs without Python objects."""
    config = application_policy_config({"inference": PoolCapacity(work_per_s=40.0)})
    profile = config["policy_kwargs"]["capacities"]["inference"]
    assert profile == {
        "work_per_s": 40.0,
        "target_utilization": 0.7,
        "target_queue_s": 2.0,
    }


def test_ray_pool_scaling_config_renders_the_serve_envelope():
    """The Ray adapter keeps replica bounds beside the capacity they constrain."""
    config = RayPoolScalingConfig(
        capacity=PoolCapacity(work_per_s=40.0),
        min_replicas=1,
        initial_replicas=2,
        max_replicas=8,
        upscale_delay_s=3.0,
        downscale_delay_s=60.0,
    )
    assert config.deployment_config() == {
        "min_replicas": 1,
        "initial_replicas": 2,
        "max_replicas": 8,
        "upscale_delay_s": 3.0,
        "downscale_delay_s": 60.0,
    }


def _context(name, *, target=1, metrics=None):
    # build the public AutoscalingContext surface consumed by the adapter
    return SimpleNamespace(
        deployment_name=name,
        target_num_replicas=target,
        aggregated_metrics=metrics,
        policy_state={},
    )


def test_ray_policy_reads_ingress_work_and_exact_pending_work():
    """Ingress work rates and exact pending work produce a coordinated Ray target."""
    policy = RayWorkloadAutoscalingPolicy(
        capacities={
            "inference": {
                "work_per_s": 100.0,
                "target_utilization": 0.5,
                "target_queue_s": 2.0,
            }
        }
    )
    ingress = _context(
        "ingress",
        metrics={
            work_rate_metric("inference"): {"a": 80.0, "b": 20.0},
            LEDGER_SNAPSHOT_AGE: {"a": 0.1, "b": 0.2},
            pending_work_metric("inference"): {"a": 240.0, "b": 240.0},
        },
    )
    inference = _context("inference", target=1)
    decisions, state = policy({"ingress-id": ingress, "inference-id": inference})
    assert decisions == {"ingress-id": 1, "inference-id": 3}
    assert state == {"ingress-id": {}, "inference-id": {}}


def test_ray_policy_holds_when_ingress_metrics_are_absent():
    """Missing custom metrics hold existing Ray targets instead of scaling down."""
    policy = RayWorkloadAutoscalingPolicy(capacities={"decode": {"work_per_s": 100.0}})
    contexts = {
        "ingress-id": _context("ingress", target=2),
        "decode-id": _context("decode", target=4),
    }
    decisions, _ = policy(contexts)
    assert decisions == {"ingress-id": 2, "decode-id": 4}


def test_ray_policy_uses_pending_work_when_arrival_window_is_empty():
    """Exact pending work remains visible after the arrival-rate window becomes empty."""
    policy = RayWorkloadAutoscalingPolicy(
        capacities={
            "decode": {
                "work_per_s": 100.0,
                "target_utilization": 0.5,
                "target_queue_s": 2.0,
            }
        }
    )
    ingress = _context(
        "ingress",
        metrics={
            work_rate_metric("decode"): {"a": 0.0},
            LEDGER_SNAPSHOT_AGE: {"a": 0.1},
            pending_work_metric("decode"): {"a": 240.0},
        },
    )
    decode = _context("decode")
    decisions, _ = policy({"ingress-id": ingress, "decode-id": decode})
    assert decisions["decode-id"] == 3


def test_ray_policy_holds_when_ledger_snapshot_is_stale():
    """A stale ledger snapshot cannot trigger a decision from obsolete pending work."""
    policy = RayWorkloadAutoscalingPolicy(
        capacities={"decode": {"work_per_s": 100.0}},
        max_snapshot_age_s=2.0,
    )
    ingress = _context(
        "ingress",
        metrics={
            work_rate_metric("decode"): {"a": 1000.0},
            LEDGER_SNAPSHOT_AGE: {"a": 3.0},
            pending_work_metric("decode"): {"a": 1000.0},
        },
    )
    decode = _context("decode", target=4)
    decisions, _ = policy({"ingress-id": ingress, "decode-id": decode})
    assert decisions["decode-id"] == 4

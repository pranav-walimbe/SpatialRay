"""
Tests the Ray actuator patches num_replicas into the serveConfigV2 and submits it for reconcile.
"""

from __future__ import annotations

from typing import Any

from spatial_ray.control.ray_actuator import RayActuator, patch_replica_targets
from spatial_ray.policy.types import Action
from spatial_ray.serve.graph import DISAGGREGATED, InferenceSpec
from spatial_ray.serve.serve_config import compile_serve_config


def _config() -> dict[str, Any]:
    # a real compiled serveConfigV2 so patching and schema validation see production shape
    inference = InferenceSpec(model_factory=lambda: None)
    return compile_serve_config(DISAGGREGATED, inference, import_path="perf.cloud.app:app")


class _FakeClient:
    # captures the schema handed to deploy_apps in place of a live Serve controller
    def __init__(self) -> None:
        self.submitted: Any = None
        self.blocking: bool | None = None

    def deploy_apps(self, config: Any, _blocking: bool = False) -> None:
        self.submitted = config
        self.blocking = _blocking


def test_patch_overrides_named_replicas_drops_autoscaling_and_leaves_the_source_intact():
    """Only named deployments change, their autoscaling is dropped, and the input is not mutated."""
    config = _config()
    config["applications"][0]["deployments"][0]["autoscaling_config"] = {"min_replicas": 1}
    patched = patch_replica_targets(config, {"decode": 5})
    deployments = {d["name"]: d for d in patched["applications"][0]["deployments"]}
    assert deployments["decode"]["num_replicas"] == 5
    assert "autoscaling_config" not in deployments["decode"]
    assert deployments["transform"]["num_replicas"] == 1
    assert config["applications"][0]["deployments"][0]["autoscaling_config"] == {"min_replicas": 1}


def test_apply_patches_forward_and_submits_a_valid_non_blocking_schema():
    """Successive applies accumulate onto the held config and each submit validates as a schema."""
    client = _FakeClient()
    actuator = RayActuator(_config(), client=client)
    actuator.apply(Action(targets={"decode": 4}))
    first = {d.name: d for d in client.submitted.applications[0].deployments}
    assert first["decode"].num_replicas == 4
    assert client.blocking is False
    actuator.apply(Action(targets={"inference": 3}))
    second = {d.name: d for d in client.submitted.applications[0].deployments}
    assert second["decode"].num_replicas == 4
    assert second["inference"].num_replicas == 3

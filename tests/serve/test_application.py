"""
Tests the Application renders one grouping as both a bound graph and a matching serveConfigV2.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from spatial_ray.scaling.policy import PoolCapacity
from spatial_ray.scaling.ray import POLICY_IMPORT_PATH, RayPoolScalingConfig
from spatial_ray.serve.application import Application
from spatial_ray.serve.graph import DISAGGREGATED, InferenceSpec


def _application(import_path="pkg.mod:app", app_name="spatialray"):
    # decode carries an explicit cap since the grouping leaves the widths to config
    inference = InferenceSpec(model_factory=lambda: None)
    decode, transform = DISAGGREGATED
    grouping = (replace(decode, max_ongoing_requests=12), transform)
    return Application(grouping, inference, import_path=import_path, app_name=app_name)


def test_graph_binds_offline():
    """The graph property binds the pools and inference without a cluster."""
    assert _application().graph is not None


def test_serve_config_import_path():
    """The compiled config re-imports the application under its own name."""
    config = _application(import_path="pkg.mod:app", app_name="ray_app").serve_config
    application = config["applications"][0]
    assert application["import_path"] == "pkg.mod:app"
    assert application["name"] == "ray_app"


def test_serve_config_deployments():
    """The compiled config lists one deployment per pool and carries each pool's knobs."""
    config = _application().serve_config
    deployments = {d["name"]: d for d in config["applications"][0]["deployments"]}
    assert list(deployments) == ["decode", "transform", "inference"]
    assert deployments["decode"]["max_ongoing_requests"] == 12


def test_serve_config_application_policy():
    """Application forwards a coordinated autoscaling policy into its Serve config."""
    application = _application()
    grouping = tuple(
        replace(spec, autoscaling_config={"min_replicas": 0, "max_replicas": 10})
        for spec in application.grouping
    )
    inference = replace(
        application.inference,
        autoscaling_config={"min_replicas": 0, "max_replicas": 10},
    )
    application = replace(
        application,
        grouping=grouping,
        inference=inference,
        autoscaling_policy={"policy_function": "pkg.policy:scale"},
    )
    config = application.serve_config["applications"][0]
    assert config["autoscaling_policy"] == {"policy_function": "pkg.policy:scale"}
    assert config["deployments"][-1]["name"] == "ingress"


def test_with_workload_autoscaling_configures_every_pool():
    """One typed mapping configures matching pool envelopes and the coordinated policy."""
    application = _application().with_workload_autoscaling(
        {
            name: RayPoolScalingConfig(
                capacity=PoolCapacity(work_per_s=capacity),
                min_replicas=1,
                max_replicas=8,
            )
            for name, capacity in {"decode": 100.0, "transform": 200.0, "inference": 40.0}.items()
        }
    )
    config = application.serve_config["applications"][0]
    assert config["autoscaling_policy"]["policy_function"] == POLICY_IMPORT_PATH
    assert {deployment["name"] for deployment in config["deployments"]} == {
        "decode",
        "transform",
        "inference",
        "ingress",
    }


def test_with_workload_autoscaling_requires_exact_pool_names():
    """A partial profile cannot silently leave a coordinated pool unmanaged."""
    with pytest.raises(ValueError, match="missing: inference, transform"):
        _application().with_workload_autoscaling(
            {"decode": RayPoolScalingConfig(capacity=PoolCapacity(work_per_s=100.0))}
        )

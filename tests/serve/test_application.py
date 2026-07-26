"""
Tests the Application renders one grouping as both a bound graph and a matching serveConfigV2.
"""

from __future__ import annotations

from spatial_ray.serve.application import Application
from spatial_ray.serve.graph import DISAGGREGATED, InferenceSpec


def _application(import_path="pkg.mod:app", app_name="spatialray"):
    inference = InferenceSpec(model_factory=lambda: None)
    return Application(DISAGGREGATED, inference, import_path=import_path, app_name=app_name)


def test_graph_binds_offline():
    """The graph property binds the pools and inference into an application without a cluster."""
    assert _application().graph is not None


def test_serve_config_carries_the_applications_import_path_and_name():
    """The compiled config re-imports the same application under the application's own name."""
    config = _application(import_path="pkg.mod:app", app_name="ray_app").serve_config
    application = config["applications"][0]
    assert application["import_path"] == "pkg.mod:app"
    assert application["name"] == "ray_app"


def test_serve_config_emits_one_deployment_per_pool_with_knobs():
    """The compiled config lists decode, transform, and inference and carries each pool's knobs."""
    config = _application().serve_config
    deployments = {d["name"]: d for d in config["applications"][0]["deployments"]}
    assert list(deployments) == ["decode", "transform", "inference"]
    assert deployments["decode"]["max_ongoing_requests"] == 64

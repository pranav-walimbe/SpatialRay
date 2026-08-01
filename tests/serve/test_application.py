"""
Tests the Application renders one grouping as both a bound graph and a matching serveConfigV2.
"""

from __future__ import annotations

from dataclasses import replace

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

"""
Tests compiling a pool grouping and inference spec into a serveConfigV2 dict.
"""

from __future__ import annotations

from dataclasses import replace

from spatial_ray.serve.graph import DISAGGREGATED, InferenceSpec
from spatial_ray.serve.serve_config import compile_serve_config


def test_compile_serve_config():
    """Lists one deployment per pool and carries each pool's knobs."""
    inference = InferenceSpec(model_factory=lambda: None)
    decode, transform = DISAGGREGATED
    grouping = (replace(decode, max_ongoing_requests=12), transform)
    config = compile_serve_config(grouping, inference, import_path="perf.cloud.app:app")
    app = config["applications"][0]
    assert app["import_path"] == "perf.cloud.app:app"
    deployments = {d["name"]: d for d in app["deployments"]}
    assert list(deployments) == ["decode", "transform", "inference"]
    assert deployments["decode"]["max_ongoing_requests"] == 12

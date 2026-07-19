"""
Tests compiling a pool grouping and inference spec into a serveConfigV2 dict.
"""

from __future__ import annotations

from spatial_ray.serve.graph import DISAGGREGATED, InferenceSpec
from spatial_ray.serve.serve_config import compile_serve_config


def test_compile_serve_config_lists_every_pool_and_inference_deployment():
    """The compiled application has one deployment entry per pool plus inference."""
    inference = InferenceSpec(model_factory=lambda: None)
    config = compile_serve_config(DISAGGREGATED, inference, import_path="perf.cluster.app:app")
    app = config["applications"][0]
    assert app["import_path"] == "perf.cluster.app:app"
    names = [d["name"] for d in app["deployments"]]
    assert names == ["decode", "transform", "inference"]


def test_compile_serve_config_carries_max_ongoing_requests():
    """A pool's max_ongoing_requests knob survives into its compiled deployment entry."""
    inference = InferenceSpec(model_factory=lambda: None)
    config = compile_serve_config(DISAGGREGATED, inference, import_path="perf.cluster.app:app")
    decode_deployment = config["applications"][0]["deployments"][0]
    assert decode_deployment["max_ongoing_requests"] == 64

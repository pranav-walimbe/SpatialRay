"""
Tests the graph builder groups the decode and transform stages and renders Serve options.
"""

from __future__ import annotations

from spatial_ray.serve.graph import (
    DISAGGREGATED,
    InferenceSpec,
    PoolSpec,
    build_graph,
    deployment_options,
)
from spatial_ray.workload.stages import decode, normalize, reproject_stage, tile


def test_disaggregated_grouping_splits_decode_from_transforms():
    """The default grouping isolates decode from the CPU transform stages."""
    assert [spec.name for spec in DISAGGREGATED] == ["decode", "transform"]
    decode_pool, transform_pool = DISAGGREGATED
    assert decode_pool.stages == (decode,)
    assert transform_pool.stages == (reproject_stage, normalize, tile)


def test_build_graph_binds_offline():
    """build_graph binds the pools and inference into an application without a cluster."""
    assert build_graph(inference=InferenceSpec(model_factory=lambda: None)) is not None


def test_deployment_options_renders_replicas_and_optional_request_cap():
    """A static spec renders num_replicas and includes max_ongoing_requests only when set."""
    capped = deployment_options(PoolSpec(name="decode", stages=(decode,), max_ongoing_requests=64))
    assert capped["num_replicas"] == 1
    assert capped["max_ongoing_requests"] == 64
    uncapped = deployment_options(PoolSpec(name="transform", stages=(reproject_stage, tile)))
    assert "max_ongoing_requests" not in uncapped


def test_deployment_options_prefers_autoscaling_over_static_replicas():
    """An autoscaling_config replaces the static num_replicas in the rendered options."""
    spec = PoolSpec(
        name="decode",
        stages=(decode,),
        num_replicas=3,
        autoscaling_config={"min_replicas": 1, "max_replicas": 8},
    )
    options = deployment_options(spec)
    assert options["autoscaling_config"] == {"min_replicas": 1, "max_replicas": 8}
    assert "num_replicas" not in options

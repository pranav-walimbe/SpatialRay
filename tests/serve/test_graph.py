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
from spatial_ray.serve.resources import node_resource
from spatial_ray.workload.stages import decode, normalize, reproject_stage, tile


def test_disaggregated_grouping():
    """The default grouping isolates decode from the transform stages."""
    assert [spec.name for spec in DISAGGREGATED] == ["decode", "transform"]
    decode_pool, transform_pool = DISAGGREGATED
    assert decode_pool.stages == (decode,)
    assert transform_pool.stages == (reproject_stage, normalize, tile)


def test_build_graph_binds_offline():
    """Binds the pools and inference without a cluster."""
    assert build_graph(inference=InferenceSpec(model_factory=lambda: None)) is not None


def test_options_request_cap():
    """Renders num_replicas and includes max_ongoing_requests only when set."""
    capped = deployment_options(PoolSpec(name="decode", stages=(decode,), max_ongoing_requests=64))
    assert capped["num_replicas"] == 1
    assert capped["max_ongoing_requests"] == 64
    uncapped = deployment_options(PoolSpec(name="transform", stages=(reproject_stage, tile)))
    assert "max_ongoing_requests" not in uncapped


def test_options_autoscaling():
    """An autoscaling_config replaces the static num_replicas."""
    spec = PoolSpec(
        name="decode",
        stages=(decode,),
        num_replicas=3,
        autoscaling_config={"min_replicas": 1, "max_replicas": 8},
    )
    options = deployment_options(spec)
    assert options["autoscaling_config"] == {"min_replicas": 1, "max_replicas": 8}
    assert "num_replicas" not in options


def test_node_resource_uses_the_pool_name():
    """Pool placement uses one predictable custom Ray resource key."""
    assert node_resource("inference") == "inference_node"

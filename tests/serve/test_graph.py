"""
Tests the graph builder groups the decode and transform stages and binds offline.
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


def test_deployment_options_sets_max_ongoing_requests_when_given():
    """A pool spec with max_ongoing_requests carries it into the Serve options."""
    spec = PoolSpec(name="decode", stages=(decode,), max_ongoing_requests=64)
    assert deployment_options(spec)["max_ongoing_requests"] == 64


def test_deployment_options_omits_max_ongoing_requests_by_default():
    """A pool spec with no max_ongoing_requests leaves Serve's own default in place."""
    spec = PoolSpec(name="transform", stages=(reproject_stage, normalize, tile))
    assert "max_ongoing_requests" not in deployment_options(spec)

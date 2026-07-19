"""
Tests the graph builder groups the decode and transform stages and binds offline.
"""

from __future__ import annotations

from spatial_ray.serve.graph import DISAGGREGATED, build_graph
from spatial_ray.workload.stages import decode, normalize, reproject_stage, tile


def test_disaggregated_grouping_splits_decode_from_transforms():
    """The default grouping isolates decode from the CPU transform stages."""
    assert [spec.name for spec in DISAGGREGATED] == ["decode", "transform"]
    decode_pool, transform_pool = DISAGGREGATED
    assert decode_pool.stages == (decode,)
    assert transform_pool.stages == (reproject_stage, normalize, tile)


def test_build_graph_binds_offline():
    """build_graph binds the pools and inference into an application without a cluster."""
    assert build_graph(model_factory=lambda: None) is not None

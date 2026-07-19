"""
Tests the stage timer records one cost entry per stage in execution order.
"""

from __future__ import annotations

from spatial_ray.workload.metadata import RasterPayload, RasterRequest, SceneRef
from spatial_ray.workload.profiler import time_pipeline


def _first(payload):
    # No-op stage used to exercise the timer
    return payload


def _second(payload):
    # No-op stage used to exercise the timer
    return payload


def _payload() -> RasterPayload:
    # Payload over an empty scene, the stages ignore its contents
    scene = SceneRef(item_id="x", epsg=32610, shape=(8, 8), transform=(1.0,) * 6, bands=())
    request = RasterRequest(
        scene=scene,
        band_names=(),
        window=(0, 0, 4, 4),
        target_epsg=3857,
        target_gsd=10.0,
        tile_size=2,
    )
    return RasterPayload(request=request)


def test_time_pipeline_reports_one_cost_per_stage():
    """time_pipeline records a named cost for each stage in order."""
    _, report = time_pipeline(_payload(), (_first, _second))
    assert [cost.name for cost in report.costs] == ["_first", "_second"]
    assert all(cost.wall_s >= 0.0 for cost in report.costs)

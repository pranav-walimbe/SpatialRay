"""
Tests the pool deployment bodies compose stages and wrap model output without a Serve cluster.
"""

from __future__ import annotations

import numpy as np

from spatial_ray.serve.deployments import InferencePool, StagePool
from spatial_ray.serve.messages import TileBatch
from spatial_ray.workload.metadata import RasterPayload, RasterRequest, SceneRef


def _mark(tag, log):
    # Stage that records its tag then passes the payload through unchanged
    def stage(payload):
        log.append(tag)
        return payload

    return stage


def _payload() -> RasterPayload:
    # Payload over an empty scene, the stages only touch the log
    scene = SceneRef(item_id="x", epsg=32610, shape=(8, 8), transform=(1.0,) * 6, bands=())
    request = RasterRequest(
        scene=scene,
        band_names=("red",),
        window=(0, 0, 4, 4),
        target_epsg=3857,
        target_gsd=10.0,
        tile_size=2,
    )
    return RasterPayload(request=request)


def test_stage_pool_runs_stages_in_order():
    """StagePool applies its stages left to right on the payload."""
    log = []
    StagePool(stages=(_mark("a", log), _mark("b", log))).run(_payload())
    assert log == ["a", "b"]


def test_inference_pool_wraps_model_output():
    """InferencePool returns the model output wrapped in Predictions."""
    pool = InferencePool(model_factory=lambda: lambda tiles: tiles.sum(axis=(1, 2, 3)))
    batch = TileBatch(request=_payload().request, tiles=np.ones((3, 1, 2, 2), dtype=np.float32))
    preds = pool.infer(batch)
    assert preds.array.shape == (3,)
    assert preds.request is batch.request

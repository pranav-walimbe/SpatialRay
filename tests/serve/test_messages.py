"""
Tests the deployment-boundary messages pickle cleanly across the pool boundaries.
"""

from __future__ import annotations

import pickle

import numpy as np

from spatial_ray.serve.messages import Predictions, TileBatch
from spatial_ray.workload.metadata import RasterRequest, SceneRef


def _request() -> RasterRequest:
    # Minimal request over an empty scene
    scene = SceneRef(item_id="x", epsg=32610, shape=(8, 8), transform=(1.0,) * 6, bands=())
    return RasterRequest(
        scene=scene,
        band_names=("red",),
        window=(0, 0, 4, 4),
        target_epsg=3857,
        target_gsd=10.0,
        tile_size=2,
    )


def test_boundary_messages_round_trip_through_pickle():
    """TileBatch and Predictions survive a pickle round-trip with arrays intact."""
    batch = pickle.loads(
        pickle.dumps(TileBatch(request=_request(), tiles=np.zeros((3, 1, 2, 2), dtype=np.float32)))
    )
    preds = pickle.loads(
        pickle.dumps(Predictions(request=_request(), array=np.zeros((3, 8), dtype=np.float32)))
    )
    assert batch.tiles.shape == (3, 1, 2, 2)
    assert preds.array.shape == (3, 8)
    assert preds.request.tile_size == 2

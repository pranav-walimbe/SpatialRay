"""
Tests the pool deployment bodies compose stages and wrap model output without a Serve cluster.
"""

from __future__ import annotations

import asyncio
import threading

import numpy as np

from spatial_ray.serve.deployments import InferencePool, StagePool
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


def test_stage_pool_order():
    """StagePool applies its stages left to right."""
    log = []
    asyncio.run(StagePool(stages=(_mark("a", log), _mark("b", log))).run(_payload()))
    assert log == ["a", "b"]


def test_stage_pool_runs_concurrently():
    """A pool overlaps requests up to its own width and queues the surplus behind them."""
    width = 4
    barrier = threading.Barrier(width, timeout=5)
    lock = threading.Lock()
    live = 0
    peak = 0

    def stage(payload):
        nonlocal live, peak
        with lock:
            live += 1
            peak = max(peak, live)
        barrier.wait()
        with lock:
            live -= 1
        return payload

    pool = StagePool(stages=(stage,), max_concurrency=width)

    async def drive():
        return await asyncio.gather(*(pool.run(_payload()) for _ in range(2 * width)))

    try:
        assert len(asyncio.run(drive())) == 2 * width
    finally:
        pool.shutdown()
    # clearing a width-wide barrier proves that many overlap and the peak proves no more do
    assert peak == width


def test_inference_pool():
    """InferencePool wraps the model output in Predictions."""
    pool = InferencePool(model_factory=lambda: lambda tiles: tiles.sum(axis=(1, 2, 3)))
    payload = _payload()
    payload.tiles = np.ones((3, 1, 2, 2), dtype=np.float32)
    preds = asyncio.run(pool.infer(payload))
    assert preds.array.shape == (3,)
    assert preds.request is payload.request


def test_inference_pool_runs_concurrently():
    """The inference pool overlaps forward passes up to its own width and queues the surplus."""
    width = 4
    barrier = threading.Barrier(width, timeout=5)
    lock = threading.Lock()
    live = 0
    peak = 0

    def model(tiles):
        nonlocal live, peak
        with lock:
            live += 1
            peak = max(peak, live)
        barrier.wait()
        with lock:
            live -= 1
        return tiles.sum(axis=(1, 2, 3))

    pool = InferencePool(model_factory=lambda: model, max_concurrency=width)

    def _tiled():
        payload = _payload()
        payload.tiles = np.ones((3, 1, 2, 2), dtype=np.float32)
        return payload

    async def drive():
        return await asyncio.gather(*(pool.infer(_tiled()) for _ in range(2 * width)))

    try:
        assert len(asyncio.run(drive())) == 2 * width
    finally:
        pool.shutdown()
    # clearing a width-wide barrier proves that many overlap and the peak proves no more do
    assert peak == width

"""
Tests the pool deployment bodies compose stages and wrap model output without a Serve cluster.
"""

from __future__ import annotations

import asyncio
import threading
import time

import numpy as np

from spatial_ray.records import RequestRecord
from spatial_ray.scaling.ledger import PendingWorkSnapshot
from spatial_ray.scaling.metrics import (
    LEDGER_SNAPSHOT_AGE,
    executing_work_metric,
    pending_work_metric,
)
from spatial_ray.serve.deployments import InferencePool, Ingress, StagePool
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


class _RemoteMethod:
    def __init__(self, callback):
        self._callback = callback

    async def remote(self, *args):
        return self._callback(*args)


class _LedgerHandle:
    def __init__(self):
        self.events = []
        self.start = _RemoteMethod(lambda request_id, pool: self.events.append(("start", pool)))
        self.finish = _RemoteMethod(lambda request_id, pool: self.events.append(("finish", pool)))
        self.cancel = _RemoteMethod(lambda request_id: self.events.append(("cancel", request_id)))


def test_stage_pool_order():
    """StagePool applies its stages left to right."""
    log = []
    asyncio.run(StagePool(stages=(_mark("a", log), _mark("b", log))).run(_payload()))
    assert log == ["a", "b"]


def test_stage_pool_tracks_request_record_lifecycle():
    """A tracked stage transitions its exact work and preserves request identity."""
    ledger = _LedgerHandle()
    pool = StagePool(stages=(_mark("a", []),), pool_name="transform", ledger=ledger)
    result = asyncio.run(pool.run(RequestRecord(request_id="request-1", payload=_payload())))
    assert isinstance(result, RequestRecord)
    assert result.request_id == "request-1"
    assert ledger.events == [("start", "transform"), ("finish", "transform")]


def test_stage_pool_cancels_request_record_on_failure():
    """A failed stage clears every remaining pool entry through ledger cancellation."""
    ledger = _LedgerHandle()

    def fail(payload):
        raise RuntimeError("stage failed")

    pool = StagePool(stages=(fail,), pool_name="decode", ledger=ledger)
    try:
        asyncio.run(pool.run(RequestRecord(request_id="request-1", payload=_payload())))
    except RuntimeError:
        pass
    else:
        raise AssertionError("stage failure was swallowed")
    assert ledger.events == [("start", "decode"), ("cancel", "request-1")]


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


def test_inference_pool_tracks_request_record_lifecycle():
    """A tracked inference call removes its executing work after model completion."""
    ledger = _LedgerHandle()
    pool = InferencePool(
        model_factory=lambda: lambda tiles: tiles.sum(axis=(1, 2, 3)),
        ledger=ledger,
    )
    payload = _payload()
    payload.tiles = np.ones((3, 1, 2, 2), dtype=np.float32)
    preds = asyncio.run(pool.infer(RequestRecord(request_id="request-1", payload=payload)))
    assert preds.array.shape == (3,)
    assert ledger.events == [("start", "inference"), ("finish", "inference")]


def test_ingress_reports_cached_exact_ledger_snapshot():
    """The synchronous Ray callback reads the latest immutable ledger snapshot."""
    ingress = Ingress([], None, ledger=object())
    ingress._ledger_snapshot = PendingWorkSnapshot(
        sequence=4,
        captured_at_s=time.time(),
        pending={"decode": 120.0},
        executing={"decode": 30.0},
    )
    stats = ingress.record_autoscaling_stats()
    assert 0.0 <= stats[LEDGER_SNAPSHOT_AGE] < 1.0
    assert stats[pending_work_metric("decode")] == 120.0
    assert stats[executing_work_metric("decode")] == 30.0


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

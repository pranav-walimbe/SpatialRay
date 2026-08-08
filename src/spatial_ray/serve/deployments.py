"""
The Serve deployment classes for the disaggregated pipeline pools and the composing ingress.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

from spatial_ray.records import RequestRecord
from spatial_ray.scaling.ledger import PendingWorkSnapshot
from spatial_ray.scaling.meter import WorkEstimator, WorkloadMeter
from spatial_ray.scaling.metrics import (
    LEDGER_SNAPSHOT_AGE,
    executing_work_metric,
    pending_work_metric,
)
from spatial_ray.serve.messages import Predictions
from spatial_ray.workload.metadata import RasterPayload, RasterRequest
from spatial_ray.workload.profiler import Stage
from spatial_ray.workload.stages.decode import _DECODE_NUM_WORKERS, decode

logger = logging.getLogger(__name__)


class StagePool:
    def __init__(
        self,
        stages: Sequence[Stage],
        max_concurrency: int | None = None,
        *,
        pool_name: str | None = None,
        ledger=None,
    ) -> None:
        self._stages = tuple(stages)
        self._pool_name = pool_name
        self._ledger = ledger
        # decode is the one stage needing shared IO concurrency
        self._io_pool = (
            ThreadPoolExecutor(max_workers=_DECODE_NUM_WORKERS) if decode in self._stages else None
        )
        # a pool sized by the admission cap keeps Serve from throttling us to its num_cpus width
        self._stage_pool = (
            ThreadPoolExecutor(max_workers=max_concurrency) if max_concurrency is not None else None
        )

    async def run(
        self, item: RasterPayload | RequestRecord[RasterPayload]
    ) -> RasterPayload | RequestRecord[RasterPayload]:
        """Run this pool's stages in order on the payload, off the replica's event loop.

        Args:
            item: Payload or tracked request record entering the pool.

        Returns:
            The payload or tracked record after all stages have run.
        """
        record, payload = _split_record(item)
        if record is not None and self._ledger is not None:
            await self._ledger.start.remote(record.request_id, self._pool_name)
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(self._stage_pool, self._run_stages, payload)
        except BaseException:
            if record is not None and self._ledger is not None:
                await self._ledger.cancel.remote(record.request_id)
            raise
        if record is None or self._ledger is None:
            return result
        await self._ledger.finish.remote(record.request_id, self._pool_name)
        return RequestRecord(request_id=record.request_id, payload=result)

    def shutdown(self) -> None:
        """Release the shared IO and stage thread pools this pool created."""
        for pool in (self._io_pool, self._stage_pool):
            if pool is not None:
                pool.shutdown()

    def _run_stages(self, payload: RasterPayload) -> RasterPayload:
        # run the blocking stage chain in the pool's executor
        for stage in self._stages:
            payload = stage(payload, self._io_pool) if stage is decode else stage(payload)
        return payload


class InferencePool:
    def __init__(
        self,
        model_factory,
        max_concurrency: int | None = None,
        *,
        pool_name: str = "inference",
        ledger=None,
    ) -> None:
        self._model = model_factory()
        self._pool_name = pool_name
        self._ledger = ledger
        # a pool sized by the admission cap keeps Serve from throttling us to its num_cpus width
        self._infer_pool = (
            ThreadPoolExecutor(max_workers=max_concurrency) if max_concurrency is not None else None
        )

    async def infer(self, item: RasterPayload | RequestRecord[RasterPayload]) -> Predictions:
        """Run the model forward pass over a payload's tiles, off the replica's event loop.

        Args:
            item: Payload or tracked record with tiles set by preprocessing.

        Returns:
            Predictions carrying the model output for the payload.
        """
        record, payload = _split_record(item)
        if record is not None and self._ledger is not None:
            await self._ledger.start.remote(record.request_id, self._pool_name)
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(self._infer_pool, self._run_model, payload)
        except BaseException:
            if record is not None and self._ledger is not None:
                await self._ledger.cancel.remote(record.request_id)
            raise
        if record is not None and self._ledger is not None:
            await self._ledger.finish.remote(record.request_id, self._pool_name)
        return result

    def shutdown(self) -> None:
        """Release the forward-pass thread pool this pool created."""
        if self._infer_pool is not None:
            self._infer_pool.shutdown()

    def _run_model(self, payload: RasterPayload) -> Predictions:
        # run the blocking forward pass in the inference executor
        array = self._model(payload.tiles)
        return Predictions(request=payload.request, array=array)


class Ingress:
    def __init__(
        self,
        pools,
        inference,
        work_estimators: dict[str, WorkEstimator] | None = None,
        ledger=None,
        snapshot_refresh_s: float = 0.25,
    ) -> None:
        self._pools = pools
        self._inference = inference
        self._workload = WorkloadMeter(work_estimators) if work_estimators else None
        self._ledger = ledger
        self._ledger_snapshot: PendingWorkSnapshot | None = None
        self._snapshot_refresh_s = snapshot_refresh_s
        self._snapshot_task: asyncio.Task[None] | None = None
        self._metrics_valid = True

    async def __call__(self, request: RasterRequest) -> Predictions:
        """Compose the preprocessing pools then inference for one request.

        Args:
            request: Raster request entering the graph.

        Returns:
            Predictions produced for the request.
        """
        self._ensure_snapshot_refresh()
        record = None
        if self._workload is not None and self._ledger is not None:
            try:
                work = self._workload.record(request)
                record = RequestRecord(
                    request_id=uuid4().hex,
                    payload=RasterPayload(request=request),
                )
                await self._ledger.register.remote(record.request_id, work)
                self._metrics_valid = True
            except asyncio.CancelledError:
                if record is not None:
                    await self._cancel(record.request_id)
                raise
            except Exception:
                logger.exception("Pending-work registration failed, holding autoscaling metrics")
                self._metrics_valid = False
                record = None
        elif self._workload is not None:
            self._workload.record(request)
        item = record or RasterPayload(request=request)
        try:
            response = self._pools[0].run.remote(item)
            for pool in self._pools[1:]:
                response = pool.run.remote(response)
            return await self._inference.infer.remote(response)
        except BaseException:
            if record is not None:
                await self._cancel(record.request_id)
            raise

    def record_autoscaling_stats(self) -> dict[str, float]:
        """Report recent stage work arrival rates to Ray Serve.

        Returns:
            Custom autoscaling metrics keyed by stable SpatialRay metric names.
        """
        stats = self._workload.snapshot() if self._workload is not None else {}
        if not self._metrics_valid:
            return stats
        snapshot = self._ledger_snapshot
        if snapshot is None:
            return stats
        stats[LEDGER_SNAPSHOT_AGE] = max(0.0, time.time() - snapshot.captured_at_s)
        pools = snapshot.pending.keys() | snapshot.executing.keys()
        for pool in pools:
            stats[pending_work_metric(pool)] = snapshot.pending.get(pool, 0.0)
            stats[executing_work_metric(pool)] = snapshot.executing.get(pool, 0.0)
        return stats

    def _ensure_snapshot_refresh(self) -> None:
        # lazily start background refresh inside the replica's running event loop
        if self._ledger is not None and self._snapshot_task is None:
            self._snapshot_task = asyncio.create_task(self._refresh_ledger_snapshot())

    async def _refresh_ledger_snapshot(self) -> None:
        # keep one immutable latest snapshot for Ray's synchronous metrics callback
        while True:
            try:
                self._ledger_snapshot = await self._ledger.snapshot.remote()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Pending-work ledger snapshot refresh failed")
            await asyncio.sleep(self._snapshot_refresh_s)

    async def _cancel(self, request_id: str) -> None:
        # preserve the request error even if best-effort cleanup also fails
        try:
            await self._ledger.cancel.remote(request_id)
        except Exception:
            logger.exception("Pending-work ledger cancellation failed for %s", request_id)


def _split_record(
    item: RasterPayload | RequestRecord[RasterPayload],
) -> tuple[RequestRecord[RasterPayload] | None, RasterPayload]:
    # expose the payload to pool logic while retaining request identity for lifecycle events
    if isinstance(item, RequestRecord):
        return item, item.payload
    return None, item

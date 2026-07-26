"""
The Serve deployment classes for the disaggregated pipeline pools and the composing ingress.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext

from ray.serve import metrics

from spatial_ray.serve.messages import Predictions, TileBatch
from spatial_ray.workload.cost import decoded_bytes, predicted_tiles
from spatial_ray.workload.metadata import RasterPayload, RasterRequest
from spatial_ray.workload.profiler import Stage
from spatial_ray.workload.stages.decode import _DECODE_NUM_WORKERS, decode

# Maps a pool's work-unit label to the function weighting a payload by that unit
_WORK_WEIGHT: dict[str, Callable[[RasterPayload], float]] = {
    "bytes": lambda payload: float(decoded_bytes(payload.request)),
    "tiles": lambda payload: float(predicted_tiles(payload.request)),
}

_MEAN_BYTES_ALPHA = (
    0.05  # slow per-request EWMA so the mean tracks the workload regime not one request
)


class _MeanBytesGauge:
    """A per-replica EWMA gauge of the decoded bytes per request the decode pool sees."""

    def __init__(self) -> None:
        self._mean: float | None = None
        self._lock = threading.Lock()
        self._gauge = metrics.Gauge(
            "spatialray_mean_decoded_bytes",
            description="EWMA of the decoded array bytes per request this decode replica sees.",
        )

    def record(self, request: RasterRequest) -> None:
        """Fold this request's decoded byte size into the running mean and publish it.

        Args:
            request: Request whose AOI window and bands set its decoded byte size.
        """
        sample = float(decoded_bytes(request))
        with self._lock:
            if self._mean is None:
                self._mean = sample
            else:
                self._mean = _MEAN_BYTES_ALPHA * sample + (1.0 - _MEAN_BYTES_ALPHA) * self._mean
            self._gauge.set(self._mean)


class _WorkGauge:
    """A per-replica gauge holding the work units a pool has in flight."""

    def __init__(self, work_unit: str, weigh: Callable[..., float]) -> None:
        self._weigh = weigh
        self._in_flight = 0.0
        self._lock = threading.Lock()
        self._gauge = metrics.Gauge(
            "spatialray_work_in_flight",
            description="Work units in flight in this pool replica, weighting requests by size.",
            tag_keys=("work_unit",),
        )
        self._gauge.set_default_tags({"work_unit": work_unit})

    @contextmanager
    def track(self, item):
        """Raise the gauge by the item's work units for the duration of the block.

        Args:
            item: Payload or batch whose work units to weigh and hold in flight.

        Returns:
            A context manager raising the gauge on entry and lowering it on exit.
        """
        weight = self._weigh(item)
        self._add(weight)
        try:
            yield
        finally:
            self._add(-weight)

    def _add(self, delta: float) -> None:
        # Shift the in-flight counter under the lock and republish it
        with self._lock:
            self._in_flight += delta
            self._gauge.set(self._in_flight)


class StagePool:
    def __init__(self, stages: Sequence[Stage], work_unit: str | None = None) -> None:
        self._stages = tuple(stages)
        # decode is the one stage needing shared IO concurrency
        self._io_pool = (
            ThreadPoolExecutor(max_workers=_DECODE_NUM_WORKERS) if decode in self._stages else None
        )
        # work_unit is set only on the Serve path, the plain multiprocess harness leaves it None
        self._work = (
            _WorkGauge(work_unit, _WORK_WEIGHT[work_unit]) if work_unit is not None else None
        )
        # the byte-work pool also tracks its mean request size so decode can size on live bytes
        self._mean_bytes = _MeanBytesGauge() if work_unit == "bytes" else None

    def run(self, payload: RasterPayload) -> RasterPayload:
        """Run this pool's stages in order on the payload.

        Args:
            payload: Payload entering the pool.

        Returns:
            The payload after all of the pool's stages have run.
        """
        if self._mean_bytes is not None:
            self._mean_bytes.record(payload.request)
        with self._work.track(payload) if self._work is not None else nullcontext():
            for stage in self._stages:
                payload = stage(payload, self._io_pool) if stage is decode else stage(payload)
        return payload

    def shutdown(self) -> None:
        """Release this pool's shared IO thread pool, if it created one."""
        if self._io_pool is not None:
            self._io_pool.shutdown()


class InferencePool:
    def __init__(self, model_factory, work_unit: str | None = None) -> None:
        self._model = model_factory()
        # inference weighs work by the tiles already carried on the incoming batch
        self._work = (
            _WorkGauge(work_unit, lambda batch: float(batch.tiles.shape[0]))
            if work_unit is not None
            else None
        )

    def infer(self, batch: TileBatch) -> Predictions:
        """Run the model forward pass over a tile batch.

        Args:
            batch: Tile batch produced by the preprocessing pools.

        Returns:
            Predictions carrying the model output for the batch.
        """
        with self._work.track(batch) if self._work is not None else nullcontext():
            array = self._model(batch.tiles)
        return Predictions(request=batch.request, array=array)


class Ingress:
    def __init__(self, pools, inference) -> None:
        self._pools = pools
        self._inference = inference

    async def __call__(self, request: RasterRequest) -> Predictions:
        """Compose the preprocessing pools then inference for one request.

        Args:
            request: Raster request entering the graph.

        Returns:
            Predictions produced for the request.
        """
        payload = RasterPayload(request=request)
        for pool in self._pools:
            payload = await pool.run.remote(payload)
        batch = TileBatch(request=request, tiles=payload.tiles)
        return await self._inference.infer.remote(batch)

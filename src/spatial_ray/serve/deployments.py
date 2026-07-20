"""
The Serve deployment classes for the disaggregated pipeline pools and the composing ingress.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from spatial_ray.serve.messages import Predictions, TileBatch
from spatial_ray.workload.metadata import RasterPayload, RasterRequest
from spatial_ray.workload.profiler import Stage
from spatial_ray.workload.stages.decode import _DECODE_NUM_WORKERS, decode


class StagePool:
    def __init__(self, stages: Sequence[Stage]) -> None:
        self._stages = tuple(stages)
        # decode is the one stage needing shared IO concurrency, one pool per replica/process
        self._io_pool = (
            ThreadPoolExecutor(max_workers=_DECODE_NUM_WORKERS) if decode in self._stages else None
        )

    def run(self, payload: RasterPayload) -> RasterPayload:
        """Run this pool's stages in order on the payload.

        Args:
            payload: Payload entering the pool.

        Returns:
            The payload after all of the pool's stages have run.
        """
        for stage in self._stages:
            payload = stage(payload, self._io_pool) if stage is decode else stage(payload)
        return payload

    def shutdown(self) -> None:
        """Release this pool's shared IO thread pool, if it created one."""
        if self._io_pool is not None:
            self._io_pool.shutdown()


class InferencePool:
    def __init__(self, model_factory) -> None:
        self._model = model_factory()

    def infer(self, batch: TileBatch) -> Predictions:
        """Run the model forward pass over a tile batch.

        Args:
            batch: Tile batch produced by the preprocessing pools.

        Returns:
            Predictions carrying the model output for the batch.
        """
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

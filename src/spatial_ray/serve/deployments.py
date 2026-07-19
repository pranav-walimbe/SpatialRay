"""
The Serve deployment classes for the disaggregated pipeline pools and the composing ingress.
"""

from __future__ import annotations

from collections.abc import Sequence

from spatial_ray.serve.messages import Predictions, TileBatch
from spatial_ray.workload.metadata import RasterPayload, RasterRequest
from spatial_ray.workload.profiler import Stage


class StagePool:
    def __init__(self, stages: Sequence[Stage]) -> None:
        self._stages = tuple(stages)

    def run(self, payload: RasterPayload) -> RasterPayload:
        """Run this pool's stages in order on the payload.

        Args:
            payload: Payload entering the pool.

        Returns:
            The payload after all of the pool's stages have run.
        """
        for stage in self._stages:
            payload = stage(payload)
        return payload


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

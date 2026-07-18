"""
Shared harness that runs a request trace through the timed preprocessing stages.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from spatial_ray.workload.metadata import RasterPayload, TraceEntry
from spatial_ray.workload.profiler import Stage, StageCost, time_pipeline
from spatial_ray.workload.stages import PIPELINE


@dataclass(frozen=True)
class RequestTiming:
    scene_id: str  # scene the request read from
    n_tiles: int  # tiles produced by the pipeline
    costs: tuple[StageCost, ...]  # per-stage wall-clock and peak-RSS costs


def run_trace(
    trace: Sequence[TraceEntry],
    stages: Sequence[Stage] = PIPELINE,
) -> list[RequestTiming]:
    """Execute each request in the trace through the timed stages.

    Args:
        trace: Requests to process in arrival order.
        stages: Preprocessing stages to time, in execution order.

    Returns:
        One timing record per trace entry, in arrival order.
    """
    timings = []
    for entry in trace:
        payload = RasterPayload(request=entry.request)
        final, report = time_pipeline(payload, stages)
        timings.append(
            RequestTiming(
                scene_id=entry.request.scene.item_id,
                n_tiles=int(final.tiles.shape[0]),
                costs=report.costs,
            )
        )
    return timings

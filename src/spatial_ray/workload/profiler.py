"""
Stage-timer utility measuring per-stage wall-clock cost and peak-memory delta.
"""

from __future__ import annotations

import resource
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from spatial_ray.workload.metadata import RasterPayload

Stage = Callable[[RasterPayload], RasterPayload]

# ru_maxrss is bytes on macOS but kibibytes on Linux, so normalize to bytes
_RU_MAXRSS_TO_BYTES = 1 if sys.platform == "darwin" else 1024


@dataclass(frozen=True)
class StageCost:
    name: str  # stage function name
    wall_s: float  # wall-clock seconds spent in the stage
    rss_delta_b: int  # peak-RSS high-water-mark increase across the stage, in bytes


@dataclass(frozen=True)
class PipelineReport:
    costs: tuple[StageCost, ...]  # per-stage costs in execution order


def time_pipeline(
    payload: RasterPayload,
    stages: Sequence[Stage],
) -> tuple[RasterPayload, PipelineReport]:
    """Run each stage in sequence, timing wall-clock cost and peak-RSS delta.

    Args:
        payload: Initial payload fed into the first stage.
        stages: Stage functions to run in order on the payload.

    Returns:
        The final payload and a report of per-stage costs.
    """
    costs = []
    for stage in stages:
        rss_before = _peak_rss_bytes()
        start = time.perf_counter()
        payload = stage(payload)
        wall_s = time.perf_counter() - start
        rss_delta = _peak_rss_bytes() - rss_before
        costs.append(StageCost(name=stage.__name__, wall_s=wall_s, rss_delta_b=rss_delta))
    return payload, PipelineReport(costs=tuple(costs))


def _peak_rss_bytes() -> int:
    # Current process peak resident set size normalized to bytes
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * _RU_MAXRSS_TO_BYTES

"""
Aggregates per-request stage measurements into a per-stage time and memory characterization.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from perf.common.measure import RequestMeasurement

_BYTES_PER_MIB = 1024 * 1024


@dataclass(frozen=True)
class StageStats:
    name: str  # stage name
    mean_wall_s: float  # mean wall-clock across requests
    mean_rss_b: float  # mean subprocess peak RSS across requests, in bytes
    mean_vram_b: float  # mean peak CUDA memory across requests, in bytes
    wall_share: float  # this stage's share of total wall-clock
    device: str  # device the stage ran on, cpu or cuda


@dataclass(frozen=True)
class Summary:
    n_requests: int  # number of requests measured
    mean_tiles: float  # mean tiles produced per request
    mean_total_wall_s: float  # mean total wall-clock per request
    stages: tuple[StageStats, ...]  # per-stage statistics in execution order


def summarize(measurements: Sequence[RequestMeasurement]) -> Summary:
    """Aggregate per-request measurements into per-stage means and wall-clock shares.

    Args:
        measurements: Per-request measurement records from a substrate run.

    Returns:
        A characterization of measured per-stage and total cost.
    """
    stage_names = tuple(stage.name for stage in measurements[0].stages)
    count = len(stage_names)
    mean_wall = [statistics.fmean(m.stages[i].wall_s for m in measurements) for i in range(count)]
    mean_rss = [
        statistics.fmean(m.stages[i].rss_peak_b for m in measurements) for i in range(count)
    ]
    mean_vram = [
        statistics.fmean(m.stages[i].vram_peak_b for m in measurements) for i in range(count)
    ]
    total_wall = sum(mean_wall)
    devices = [measurements[0].stages[i].device for i in range(count)]
    stages = tuple(
        StageStats(
            name=name,
            mean_wall_s=wall,
            mean_rss_b=rss,
            mean_vram_b=vram,
            wall_share=wall / total_wall if total_wall else 0.0,
            device=device,
        )
        for name, wall, rss, vram, device in zip(
            stage_names, mean_wall, mean_rss, mean_vram, devices
        )
    )
    return Summary(
        n_requests=len(measurements),
        mean_tiles=statistics.fmean(m.n_tiles for m in measurements),
        mean_total_wall_s=total_wall,
        stages=stages,
    )


def format_summary(summary: Summary) -> str:
    """Render a characterization summary as a human-readable per-stage table.

    Args:
        summary: Aggregated characterization to render.

    Returns:
        A multi-line string with per-stage time, memory, shares, and the request total.
    """
    lines = [
        f"requests: {summary.n_requests}   mean tiles/request: {summary.mean_tiles:.1f}",
        "per-stage mean cost:",
    ]
    for stage in summary.stages:
        vram = "-" if stage.mean_vram_b <= 0 else f"{stage.mean_vram_b / _BYTES_PER_MIB:.1f}"
        lines.append(
            f"  {stage.name:<16} {stage.mean_wall_s * 1e3:8.1f} ms   "
            f"{stage.mean_rss_b / _BYTES_PER_MIB:7.1f} MiB peak  "
            f"{vram:>8} MiB vram   {stage.wall_share:5.1%}   {stage.device}"
        )
    lines.append(f"total: {summary.mean_total_wall_s * 1e3:.1f} ms/request")
    return "\n".join(lines)

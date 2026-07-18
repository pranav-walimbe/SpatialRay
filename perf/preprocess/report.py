"""
Aggregates per-request stage timings into a characterization of preprocessing cost.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from perf.common.harness import RequestTiming

_BYTES_PER_MIB = 1024 * 1024


@dataclass(frozen=True)
class StageStats:
    name: str  # stage function name
    mean_wall_s: float  # mean wall-clock across requests
    mean_rss_b: float  # mean peak-RSS delta across requests, in bytes
    wall_share: float  # this stage's share of total preprocessing wall-clock


@dataclass(frozen=True)
class CharacterizationSummary:
    n_requests: int  # number of requests measured
    mean_tiles: float  # mean tiles produced per request
    mean_total_wall_s: float  # mean total preprocessing wall-clock per request
    stages: tuple[StageStats, ...]  # per-stage statistics in execution order


def summarize(timings: Sequence[RequestTiming]) -> CharacterizationSummary:
    """Aggregate per-request timings into per-stage means and wall-clock shares.

    Args:
        timings: Per-request timing records from run_trace.

    Returns:
        A characterization of measured per-stage and total preprocessing cost.
    """
    stage_names = tuple(cost.name for cost in timings[0].costs)
    mean_wall = [
        statistics.fmean(timing.costs[i].wall_s for timing in timings)
        for i in range(len(stage_names))
    ]
    mean_rss = [
        statistics.fmean(timing.costs[i].rss_delta_b for timing in timings)
        for i in range(len(stage_names))
    ]
    total_wall = sum(mean_wall)
    stages = tuple(
        StageStats(
            name=name,
            mean_wall_s=wall,
            mean_rss_b=rss,
            wall_share=wall / total_wall if total_wall else 0.0,
        )
        for name, wall, rss in zip(stage_names, mean_wall, mean_rss)
    )
    return CharacterizationSummary(
        n_requests=len(timings),
        mean_tiles=statistics.fmean(timing.n_tiles for timing in timings),
        mean_total_wall_s=total_wall,
        stages=stages,
    )


def format_summary(summary: CharacterizationSummary) -> str:
    """Render a characterization summary as a human-readable per-stage table.

    Args:
        summary: Aggregated characterization to render.

    Returns:
        A multi-line string with per-stage costs, shares, and the request total.
    """
    lines = [
        f"requests: {summary.n_requests}   mean tiles/request: {summary.mean_tiles:.1f}",
        "per-stage mean cost:",
    ]
    for stage in summary.stages:
        lines.append(
            f"  {stage.name:<16} {stage.mean_wall_s * 1e3:8.1f} ms   "
            f"{stage.mean_rss_b / _BYTES_PER_MIB:7.1f} MiB   {stage.wall_share:5.1%}"
        )
    lines.append(f"total preprocessing: {summary.mean_total_wall_s * 1e3:.1f} ms/request")
    return "\n".join(lines)

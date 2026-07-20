"""
Stats types and a renderer for the concurrent load-test harness's per-stage and whole-run metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

_BYTES_PER_MIB = 1024 * 1024


@dataclass(frozen=True)
class StageLoadStats:
    name: str  # stage name
    device: str  # device the stage ran on, cpu or cuda
    n_requests: int  # requests this stage processed
    throughput_req_s: float  # requests completed per second over the stage's active wall time
    throughput_tiles_s: float | None  # tiles produced per second, None where tiles don't apply
    latency_mean_ms: float  # mean per-request stage latency
    latency_median_ms: float  # p50 per-request stage latency
    latency_p99_ms: float  # p99 per-request stage latency
    peak_rss_b: int  # peak resident set size of the stage process, in bytes


@dataclass(frozen=True)
class RunStats:
    n_requests: int  # requests fed into the run
    wall_s: float  # total wall-clock of the run
    cpu_util_mean: float  # mean system CPU utilization percent over the run
    cpu_util_peak: float  # peak system CPU utilization percent sampled during the run
    gpu_util_mean: float | None  # mean GPU utilization percent, None on cpu hardware
    gpu_util_peak: float | None  # peak GPU utilization percent, None on cpu hardware
    stages: tuple[StageLoadStats, ...]  # per-stage stats in pipeline order


def format_run_summary(run: RunStats) -> str:
    """Render a run's whole-run utilization and per-stage stats as a human-readable report.

    Args:
        run: Aggregated load-test run stats to render.

    Returns:
        A multi-line string with whole-run utilization and a per-stage stats table.
    """
    lines = [
        f"requests: {run.n_requests}   wall: {run.wall_s:.1f} s",
        f"cpu util: mean {run.cpu_util_mean:.1f}%   peak {run.cpu_util_peak:.1f}%",
    ]
    if run.gpu_util_mean is not None:
        lines.append(f"gpu util: mean {run.gpu_util_mean:.1f}%   peak {run.gpu_util_peak:.1f}%")
    lines.append("per-stage:")
    for stage in run.stages:
        tiles = "-" if stage.throughput_tiles_s is None else f"{stage.throughput_tiles_s:.1f}"
        lines.append(
            f"  {stage.name:<10} {stage.n_requests:5d} reqs   "
            f"{stage.throughput_req_s:6.2f} req/s   {tiles:>8} tiles/s   "
            f"latency mean {stage.latency_mean_ms:7.1f} ms  "
            f"p50 {stage.latency_median_ms:7.1f} ms  p99 {stage.latency_p99_ms:7.1f} ms   "
            f"peak {stage.peak_rss_b / _BYTES_PER_MIB:7.1f} MiB   {stage.device}"
        )
    return "\n".join(lines)

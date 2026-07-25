"""
Config loading and the metrics-figure renderer shared across the perf cloud harness.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from perf.cloud.harness import Report

_CONFIG_PATH = Path(__file__).parent / "config.yaml"
_GIB = 2**30
_MIB = 2**20


def load_config() -> dict:
    """Parse config.yaml, the shared source of truth for the disaggregated cluster shape.

    Returns:
        The parsed configuration mapping, including the per-pool cluster spec.
    """
    return yaml.safe_load(_CONFIG_PATH.read_text())


def save_report(report: Report, path) -> None:
    """Render the run's metrics into a six-panel figure and write it to path.

    Args:
        report: Run report holding the sampled metrics time-series and latency stats.
        path: Destination path for the PNG image.
    """
    plt.switch_backend("Agg")
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    throughput = report.n_requests / report.wall_s if report.wall_s else 0.0
    fig.suptitle(
        f"{report.model_name}  {report.hardware}  "
        f"{report.n_requests} reqs  {report.wall_s:.0f} s  {throughput:.2f} req/s"
    )
    times = [snapshot.t_s for snapshot in report.samples]
    _plot_cpu(axes[0][0], report, times)
    _plot_gpu(axes[0][1], report, times)
    _plot_queue(axes[1][0], report, times)
    _plot_work(axes[1][1], report, times)
    _plot_latency(axes[2][0], report)
    _plot_memory(axes[2][1], report, times)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_text_report(report: Report, path) -> None:
    """Render the run's metrics into aligned text tables and write them to path.

    Args:
        report: Run report holding the sampled metrics time-series and latency stats.
        path: Destination path for the text file.
    """
    throughput = report.n_requests / report.wall_s if report.wall_s else 0.0
    lines = [
        "=== SpatialRay perf report ===",
        f"model         {report.model_name}",
        f"hardware      {report.hardware}",
        f"requests      {report.n_requests}",
        f"wall_s        {report.wall_s:.2f}",
        f"throughput    {throughput:.2f} req/s",
        f"samples       {len(report.samples)}",
        "",
    ]
    # per-deployment latency straight from the Serve histogram
    latency_rows = [
        (
            deployment,
            str(stats["n_requests"]),
            f"{stats['latency_mean_ms']:.1f}",
            f"{stats['latency_p50_ms']:.1f}",
            f"{stats['latency_p99_ms']:.1f}",
        )
        for deployment, stats in sorted(report.latency.items())
    ]
    lines += _section(
        "per-stage latency (ms)", ("deployment", "n", "mean_ms", "p50_ms", "p99_ms"), latency_rows
    )
    lines += _reduced_section("cpu utilization (%)", report, "node_cpu", report.roles)

    # per-node gpu utilization alongside the vram it used
    gpu_rows = []
    for ip in _keys(report.samples, "node_gpu"):
        util = _reduce(report.samples, "node_gpu", ip)
        vram = _reduce(report.samples, "node_gram", ip, _GIB)
        if util is None and vram is None:
            continue
        util_cells = (f"{util[0]:.2f}", f"{util[1]:.2f}") if util else ("-", "-")
        vram_cells = (f"{vram[0]:.2f}", f"{vram[1]:.2f}") if vram else ("-", "-")
        gpu_rows.append((report.roles.get(ip, ip), *util_cells, *vram_cells))
    lines += _section(
        "gpu utilization (%) / vram (GiB)",
        ("node", "util_mean", "util_peak", "vram_mean", "vram_peak"),
        gpu_rows,
    )
    lines += _reduced_section("node memory (GiB)", report, "node_mem", report.roles, _GIB)
    lines += _reduced_section("queue depth (requests)", report, "queue", None)

    # per-pool work in flight labeled by unit with bytes pools reduced to MiB
    work_rows = []
    for deployment in _keys(report.samples, "work"):
        unit = report.work_units.get(deployment, "?")
        stats = _reduce(report.samples, "work", deployment, _MIB if unit == "bytes" else 1.0)
        if stats is None:
            continue
        display_unit = "MiB" if unit == "bytes" else unit
        work_rows.append((deployment, display_unit, f"{stats[0]:.2f}", f"{stats[1]:.2f}"))
    lines += _section("work in flight", ("pool", "unit", "mean", "peak"), work_rows)

    Path(path).write_text("\n".join(lines) + "\n")


def _reduced_section(title, report, field, roles, scale=1.0):
    # a mean and peak table for one per-node or per-pool time-series field
    rows = []
    for key in _keys(report.samples, field):
        stats = _reduce(report.samples, field, key, scale)
        if stats is None:
            continue
        label = roles.get(key, key) if roles else key
        rows.append((label, f"{stats[0]:.2f}", f"{stats[1]:.2f}"))
    return _section(title, ("name", "mean", "peak"), rows)


def _section(title, header, rows):
    # a titled block wrapping an aligned table or its no-data placeholder
    if not rows:
        return [f"--- {title} ---", "no data", ""]
    columns = [header, *rows]
    widths = [max(len(row[i]) for row in columns) for i in range(len(header))]
    aligned = [
        "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip() for row in columns
    ]
    return [f"--- {title} ---", *aligned, ""]


def _reduce(samples, field, key, scale=1.0):
    # mean and peak of one key's non-NaN values across the run or None when never present
    values = [value / scale for value in _series(samples, field, key) if not math.isnan(value)]
    if not values:
        return None
    return sum(values) / len(values), max(values)


def _plot_cpu(ax, report, times):
    # Per-node CPU utilization over the run, labeled by the stage each node hosts
    for ip in _keys(report.samples, "node_cpu"):
        ax.plot(times, _series(report.samples, "node_cpu", ip), label=report.roles.get(ip, ip))
    _finish(ax, "CPU utilization", "%", report.samples, "node_cpu")


def _plot_gpu(ax, report, times):
    # GPU utilization on the left axis and GPU memory used on a right twin axis
    for ip in _keys(report.samples, "node_gpu"):
        ax.plot(times, _series(report.samples, "node_gpu", ip), label=report.roles.get(ip, ip))
    twin = ax.twinx()
    for ip in _keys(report.samples, "node_gram"):
        gram = [value / _GIB for value in _series(report.samples, "node_gram", ip)]
        twin.plot(times, gram, linestyle="--")
    twin.set_ylabel("VRAM (GiB)")
    _finish(ax, "GPU utilization + VRAM", "%", report.samples, "node_gpu")


def _plot_queue(ax, report, times):
    # Queries in flight per pool, the native ongoing-requests saturation signal
    for deployment in _keys(report.samples, "queue"):
        ax.plot(times, _series(report.samples, "queue", deployment), label=deployment)
    _finish(ax, "queue depth", "requests", report.samples, "queue")


def _plot_work(ax, report, times):
    # Work in flight per pool, bytes pools on the left axis and tile pools on a right twin
    twin = ax.twinx()
    for deployment in _keys(report.samples, "work"):
        series = _series(report.samples, "work", deployment)
        if report.work_units.get(deployment) == "bytes":
            ax.plot(times, [value / _MIB for value in series], label=f"{deployment} (MiB)")
        else:
            twin.plot(times, series, linestyle="--", label=f"{deployment} (tiles)")
    twin.set_ylabel("tiles in flight")
    ax.set_ylabel("bytes in flight (MiB)")
    ax.set_xlabel("s")
    ax.set_title("work in flight")
    _merge_legends(ax, twin)


def _plot_latency(ax, report):
    # Grouped mean, p50, and p99 processing latency per deployment from the Serve histogram
    deployments = sorted(report.latency)
    if not deployments:
        _empty(ax, "per-stage latency")
        return
    x = np.arange(len(deployments))
    for offset, key, label in (
        (-0.25, "latency_mean_ms", "mean"),
        (0.0, "latency_p50_ms", "p50"),
        (0.25, "latency_p99_ms", "p99"),
    ):
        ax.bar(x + offset, [report.latency[d][key] for d in deployments], 0.25, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(deployments)
    ax.set_ylabel("ms")
    ax.set_title("per-stage latency")
    ax.legend(fontsize="x-small")


def _plot_memory(ax, report, times):
    # Per-node system memory used, the per-stage memory demand under one node per stage
    for ip in _keys(report.samples, "node_mem"):
        mem = [value / _GIB for value in _series(report.samples, "node_mem", ip)]
        ax.plot(times, mem, label=report.roles.get(ip, ip))
    _finish(ax, "node memory used", "GiB", report.samples, "node_mem")


def _keys(samples, field):
    # Sorted union of the dict keys seen for a snapshot field across the run
    keys = set()
    for snapshot in samples:
        keys |= set(getattr(snapshot, field))
    return sorted(keys)


def _series(samples, field, key):
    # A field's values for one key across every snapshot, NaN where the key is absent
    return [getattr(snapshot, field).get(key, float("nan")) for snapshot in samples]


def _finish(ax, title, ylabel, samples, field):
    # Apply the shared title, labels, and legend, or a placeholder when the panel has no series
    if not _keys(samples, field):
        _empty(ax, title)
        return
    ax.set_title(title)
    ax.set_xlabel("s")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize="x-small")


def _merge_legends(primary, twin):
    # Combine the legends of a twinned pair of axes into one
    handles, labels = primary.get_legend_handles_labels()
    twin_handles, twin_labels = twin.get_legend_handles_labels()
    if handles or twin_handles:
        primary.legend(handles + twin_handles, labels + twin_labels, fontsize="x-small")


def _empty(ax, title):
    # Mark a panel whose metric was not present in the scrape
    ax.set_title(title)
    ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)

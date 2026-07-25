"""
Config loading and the text metrics-report renderer shared across the perf cloud harness.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

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


def save_text_report(report: Report, path) -> None:
    """Render the run's metrics into aligned text tables and write them to path.

    Args:
        report: Run report holding the sampled metrics time-series and latency stats.
        path: Destination path for the text file.
    """
    throughput = report.n_requests / report.wall_s if report.wall_s else 0.0
    title = "SpatialRay perf report"
    bar = "═" * (len(title) + 4)
    lines = [
        bar,
        f"  {title}",
        bar,
        "",
        f"  model        {report.model_name}",
        f"  hardware     {report.hardware}",
        f"  requests     {report.n_requests}",
        f"  rate         {report.rate_per_s:.2f} req/s driven",
        f"  throughput   {throughput:.2f} req/s achieved",
        f"  wall         {report.wall_s:.2f} s",
        f"  samples      {len(report.samples)}",
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
        "Per-stage latency (ms)", ("deployment", "n", "mean_ms", "p50_ms", "p99_ms"), latency_rows
    )
    lines += _reduced_section("CPU utilization (%)", report, "node_cpu", report.roles)

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
        "GPU utilization (%) / VRAM (GiB)",
        ("node", "util_mean", "util_peak", "vram_mean", "vram_peak"),
        gpu_rows,
    )
    lines += _reduced_section("Node memory (GiB)", report, "node_mem", report.roles, _GIB)
    lines += _reduced_section("Queue depth (requests)", report, "queue", None)

    # per-pool work in flight labeled by unit with bytes pools reduced to MiB
    work_rows = []
    for deployment in _keys(report.samples, "work"):
        unit = report.work_units.get(deployment, "?")
        stats = _reduce(report.samples, "work", deployment, _MIB if unit == "bytes" else 1.0)
        if stats is None:
            continue
        display_unit = "MiB" if unit == "bytes" else unit
        work_rows.append((deployment, display_unit, f"{stats[0]:.2f}", f"{stats[1]:.2f}"))
    lines += _section("Work in flight", ("pool", "unit", "mean", "peak"), work_rows)

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    # a titled block wrapping a bordered table with left-aligned labels and right-aligned numbers
    if not rows:
        return [title, "  (no data)", ""]
    columns = [header, *rows]
    widths = [max(len(row[i]) for row in columns) for i in range(len(header))]

    def to_line(cells):
        inner = "│".join(
            f" {cell.ljust(widths[i]) if i == 0 else cell.rjust(widths[i])} "
            for i, cell in enumerate(cells)
        )
        return f"│{inner}│"

    seg = ["─" * (width + 2) for width in widths]
    top = "┌" + "┬".join(seg) + "┐"
    mid = "├" + "┼".join(seg) + "┤"
    bottom = "└" + "┴".join(seg) + "┘"
    return [title, top, to_line(header), mid, *(to_line(row) for row in rows), bottom, ""]


def _reduce(samples, field, key, scale=1.0):
    # mean and peak of one key's non-NaN values across the run or None when never present
    values = [value / scale for value in _series(samples, field, key) if not math.isnan(value)]
    if not values:
        return None
    return sum(values) / len(values), max(values)


def _keys(samples, field):
    # Sorted union of the dict keys seen for a snapshot field across the run
    keys = set()
    for snapshot in samples:
        keys |= set(getattr(snapshot, field))
    return sorted(keys)


def _series(samples, field, key):
    # A field's values for one key across every snapshot, NaN where the key is absent
    return [getattr(snapshot, field).get(key, float("nan")) for snapshot in samples]

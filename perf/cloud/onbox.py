"""
On-box entry that measures the fixed trace stage by stage and prints the per-stage report.
"""

from __future__ import annotations

import argparse

from perf.cloud.pipeline import measure_trace
from perf.common.models import DEFAULT_MODEL, load
from perf.common.report import format_summary, summarize
from perf.common.trace import build_default_trace


def main() -> None:
    """Measure the fixed trace stage by stage on the chosen hardware and print the summary."""
    parser = argparse.ArgumentParser(description="Subprocess-per-stage perf characterization.")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="model module under perf.common.models"
    )
    parser.add_argument(
        "--hardware",
        default="cpu",
        choices=("cpu", "gpu"),
        help="hardware target selecting the inference device",
    )
    args = parser.parse_args()
    model = load(args.model)
    trace = build_default_trace(model)
    measurements = measure_trace(trace, model=args.model, hardware=args.hardware)
    print(format_summary(summarize(measurements)))


if __name__ == "__main__":
    main()

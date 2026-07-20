"""
On-box entry that replays the load-test trace and prints the whole-run report.
"""

from __future__ import annotations

import argparse

from perf.cloud.harness import run
from perf.common.models import DEFAULT_MODEL
from perf.common.report import format_run_summary
from perf.common.trace import RATE_PER_S


def main() -> None:
    """Replay an n-request Poisson trace on the chosen hardware and print the run summary."""
    parser = argparse.ArgumentParser(description="Concurrent multiprocess load-test harness.")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="model module under perf.common.models"
    )
    parser.add_argument(
        "--hardware",
        default="cpu",
        choices=("cpu", "gpu"),
        help="hardware target selecting the inference device",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=500,
        help="number of requests the Poisson trace generates",
    )
    parser.add_argument(
        "--rate", type=float, default=RATE_PER_S, help="mean Poisson arrival rate in requests/s"
    )
    args = parser.parse_args()
    run_stats = run(
        model_name=args.model,
        hardware=args.hardware,
        n_requests=args.requests,
        rate_per_s=args.rate,
    )
    print(format_run_summary(run_stats))


if __name__ == "__main__":
    main()

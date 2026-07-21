"""
Drives the disaggregated Serve load test on the connected Ray cluster and writes the metrics figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from perf.cloud.harness import run
from perf.cloud.utils import save_report
from perf.common.models import DEFAULT_MODEL

_ASSETS_DIR = Path(__file__).parents[2] / "assets"


def main() -> None:
    """Drive an n-request Poisson trace through the Serve graph and write the metrics figure."""
    parser = argparse.ArgumentParser(description="Disaggregated Serve load-test harness.")
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
        "--requests", type=int, default=1000, help="number of requests the Poisson trace generates"
    )
    parser.add_argument(
        "--rate", type=float, default=1.0, help="mean Poisson arrival rate in requests/s"
    )
    parser.add_argument("--out", default=None, help="destination PNG path for the metrics figure")
    args = parser.parse_args()

    report = run(
        model_name=args.model,
        hardware=args.hardware,
        n_requests=args.requests,
        rate_per_s=args.rate,
    )
    out = Path(args.out) if args.out else _ASSETS_DIR / f"perf-{args.hardware}-{args.model}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_report(report, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

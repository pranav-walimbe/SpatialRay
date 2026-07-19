"""
Runs a trace through the disaggregated Serve graph and reports the model forward-pass time.
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Sequence

from ray import serve

from perf.common.models import DEFAULT_MODEL, load
from perf.common.trace import build_default_trace
from spatial_ray.serve.graph import InferenceSpec, build_graph
from spatial_ray.workload.metadata import RasterPayload, RasterRequest
from spatial_ray.workload.stages import PIPELINE

FORWARD_REPEATS = 5  # forward passes timed to average the isolated forward cost

HARDWARE_NUM_GPUS = {"cpu": 0, "gpu": 1}  # inference-pool GPU request per hardware target


def main() -> None:
    """Deploy the graph for the chosen model, run a trace through it, and report timings."""
    parser = argparse.ArgumentParser(description="Run a trace through the disaggregated graph.")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="model module under perf.common.models"
    )
    parser.add_argument(
        "--hardware",
        default="cpu",
        choices=tuple(HARDWARE_NUM_GPUS),
        help="hardware target that sets the inference pool GPU request",
    )
    args = parser.parse_args()
    model = load(args.model)
    trace = build_default_trace(model)
    inference = InferenceSpec(
        model_factory=model.build,
        ray_actor_options={"num_gpus": HARDWARE_NUM_GPUS[args.hardware]},
    )
    handle = serve.run(build_graph(inference=inference))
    latencies = [_request_s(handle, entry.request) for entry in trace]
    serve.shutdown()
    forward_ms = _forward_ms(model.build(), _tiles(trace[0].request))
    print(_summary(args.model, latencies, forward_ms))


def _request_s(handle, request: RasterRequest) -> float:
    # Send one request through the graph and return its end-to-end seconds
    start = time.perf_counter()
    handle.remote(request).result()
    return time.perf_counter() - start


def _tiles(request: RasterRequest):
    # Run the phase-1 stages locally to produce a real tile batch for the forward timing
    payload = RasterPayload(request=request)
    for stage in PIPELINE:
        payload = stage(payload)
    return payload.tiles


def _forward_ms(model, tiles) -> float:
    # Time the isolated model forward on a real tile batch in milliseconds per pass
    model(tiles)
    start = time.perf_counter()
    for _ in range(FORWARD_REPEATS):
        model(tiles)
    return (time.perf_counter() - start) / FORWARD_REPEATS * 1e3


def _summary(model_name: str, latencies: Sequence[float], forward_ms: float) -> str:
    # Render request-latency and isolated-forward stats as a short block
    ms = [s * 1e3 for s in latencies]
    return (
        f"model: {model_name}   requests: {len(latencies)}\n"
        f"request latency mean {statistics.fmean(ms):.1f} ms   p50 {statistics.median(ms):.1f} ms\n"
        f"isolated forward {forward_ms:.1f} ms/pass"
    )


if __name__ == "__main__":
    main()

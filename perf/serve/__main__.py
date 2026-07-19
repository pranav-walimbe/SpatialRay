"""
Runs a trace through the disaggregated Serve graph and reports the model forward-pass time.
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Sequence

from ray import serve

from perf.serve.models import DEFAULT_MODEL, load
from spatial_ray.serve.graph import build_graph
from spatial_ray.workload.catalog import resolve_scenes
from spatial_ray.workload.metadata import RasterPayload, RasterRequest
from spatial_ray.workload.stages import PIPELINE
from spatial_ray.workload.trace import build_trace

STAC_API_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"

# Fixed, low-cloud Sentinel-2 L2A scenes over one tile (10SEH, San Francisco Bay)
SCENE_IDS: tuple[str, ...] = (
    "S2A_10SEH_20230708_0_L2A",
    "S2B_10SEH_20230713_0_L2A",
    "S2A_10SEH_20230728_0_L2A",
    "S2A_10SEH_20230731_0_L2A",
)

WINDOW_SIZE = 1024  # native-pixel side length of each request's AOI window
TARGET_EPSG = 3857  # Web Mercator, the standard tile-serving CRS
TARGET_GSD = 10.0  # target ground sample distance in meters

RATE_PER_S = 1.0  # Poisson mean arrival rate
DURATION_S = 4.0  # arrival horizon
SEED = 0  # trace reproducibility seed
FORWARD_REPEATS = 5  # forward passes timed to average the isolated forward cost


def main() -> None:
    """Deploy the graph for the chosen model, run a trace through it, and report timings."""
    parser = argparse.ArgumentParser(description="Run a trace through the disaggregated graph.")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="model module under perf.serve.models"
    )
    args = parser.parse_args()
    model = load(args.model)
    scenes = resolve_scenes(
        SCENE_IDS, stac_api_url=STAC_API_URL, collection=COLLECTION, band_names=model.BAND_NAMES
    )
    trace = build_trace(
        scenes,
        rate_per_s=RATE_PER_S,
        duration_s=DURATION_S,
        window_size=WINDOW_SIZE,
        band_names=model.BAND_NAMES,
        target_epsg=TARGET_EPSG,
        target_gsd=TARGET_GSD,
        tile_size=model.TILE_SIZE,
        seed=SEED,
    )
    handle = serve.run(build_graph(model_factory=model.build))
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

"""
Runs the fixed scene set through the timed stages to characterize preprocessing performance.
"""

from __future__ import annotations

from perf.common.harness import run_trace
from perf.preprocess.report import format_summary, summarize
from spatial_ray.workload.catalog import resolve_scenes
from spatial_ray.workload.trace import build_trace

STAC_API_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"

# Fixed, low-cloud Sentinel-2 L2A scenes over one tile (10SEH, San Francisco Bay).
# These item ids are stable in the open sentinel-cogs bucket and pin the workload.
SCENE_IDS: tuple[str, ...] = (
    "S2A_10SEH_20230708_0_L2A",
    "S2B_10SEH_20230713_0_L2A",
    "S2A_10SEH_20230728_0_L2A",
    "S2A_10SEH_20230731_0_L2A",
)

# The four 10 m surface-reflectance bands used as the model input
BANDS: tuple[str, ...] = ("red", "green", "blue", "nir")

WINDOW_SIZE = 1024  # native-pixel side length of each request's AOI window
TARGET_EPSG = 3857  # Web Mercator, the standard tile-serving CRS
TARGET_GSD = 10.0  # target ground sample distance in meters
TILE_SIZE = 224  # model-input tile side length

RATE_PER_S = 1.0  # Poisson mean arrival rate
DURATION_S = 6.0  # arrival horizon
SEED = 0  # trace reproducibility seed


def main() -> None:
    """Resolve the fixed scenes, run the trace, and print the measured stage costs.

    Returns:
        None.
    """
    scenes = resolve_scenes(
        SCENE_IDS, stac_api_url=STAC_API_URL, collection=COLLECTION, band_names=BANDS
    )
    trace = build_trace(
        scenes,
        rate_per_s=RATE_PER_S,
        duration_s=DURATION_S,
        window_size=WINDOW_SIZE,
        band_names=BANDS,
        target_epsg=TARGET_EPSG,
        target_gsd=TARGET_GSD,
        tile_size=TILE_SIZE,
        seed=SEED,
    )
    print(format_summary(summarize(run_trace(trace))))


if __name__ == "__main__":
    main()

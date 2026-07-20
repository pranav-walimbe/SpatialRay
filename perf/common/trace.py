"""
Poisson trace generator plus the fixed Sentinel-2 default trace shared across the perf runs.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from types import ModuleType

from spatial_ray.workload.catalog import resolve_scenes
from spatial_ray.workload.metadata import RasterRequest, SceneRef


@dataclass(frozen=True)
class TraceEntry:
    arrival_s: float  # arrival timestamp in seconds since trace start
    request: RasterRequest  # request that arrives at `arrival_s`


def poisson_arrivals(rate_per_s: float, n: int, rng: random.Random) -> list[float]:
    """Draw a fixed number of Poisson-process arrival times.

    Args:
        rate_per_s: Mean arrival rate lambda in requests per second.
        n: Number of arrivals to draw.
        rng: Seeded random source for reproducibility.

    Returns:
        n ascending arrival timestamps starting from t=0.
    """
    arrivals = []
    clock = 0.0
    for _ in range(n):
        clock += rng.expovariate(rate_per_s)
        arrivals.append(clock)
    return arrivals


def build_trace(
    scenes: Sequence[SceneRef],
    *,
    rate_per_s: float,
    n: int,
    window_size: int,
    band_names: Sequence[str],
    target_epsg: int,
    target_gsd: float,
    tile_size: int,
    seed: int,
    normalize_mean: Sequence[float] | None = None,
    normalize_std: Sequence[float] | None = None,
) -> list[TraceEntry]:
    """Generate a Poisson trace of exactly n requests sampled uniformly over the scene set.

    Each arrival draws a scene uniformly and a random in-bounds AOI window, so
    burstiness comes from arrival timing rather than from scene skew.

    Args:
        scenes: Fixed scene set to sample requests from.
        rate_per_s: Mean arrival rate lambda in requests per second.
        n: Number of requests to generate.
        window_size: Side length in native pixels of the square AOI window.
        band_names: Ordered band names each request decodes.
        target_epsg: CRS every request reprojects into.
        target_gsd: Target ground sample distance in meters.
        tile_size: Side length of the square model-input tiles.
        seed: Seed for the arrival and sampling random source.
        normalize_mean: Per-band reflectance mean for standardization, or None to clip to [0, 1].
        normalize_std: Per-band reflectance std, paired with normalize_mean.

    Returns:
        n trace entries ordered by arrival time.
    """
    mean = tuple(normalize_mean) if normalize_mean is not None else None
    std = tuple(normalize_std) if normalize_std is not None else None
    rng = random.Random(seed)
    arrivals = poisson_arrivals(rate_per_s, n, rng)
    entries = []
    for arrival in arrivals:
        scene = rng.choice(scenes)
        window = _sample_window(scene, window_size, rng)
        request = RasterRequest(
            scene=scene,
            band_names=tuple(band_names),
            window=window,
            target_epsg=target_epsg,
            target_gsd=target_gsd,
            tile_size=tile_size,
            normalize_mean=mean,
            normalize_std=std,
        )
        entries.append(TraceEntry(arrival_s=arrival, request=request))
    return entries


def _sample_window(
    scene: SceneRef,
    size: int,
    rng: random.Random,
) -> tuple[int, int, int, int]:
    # Draw a random square window fully inside the scene's native grid
    rows, cols = scene.shape
    row_off = rng.randint(0, rows - size)
    col_off = rng.randint(0, cols - size)
    return row_off, col_off, size, size


STAC_API_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"

# Fixed sentinel-2 L2A scenes over one tile (10SEH, San Francisco Bay)
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
SEED = 0  # trace reproducibility seed


def build_default_trace(
    model: ModuleType, *, n: int, rate_per_s: float = RATE_PER_S
) -> list[TraceEntry]:
    """Resolve the fixed scenes and build an n-request Poisson trace for a model.

    Args:
        model: Loaded model module exposing BAND_NAMES, TILE_SIZE, and optional
            NORMALIZE_MEAN and NORMALIZE_STD.
        n: Number of requests to generate.
        rate_per_s: Mean Poisson arrival rate in requests per second.

    Returns:
        n trace entries in arrival order for the fixed scene set.
    """
    scenes = resolve_scenes(
        SCENE_IDS, stac_api_url=STAC_API_URL, collection=COLLECTION, band_names=model.BAND_NAMES
    )
    return build_trace(
        scenes,
        rate_per_s=rate_per_s,
        n=n,
        window_size=WINDOW_SIZE,
        band_names=model.BAND_NAMES,
        target_epsg=TARGET_EPSG,
        target_gsd=TARGET_GSD,
        tile_size=model.TILE_SIZE,
        seed=SEED,
        normalize_mean=getattr(model, "NORMALIZE_MEAN", None),
        normalize_std=getattr(model, "NORMALIZE_STD", None),
    )

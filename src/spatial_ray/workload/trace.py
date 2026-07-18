"""
Synthetic Poisson-arrival trace generator over a fixed scene set.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from spatial_ray.workload.metadata import RasterRequest, SceneRef, TraceEntry


def poisson_arrivals(rate_per_s: float, duration_s: float, rng: random.Random) -> list[float]:
    """Draw Poisson-process arrival times over a fixed horizon.

    Args:
        rate_per_s: Mean arrival rate lambda in requests per second.
        duration_s: Length of the arrival horizon in seconds.
        rng: Seeded random source for reproducibility.

    Returns:
        Ascending arrival timestamps within [0, duration_s).
    """
    arrivals = []
    clock = rng.expovariate(rate_per_s)
    while clock < duration_s:
        arrivals.append(clock)
        clock += rng.expovariate(rate_per_s)
    return arrivals


def build_trace(
    scenes: Sequence[SceneRef],
    *,
    rate_per_s: float,
    duration_s: float,
    window_size: int,
    band_names: Sequence[str],
    target_epsg: int,
    target_gsd: float,
    tile_size: int,
    seed: int,
) -> list[TraceEntry]:
    """Generate a Poisson trace of requests sampled uniformly over the scene set.

    Each arrival draws a scene uniformly and a random in-bounds AOI window, so
    burstiness comes from arrival timing rather than from scene skew.

    Args:
        scenes: Fixed scene set to sample requests from.
        rate_per_s: Mean arrival rate lambda in requests per second.
        duration_s: Length of the arrival horizon in seconds.
        window_size: Side length in native pixels of the square AOI window.
        band_names: Ordered band names each request decodes.
        target_epsg: CRS every request reprojects into.
        target_gsd: Target ground sample distance in meters.
        tile_size: Side length of the square model-input tiles.
        seed: Seed for the arrival and sampling random source.

    Returns:
        Trace entries ordered by arrival time.
    """
    rng = random.Random(seed)
    arrivals = poisson_arrivals(rate_per_s, duration_s, rng)
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

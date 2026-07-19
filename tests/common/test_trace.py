"""
Tests the Poisson arrival generator and trace builder over a fixed scene set.
"""

from __future__ import annotations

import random

from perf.common.trace import build_trace, poisson_arrivals
from spatial_ray.workload.metadata import BandProfile, SceneRef


def _scene(shape: tuple[int, int]) -> SceneRef:
    # Single-band scene over a square native grid
    band = BandProfile(
        name="red",
        href="red.tif",
        data_type="uint16",
        nodata=0.0,
        scale=1.0,
        offset=0.0,
        gsd=10.0,
    )
    return SceneRef(
        item_id="x",
        epsg=32610,
        shape=shape,
        transform=(10.0, 0.0, 0.0, 0.0, -10.0, 0.0),
        bands=(band,),
    )


def test_poisson_arrivals_are_sorted_within_horizon():
    """poisson_arrivals returns ascending timestamps inside the horizon."""
    arrivals = poisson_arrivals(2.0, 5.0, random.Random(0))
    assert arrivals == sorted(arrivals)
    assert all(0.0 <= a < 5.0 for a in arrivals)


def test_build_trace_samples_windows_in_bounds():
    """build_trace produces requests whose AOI windows fit the scene grid."""
    entries = build_trace(
        (_scene((100, 100)),),
        rate_per_s=5.0,
        duration_s=3.0,
        window_size=10,
        band_names=("red",),
        target_epsg=3857,
        target_gsd=10.0,
        tile_size=2,
        seed=0,
    )
    assert entries
    for entry in entries:
        row_off, col_off, height, width = entry.request.window
        assert (height, width) == (10, 10)
        assert 0 <= row_off <= 90
        assert 0 <= col_off <= 90

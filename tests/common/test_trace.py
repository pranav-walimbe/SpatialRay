"""
Tests the Poisson arrival generator and trace builder over a fixed scene set.
"""

from __future__ import annotations

import random

from perf.common.trace import build_trace, poisson_arrivals
from spatial_ray.workload.metadata import BandProfile, SceneRef

_TRACE_KWARGS = dict(
    rate_per_s=5.0,
    n=8,
    window_size=10,
    band_names=("red",),
    target_epsg=3857,
    target_gsd=10.0,
    tile_size=2,
    seed=0,
)


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


def test_poisson_arrivals():
    """Seed-reproducible, ascending, and tighter at a higher rate."""
    slow = poisson_arrivals(1.0, 200, random.Random(0))
    assert len(slow) == 200
    assert slow == sorted(slow)
    assert poisson_arrivals(1.0, 200, random.Random(0)) == slow
    fast = poisson_arrivals(10.0, 200, random.Random(0))
    assert fast[-1] < slow[-1]


def test_build_trace():
    """Yields n arrival-ordered in-grid requests, repeatable under a seed."""
    entries = build_trace((_scene((100, 100)),), **_TRACE_KWARGS)
    assert len(entries) == _TRACE_KWARGS["n"]
    assert [e.arrival_s for e in entries] == sorted(e.arrival_s for e in entries)
    for entry in entries:
        row_off, col_off, height, width = entry.request.window
        assert (height, width) == (10, 10)
        assert 0 <= row_off <= 90 and 0 <= col_off <= 90
    repeat = build_trace((_scene((100, 100)),), **_TRACE_KWARGS)
    assert [e.request.window for e in repeat] == [e.request.window for e in entries]

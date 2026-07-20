"""
Tests the CPU preprocessing stages transform payloads correctly without touching the network.
"""

from __future__ import annotations

import numpy as np
from affine import Affine
from rasterio.windows import Window

from spatial_ray.workload.metadata import BandProfile, RasterPayload, RasterRequest, SceneRef
from spatial_ray.workload.stages import normalize, reproject_stage, tile
from spatial_ray.workload.stages.decode import _align_to_blocks


def _band(
    name: str, *, scale: float = 1.0, offset: float = 0.0, nodata: float = 0.0
) -> BandProfile:
    # Band profile with the radiometry the normalize stage reads
    return BandProfile(
        name=name,
        href=f"{name}.tif",
        data_type="uint16",
        nodata=nodata,
        scale=scale,
        offset=offset,
        gsd=10.0,
    )


def _payload(bands: tuple[BandProfile, ...], *, tile_size: int = 2) -> RasterPayload:
    # Payload over a synthetic single-tile scene, array set by each test
    scene = SceneRef(
        item_id="x",
        epsg=32610,
        shape=(16, 16),
        transform=(10.0, 0.0, 0.0, 0.0, -10.0, 0.0),
        bands=bands,
    )
    request = RasterRequest(
        scene=scene,
        band_names=tuple(b.name for b in bands),
        window=(0, 0, 4, 4),
        target_epsg=3857,
        target_gsd=10.0,
        tile_size=tile_size,
    )
    return RasterPayload(request=request)


def test_normalize_scales_clips_and_zeros_nodata():
    """Scales digital numbers, clips to [0, 1], and zeros nodata pixels."""
    payload = _payload((_band("red", scale=0.0001),))
    payload.array = np.array([[[0, 10000, 20000]]], dtype=np.uint16)
    normalize(payload)
    assert payload.array.dtype == np.float32
    assert np.allclose(payload.array, np.array([[[0.0, 1.0, 1.0]]], dtype=np.float32))


def test_tile_cuts_row_major_blocks():
    """Cuts the array into row-major square tiles of the requested size."""
    payload = _payload((_band("red"),), tile_size=2)
    payload.array = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    tile(payload)
    assert payload.tiles.shape == (4, 1, 2, 2)
    assert np.array_equal(payload.tiles[0, 0], [[0, 1], [4, 5]])
    assert np.array_equal(payload.tiles[3, 0], [[10, 11], [14, 15]])


def test_align_to_blocks_is_a_no_op_for_an_already_aligned_window():
    """A window that already sits on block boundaries is returned unchanged."""
    window = Window(512, 1024, 512, 512)
    assert _align_to_blocks(window, (512, 512)) == window


def test_align_to_blocks_snaps_outward_to_the_block_grid():
    """An unaligned window is expanded outward to the nearest full blocks."""
    window = Window(600, 100, 200, 50)
    aligned = _align_to_blocks(window, (512, 512))
    assert (aligned.col_off, aligned.row_off) == (512, 0)
    assert (aligned.width, aligned.height) == (512, 512)


def test_reproject_changes_crs_and_preserves_bands():
    """reproject_stage warps into the target CRS while keeping the band count."""
    payload = _payload((_band("red"),))
    payload.array = np.ones((1, 16, 16), dtype=np.float32)
    payload.epsg = 32610
    payload.transform = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 4000000.0)
    reproject_stage(payload)
    assert payload.epsg == 3857
    assert payload.array.ndim == 3
    assert payload.array.shape[0] == 1

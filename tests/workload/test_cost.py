"""
Tests the static work estimators against the arrays the decode and tile stages actually produce.
"""

from __future__ import annotations

from spatial_ray.workload.cost import decoded_bytes, predicted_tiles
from spatial_ray.workload.metadata import BandProfile, RasterRequest, SceneRef


def _band(name: str, data_type: str) -> BandProfile:
    # Band over a 10 m native grid
    return BandProfile(
        name=name,
        href=f"{name}.tif",
        data_type=data_type,
        nodata=0.0,
        scale=1.0,
        offset=0.0,
        gsd=10.0,
    )


def _request(bands: tuple[BandProfile, ...], band_names: tuple[str, ...]) -> RasterRequest:
    # AOI request over a square native grid feeding both estimators
    scene = SceneRef(
        item_id="x",
        epsg=32610,
        shape=(512, 512),
        transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 4000000.0),
        bands=bands,
    )
    return RasterRequest(
        scene=scene,
        band_names=band_names,
        window=(0, 0, 256, 256),
        target_epsg=3857,
        target_gsd=10.0,
        tile_size=64,
    )


def test_decoded_bytes_sums_per_band_itemsize():
    """decoded_bytes multiplies the window area by the summed per-band itemsize."""
    request = _request((_band("red", "uint16"), _band("nir", "float32")), ("red", "nir"))
    assert decoded_bytes(request) == 256 * 256 * (2 + 4)


def test_predicted_tiles_is_positive_for_a_tiled_window():
    """predicted_tiles returns a positive whole-tile count for an in-bounds window."""
    request = _request((_band("red", "uint16"),), ("red",))
    assert predicted_tiles(request) > 0

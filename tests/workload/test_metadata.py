"""
Tests the workload metadata helpers index bands by their asset name.
"""

from __future__ import annotations

from spatial_ray.workload.metadata import BandProfile, SceneRef, band_map


def _band(name: str) -> BandProfile:
    # Minimal band profile with placeholder radiometry
    return BandProfile(
        name=name,
        href=f"{name}.tif",
        data_type="uint16",
        nodata=0.0,
        scale=1.0,
        offset=0.0,
        gsd=10.0,
    )


def test_band_map_indexes_bands_by_name():
    """band_map keys each band profile by its asset name."""
    scene = SceneRef(
        item_id="x",
        epsg=32610,
        shape=(10, 10),
        transform=(1.0,) * 6,
        bands=(_band("red"), _band("nir")),
    )
    mapping = band_map(scene)
    assert set(mapping) == {"red", "nir"}
    assert mapping["nir"].href == "nir.tif"

"""
Tests scene_from_item maps a STAC item's projection and raster fields into typed scene metadata.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pystac import Asset, Item

from spatial_ray.workload.catalog import scene_from_item


def _item() -> Item:
    # a STAC item with the proj and raster extension fields
    red = Asset(
        href="https://example.com/red.tif",
        extra_fields={
            "proj:shape": [10980, 10980],
            "proj:transform": [10.0, 0.0, 600000.0, 0.0, -10.0, 4200000.0],
            "raster:bands": [{"data_type": "uint16", "nodata": 0, "scale": 0.0001, "offset": -0.1}],
            "gsd": 10.0,
        },
    )
    item = Item(
        id="S2A_10SEH_20230708_0_L2A",
        geometry=None,
        bbox=None,
        datetime=datetime(2023, 7, 8, tzinfo=timezone.utc),
        properties={"proj:code": "EPSG:32610"},
    )
    item.add_asset("red", red)
    return item


def test_scene_from_item():
    """Reads the EPSG, native grid, and per-band radiometry from the item."""
    scene = scene_from_item(_item(), band_names=("red",))
    assert scene.item_id == "S2A_10SEH_20230708_0_L2A"
    assert scene.epsg == 32610
    assert scene.shape == (10980, 10980)
    assert scene.transform == (10.0, 0.0, 600000.0, 0.0, -10.0, 4200000.0)
    (band,) = scene.bands
    assert band.href == "https://example.com/red.tif"
    assert (band.data_type, band.scale, band.offset, band.gsd) == ("uint16", 0.0001, -0.1, 10.0)


def test_scene_from_item_epsg_fallbacks():
    """Falls back to the deprecated integer proj:epsg and to proj:code on the reference asset."""
    epsg_item = _item()
    epsg_item.properties = {"proj:epsg": 32610}
    assert scene_from_item(epsg_item, band_names=("red",)).epsg == 32610
    asset_item = _item()
    asset_item.properties = {}
    asset_item.assets["red"].extra_fields["proj:code"] = "EPSG:32610"
    assert scene_from_item(asset_item, band_names=("red",)).epsg == 32610

"""
Catalog-agnostic resolution of STAC items into typed scene metadata.
"""

from __future__ import annotations

from collections.abc import Sequence

from pystac import Item
from pystac_client import Client

from spatial_ray.workload.metadata import BandProfile, SceneRef


def resolve_scenes(
    item_ids: Sequence[str],
    *,
    stac_api_url: str,
    collection: str,
    band_names: Sequence[str],
    reference_band: str | None = None,
) -> tuple[SceneRef, ...]:
    """Resolve STAC item ids into typed scene metadata, preserving id order.

    Args:
        item_ids: STAC item ids to resolve.
        stac_api_url: Base URL of the STAC API to query.
        collection: Collection the items belong to.
        band_names: Ordered asset names to include as bands in each scene.
        reference_band: Asset defining the native grid, or None to use band_names[0].

    Returns:
        One SceneRef per id in item_ids, in that order.
    """
    client = Client.open(stac_api_url)
    search = client.search(collections=[collection], ids=list(item_ids))
    items = {item.id: item for item in search.items()}
    return tuple(scene_from_item(items[sid], band_names, reference_band) for sid in item_ids)


def scene_from_item(
    item: Item,
    band_names: Sequence[str],
    reference_band: str | None = None,
) -> SceneRef:
    """Build a SceneRef from a STAC item's projection and raster extension fields.

    Args:
        item: STAC item carrying proj: and raster: extension fields.
        band_names: Ordered asset names to include as bands.
        reference_band: Asset defining the native grid, or None to use band_names[0].

    Returns:
        Typed scene metadata for the item.
    """
    ref_name = reference_band if reference_band is not None else band_names[0]
    ref = item.assets[ref_name].extra_fields
    epsg = _parse_epsg(item.properties["proj:code"])
    shape = tuple(ref["proj:shape"])
    transform = tuple(ref["proj:transform"])
    bands = tuple(_band_profile(item, name) for name in band_names)
    return SceneRef(item_id=item.id, epsg=epsg, shape=shape, transform=transform, bands=bands)


def _band_profile(item: Item, name: str) -> BandProfile:
    # Build a BandProfile from a single asset's raster:bands entry and href
    asset = item.assets[name]
    raster = asset.extra_fields["raster:bands"][0]
    return BandProfile(
        name=name,
        href=asset.href,
        data_type=raster["data_type"],
        nodata=float(raster.get("nodata", 0.0)),
        scale=float(raster.get("scale", 1.0)),
        offset=float(raster.get("offset", 0.0)),
        gsd=float(asset.extra_fields["gsd"]),
    )


def _parse_epsg(proj_code: str) -> int:
    # Parse an integer EPSG code from a STAC proj:code string like "EPSG:32610"
    return int(proj_code.split(":")[1])

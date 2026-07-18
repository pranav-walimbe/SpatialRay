"""
Typed raster and request metadata grounded in the fixed scenes' STAC raster-extension fields.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from affine import Affine


@dataclass(frozen=True)
class BandProfile:
    name: str  # STAC asset key, e.g. "red"
    href: str  # COG asset href (s3/https), read remotely at request time
    data_type: str  # numpy dtype string from raster:bands, e.g. "uint16"
    nodata: float  # nodata sentinel from raster:bands
    scale: float  # reflectance scale factor from raster:bands
    offset: float  # reflectance offset from raster:bands
    gsd: float  # native ground sample distance in meters


@dataclass(frozen=True)
class SceneRef:
    item_id: str  # STAC item id, stable in the public sentinel-cogs bucket
    epsg: int  # native CRS parsed from the item's proj:code
    shape: tuple[int, int]  # native (rows, cols) of the reference band grid
    transform: tuple[float, ...]  # native affine, 6 coefficients
    bands: tuple[BandProfile, ...]  # per-band radiometric metadata and hrefs


@dataclass(frozen=True)
class RasterRequest:
    scene: SceneRef  # scene this request reads from
    band_names: tuple[str, ...]  # ordered subset of scene bands to decode
    window: tuple[int, int, int, int]  # AOI in native pixels: (row_off, col_off, height, width)
    target_epsg: int  # CRS to reproject into
    target_gsd: float  # target ground sample distance in meters
    tile_size: int  # side length of the square model-input tiles


@dataclass
class RasterPayload:
    request: RasterRequest  # immutable originating request
    array: np.ndarray | None = None  # (bands, rows, cols), None until decode
    epsg: int | None = None  # CRS of `array`: native after decode, target after reproject
    transform: Affine | None = None  # affine of `array`
    tiles: np.ndarray | None = None  # (n_tiles, bands, tile, tile), None until the tile stage


@dataclass(frozen=True)
class TraceEntry:
    arrival_s: float  # arrival timestamp in seconds since trace start
    request: RasterRequest  # request that arrives at `arrival_s`


def band_map(scene: SceneRef) -> dict[str, BandProfile]:
    """Index a scene's band profiles by asset name.

    Args:
        scene: Scene whose bands to index.

    Returns:
        Mapping from band name to its BandProfile.
    """
    return {b.name: b for b in scene.bands}

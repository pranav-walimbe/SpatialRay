"""
The four preprocessing stages as modular functions reading directly from remote COGs.
"""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rasterio.windows import Window

from spatial_ray.workload.metadata import RasterPayload, band_map

# GDAL configuration for efficient partial reads of public COGs over HTTP.
# Applied per read so the stages carry no process-global side effects.
_GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "AWS_NO_SIGN_REQUEST": "YES",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "VSI_CACHE": "TRUE",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
}


def decode(payload: RasterPayload) -> RasterPayload:
    """Read the request's AOI window from each remote band COG into a stacked array.

    Args:
        payload: Payload carrying the request, with array unset.

    Returns:
        The payload with array, epsg, and transform set to the native decode.
    """
    request = payload.request
    scene = request.scene
    bands = band_map(scene)
    row_off, col_off, height, width = request.window
    window = Window(col_off, row_off, width, height)
    stacked = []
    with rasterio.Env(**_GDAL_ENV):
        for name in request.band_names:
            with rasterio.open(bands[name].href) as src:
                stacked.append(src.read(1, window=window))
                transform = src.window_transform(window)
    payload.array = np.stack(stacked, axis=0)
    payload.epsg = scene.epsg
    payload.transform = transform
    return payload


def reproject_stage(payload: RasterPayload) -> RasterPayload:
    """Warp the decoded array into the request's target CRS at its target GSD.

    Args:
        payload: Payload with array, epsg, and transform set by decode.

    Returns:
        The payload with array, epsg, and transform updated to the target grid.
    """
    request = payload.request
    src_crs = CRS.from_epsg(payload.epsg)
    dst_crs = CRS.from_epsg(request.target_epsg)
    bands, src_h, src_w = payload.array.shape
    left, bottom, right, top = array_bounds(src_h, src_w, payload.transform)
    dst_transform, dst_w, dst_h = calculate_default_transform(
        src_crs, dst_crs, src_w, src_h, left, bottom, right, top, resolution=request.target_gsd
    )
    dst = np.empty((bands, dst_h, dst_w), dtype=payload.array.dtype)
    reproject(
        source=payload.array,
        destination=dst,
        src_transform=payload.transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
    )
    payload.array = dst
    payload.epsg = request.target_epsg
    payload.transform = dst_transform
    return payload


def normalize(payload: RasterPayload) -> RasterPayload:
    """Convert per-band digital numbers to clipped surface reflectance in [0, 1].

    Args:
        payload: Payload with array set to warped digital numbers.

    Returns:
        The payload with array replaced by a float32 reflectance array.
    """
    bands = band_map(payload.request.scene)
    out = np.empty(payload.array.shape, dtype=np.float32)
    for i, name in enumerate(payload.request.band_names):
        profile = bands[name]
        band = payload.array[i].astype(np.float32)
        nodata_mask = payload.array[i] == profile.nodata
        band = band * profile.scale + profile.offset
        np.clip(band, 0.0, 1.0, out=band)
        band[nodata_mask] = 0.0
        out[i] = band
    payload.array = out
    return payload


def tile(payload: RasterPayload) -> RasterPayload:
    """Cut the array into a batch of square tiles sized for the model input.

    Args:
        payload: Payload with array set to a normalized (bands, rows, cols) array.

    Returns:
        The payload with tiles set to a (n_tiles, bands, tile, tile) batch.
    """
    size = payload.request.tile_size
    bands, rows, cols = payload.array.shape
    n_rows, n_cols = rows // size, cols // size
    crop = payload.array[:, : n_rows * size, : n_cols * size]
    grid = crop.reshape(bands, n_rows, size, n_cols, size)
    payload.tiles = grid.transpose(1, 3, 0, 2, 4).reshape(-1, bands, size, size)
    return payload


# The stages in execution order, ready to wrap one-to-one in stage-2 Serve deployments
PIPELINE = (decode, reproject_stage, normalize, tile)

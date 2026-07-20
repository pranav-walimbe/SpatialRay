"""
CPU-bound transform stages: reproject, normalize, and tile.
"""

from __future__ import annotations

import numpy as np
from rasterio.crs import CRS
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject

from spatial_ray.workload.metadata import RasterPayload, band_map


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
    """Convert per-band digital numbers to a model-ready reflectance array.

    Args:
        payload: Payload with array set to warped digital numbers.

    Returns:
        The payload with array replaced by a float32 model-ready array.
    """
    request = payload.request
    bands = band_map(request.scene)
    mean, std = request.normalize_mean, request.normalize_std
    standardize = mean is not None and std is not None
    out = np.empty(payload.array.shape, dtype=np.float32)
    for i, name in enumerate(request.band_names):
        profile = bands[name]
        nodata_mask = payload.array[i] == profile.nodata
        band = payload.array[i].astype(np.float32) * profile.scale + profile.offset
        if standardize:
            band = (band - mean[i]) / std[i]
        else:
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

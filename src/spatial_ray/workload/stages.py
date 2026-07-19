"""
The four preprocessing stages as modular functions reading directly from remote COGs.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rasterio.windows import Window, from_bounds
from rasterio.windows import bounds as window_bounds
from rasterio.windows import transform as window_transform

from spatial_ray.workload.metadata import RasterPayload, band_map

# GDAL configuration for efficient partial reads of public COGs over HTTP.
# Applied per read so the stages carry no process-global side effects.
_GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "AWS_NO_SIGN_REQUEST": "YES",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "CPL_VSIL_CURL_CHUNK_SIZE": "1048576",
    "GDAL_CACHEMAX": "512",
    "VSI_CACHE": "TRUE",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
}


def _align_to_blocks(window: Window, block_shape: tuple[int, int]) -> Window:
    # Snap a pixel window outward to whole block_shape units so GDAL fetches only full tiles
    block_h, block_w = block_shape
    row_off = math.floor(window.row_off / block_h) * block_h
    col_off = math.floor(window.col_off / block_w) * block_w
    row_end = math.ceil((window.row_off + window.height) / block_h) * block_h
    col_end = math.ceil((window.col_off + window.width) / block_w) * block_w
    return Window(col_off, row_off, col_end - col_off, row_end - row_off)


def decode(payload: RasterPayload) -> RasterPayload:
    """Read the request's AOI from each remote band COG, resampling to the reference grid.

    Args:
        payload: Payload carrying the request, with array unset.

    Returns:
        The payload with array, epsg, and transform set to the reference-grid decode.
    """
    request = payload.request
    scene = request.scene
    bands = band_map(scene)
    row_off, col_off, height, width = request.window
    ref_transform = Affine(*scene.transform)
    ref_window = Window(col_off, row_off, width, height)
    aoi_bounds = window_bounds(ref_window, ref_transform)

    def _read_band(name: str) -> np.ndarray:
        # Read one band's AOI, block-aligning the fetch when its native grid matches the reference
        with rasterio.open(bands[name].href) as src:
            band_window = from_bounds(*aoi_bounds, transform=src.transform)
            same_resolution = math.isclose(src.transform.a, ref_transform.a) and math.isclose(
                src.transform.e, ref_transform.e
            )
            if not same_resolution:
                return src.read(
                    1,
                    window=band_window,
                    out_shape=(height, width),
                    resampling=Resampling.bilinear,
                )
            aligned = _align_to_blocks(band_window, src.block_shapes[0])
            array = src.read(1, window=aligned, resampling=Resampling.bilinear)
            row_start = round(band_window.row_off - aligned.row_off)
            col_start = round(band_window.col_off - aligned.col_off)
            return array[row_start : row_start + height, col_start : col_start + width]

    with (
        rasterio.Env(**_GDAL_ENV),
        ThreadPoolExecutor(max_workers=len(request.band_names)) as pool,
    ):
        stacked = list(pool.map(_read_band, request.band_names))
    payload.array = np.stack(stacked, axis=0)
    payload.epsg = scene.epsg
    payload.transform = window_transform(ref_window, ref_transform)
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
    """Convert per-band digital numbers to a model-ready reflectance array.

    With no per-request stats each band becomes surface reflectance clipped to [0, 1]. When the
    request carries normalize_mean and normalize_std the band is standardized to them, unclipped.

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


# The stages in execution order, ready to wrap one-to-one in stage-2 Serve deployments
PIPELINE = (decode, reproject_stage, normalize, tile)

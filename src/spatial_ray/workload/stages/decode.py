"""
Reads the request's AOI from each remote band COG, resampling to the reference grid.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rasterio
from affine import Affine
from rasterio.warp import Resampling
from rasterio.windows import Window, from_bounds
from rasterio.windows import bounds as window_bounds
from rasterio.windows import transform as window_transform

from spatial_ray.workload.metadata import RasterPayload, band_map

# GDAL configuration for efficient partial reads of public COGs over HTTP
_GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "AWS_NO_SIGN_REQUEST": "YES",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "CPL_VSIL_CURL_CHUNK_SIZE": "1048576",
    "GDAL_CACHEMAX": 512,
    "VSI_CACHE": "TRUE",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
}

_DECODE_NUM_WORKERS = 64  # shared IO thread pool width across all bands and blocks
_DECODE_PREFETCH = 8  # block reads submitted per wave


def _align_to_blocks(window: Window, block_shape: tuple[int, int]) -> Window:
    # Snap a pixel window outward to whole block_shape units so GDAL fetches only full tiles
    block_h, block_w = block_shape
    row_off = math.floor(window.row_off / block_h) * block_h
    col_off = math.floor(window.col_off / block_w) * block_w
    row_end = math.ceil((window.row_off + window.height) / block_h) * block_h
    col_end = math.ceil((window.col_off + window.width) / block_w) * block_w
    return Window(col_off, row_off, col_end - col_off, row_end - row_off)


def _block_windows(window: Window, block_shape: tuple[int, int]) -> list[Window]:
    # Split a block-aligned window into its constituent native block_shape tiles
    block_h, block_w = block_shape
    return [
        Window(col, row, block_w, block_h)
        for row in range(int(window.row_off), int(window.row_off + window.height), block_h)
        for col in range(int(window.col_off), int(window.col_off + window.width), block_w)
    ]


def decode(payload: RasterPayload, pool: ThreadPoolExecutor | None = None) -> RasterPayload:
    """Read the request's AOI from each remote band COG, resampling to the reference grid.

    Args:
        payload: Payload carrying the request, with array unset.
        pool: Shared IO thread pool to read blocks with. A persistent stage process should
            create one pool at startup and pass it in on every call, reusing it across requests.
            None auto-creates and closes a pool scoped to just this call.

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

    plans = {}  # band name -> (aligned window, band window), or None for a whole-band resample
    arrays = {}  # band name -> destination array, block reads land directly into this
    tasks = []  # (band name, href, read window, dest offset or None for a whole-band resample)

    with rasterio.Env(**_GDAL_ENV):
        for name in request.band_names:
            href = bands[name].href
            with rasterio.open(href) as src:
                band_window = from_bounds(*aoi_bounds, transform=src.transform)
                same_resolution = math.isclose(src.transform.a, ref_transform.a) and math.isclose(
                    src.transform.e, ref_transform.e
                )
                if not same_resolution:
                    plans[name] = None
                    tasks.append((name, href, band_window, None))
                    continue
                block_shape = src.block_shapes[0]
                aligned = _align_to_blocks(band_window, block_shape)
                plans[name] = (aligned, band_window)
                arrays[name] = np.empty(
                    (int(aligned.height), int(aligned.width)), dtype=src.dtypes[0]
                )
                for block in _block_windows(aligned, block_shape):
                    offset = (
                        int(block.row_off - aligned.row_off),
                        int(block.col_off - aligned.col_off),
                    )
                    tasks.append((name, href, block, offset))

        def _read(task):
            # Open a fresh handle per task, GDAL datasets are not safe to share across threads
            name, href, window, offset = task
            with rasterio.open(href) as src:
                if offset is None:
                    data = src.read(
                        1,
                        window=window,
                        out_shape=(height, width),
                        resampling=Resampling.bilinear,
                    )
                else:
                    data = src.read(1, window=window, resampling=Resampling.bilinear)
            return name, offset, data

        owns_pool = pool is None
        active_pool = pool or ThreadPoolExecutor(max_workers=_DECODE_NUM_WORKERS)
        try:
            for i in range(0, len(tasks), _DECODE_PREFETCH):
                for name, offset, data in active_pool.map(_read, tasks[i : i + _DECODE_PREFETCH]):
                    if offset is None:
                        arrays[name] = data
                    else:
                        r0, c0 = offset
                        arrays[name][r0 : r0 + data.shape[0], c0 : c0 + data.shape[1]] = data
        finally:
            if owns_pool:
                active_pool.shutdown()

    stacked = []
    for name in request.band_names:
        array = arrays[name]
        plan = plans[name]
        if plan is not None:
            aligned, band_window = plan
            row_start = round(band_window.row_off - aligned.row_off)
            col_start = round(band_window.col_off - aligned.col_off)
            array = array[row_start : row_start + height, col_start : col_start + width]
        stacked.append(array)

    payload.array = np.stack(stacked, axis=0)
    payload.epsg = scene.epsg
    payload.transform = window_transform(ref_window, ref_transform)
    return payload

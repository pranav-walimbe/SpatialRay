"""
Static work estimators mapping request metadata to per-stage work units.
"""

from __future__ import annotations

import numpy as np
from affine import Affine
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds

from spatial_ray.workload.metadata import RasterRequest, band_map


def decoded_bytes(request: RasterRequest) -> int:
    """Estimate the byte size of the array decode produces for a request.

    Args:
        request: Request whose AOI window and bands set the decoded array size.

    Returns:
        Bytes of the (bands, height, width) decoded array on the reference grid.
    """
    _, _, height, width = request.window
    bands = band_map(request.scene)
    itemsize = sum(np.dtype(bands[name].data_type).itemsize for name in request.band_names)
    return height * width * itemsize


def predicted_tiles(request: RasterRequest) -> int:
    """Predict the number of model tiles a request yields after reproject and tiling.

    Args:
        request: Request whose window, target CRS, GSD, and tile size set the tile count.

    Returns:
        Count of whole tile_size squares the reprojected array is cut into.
    """
    scene = request.scene
    row_off, col_off, height, width = request.window
    ref_transform = Affine(*scene.transform)
    ref_window = Window(col_off, row_off, width, height)
    left, bottom, right, top = window_bounds(ref_window, ref_transform)
    _, dst_w, dst_h = calculate_default_transform(
        CRS.from_epsg(scene.epsg),
        CRS.from_epsg(request.target_epsg),
        width,
        height,
        left,
        bottom,
        right,
        top,
        resolution=request.target_gsd,
    )
    size = request.tile_size
    return (dst_h // size) * (dst_w // size)

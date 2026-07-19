"""
Picklable deployment-boundary messages passed between the disaggregated Serve pools.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spatial_ray.workload.metadata import RasterRequest


@dataclass
class TileBatch:
    request: RasterRequest  # request the tiles were produced for
    tiles: np.ndarray  # (n_tiles, bands, tile, tile) preprocessed tile batch


@dataclass
class Predictions:
    request: RasterRequest  # request the predictions were produced for
    array: np.ndarray  # model output embeddings, (n_tiles, embed_dim)

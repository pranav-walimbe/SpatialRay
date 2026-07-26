"""
Picklable deployment-boundary messages passed between the disaggregated Serve pools.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spatial_ray.workload.metadata import RasterRequest


@dataclass
class Predictions:
    request: RasterRequest  # request the predictions were produced for
    array: np.ndarray  # model output embeddings, (n_tiles, embed_dim)

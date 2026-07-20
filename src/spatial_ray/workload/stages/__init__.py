"""
The four preprocessing stages as modular functions reading directly from remote COGs.
"""

from __future__ import annotations

from spatial_ray.workload.stages.decode import decode
from spatial_ray.workload.stages.transform import normalize, reproject_stage, tile

# The stages in execution order, ready to wrap one-to-one in stage-2 Serve deployments
PIPELINE = (decode, reproject_stage, normalize, tile)

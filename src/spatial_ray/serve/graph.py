"""
The configurable-granularity builder that maps a stage-to-pool grouping onto the Serve graph.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ray import serve

from spatial_ray.serve.deployments import InferencePool, Ingress, StagePool
from spatial_ray.workload.profiler import Stage
from spatial_ray.workload.stages import decode, normalize, reproject_stage, tile

DEFAULT_NUM_REPLICAS = 1
INFERENCE_NUM_GPUS = 0


@dataclass(frozen=True)
class PoolSpec:
    name: str  # Serve deployment name for the pool
    stages: tuple[Stage, ...]  # phase-1 stage functions the pool runs in order
    num_replicas: int  # static replica count for the pool


DISAGGREGATED: tuple[PoolSpec, ...] = (
    PoolSpec(name="decode", stages=(decode,), num_replicas=DEFAULT_NUM_REPLICAS),
    PoolSpec(
        name="transform",
        stages=(reproject_stage, normalize, tile),
        num_replicas=DEFAULT_NUM_REPLICAS,
    ),
)


def build_graph(
    grouping: Sequence[PoolSpec] = DISAGGREGATED,
    *,
    model_factory: Callable[[], object],
):
    """Bind the preprocessing pools and the inference pool into one Serve application.

    Args:
        grouping: Ordered pool specs mapping stage groups onto preprocessing pools.
        model_factory: Zero-arg factory building the inference model on each replica.

    Returns:
        The bound ingress application ready for serve.run or the serve run CLI.
    """
    pools = [
        serve.deployment(StagePool)
        .options(name=spec.name, num_replicas=spec.num_replicas)
        .bind(spec.stages)
        for spec in grouping
    ]
    inference = (
        serve.deployment(InferencePool)
        .options(
            name="inference",
            num_replicas=DEFAULT_NUM_REPLICAS,
            ray_actor_options={"num_gpus": INFERENCE_NUM_GPUS},
        )
        .bind(model_factory)
    )
    return serve.deployment(Ingress).options(name="ingress").bind(pools, inference)

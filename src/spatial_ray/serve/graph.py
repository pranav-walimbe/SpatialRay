"""
The configurable-granularity builder that maps a stage-to-pool grouping onto the Serve graph.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ray import serve

from spatial_ray.serve.deployments import InferencePool, Ingress, StagePool
from spatial_ray.workload.profiler import Stage
from spatial_ray.workload.stages import decode, normalize, reproject_stage, tile

DEFAULT_NUM_REPLICAS = 1


@dataclass(frozen=True)
class PoolSpec:
    name: str  # Serve deployment name for the pool
    stages: tuple[Stage, ...]  # phase-1 stage functions the pool runs in order
    num_replicas: int = DEFAULT_NUM_REPLICAS  # static replica count when not autoscaling
    max_ongoing_requests: int | None = None  # per-replica request cap, None keeps Serve's default
    ray_actor_options: Mapping[str, Any] = field(default_factory=dict)  # per-replica resources
    autoscaling_config: Mapping[str, Any] | None = None  # Serve autoscaling, overrides num_replicas


@dataclass(frozen=True)
class InferenceSpec:
    model_factory: Callable[[], object]  # zero-arg factory building the model on each replica
    name: str = "inference"  # Serve deployment name for the inference pool
    num_replicas: int = DEFAULT_NUM_REPLICAS  # static replica count when not autoscaling
    max_ongoing_requests: int | None = None  # per-replica request cap, None keeps Serve's default
    ray_actor_options: Mapping[str, Any] = field(default_factory=dict)  # per-replica resources
    autoscaling_config: Mapping[str, Any] | None = None  # Serve autoscaling, overrides num_replicas


DISAGGREGATED: tuple[PoolSpec, ...] = (
    PoolSpec(name="decode", stages=(decode,), max_ongoing_requests=64),
    PoolSpec(name="transform", stages=(reproject_stage, normalize, tile)),
)


def deployment_options(spec: PoolSpec | InferenceSpec) -> dict[str, Any]:
    """Render a pool or inference spec's shared fields as Serve deployment options.

    Args:
        spec: Pool or inference spec to render.

    Returns:
        Kwargs for Serve's .options(), and equally the shape of one serveConfigV2 deployment
        entry: name, ray_actor_options, an optional max_ongoing_requests, and either
        autoscaling_config or num_replicas.
    """
    options: dict[str, Any] = {
        "name": spec.name,
        "ray_actor_options": dict(spec.ray_actor_options),
    }
    if spec.max_ongoing_requests is not None:
        options["max_ongoing_requests"] = spec.max_ongoing_requests
    if spec.autoscaling_config is not None:
        options["autoscaling_config"] = dict(spec.autoscaling_config)
    else:
        options["num_replicas"] = spec.num_replicas
    return options


def build_graph(
    grouping: Sequence[PoolSpec] = DISAGGREGATED,
    *,
    inference: InferenceSpec,
):
    """Bind the preprocessing pools and the inference pool into one Serve application.

    Args:
        grouping: Ordered pool specs mapping stage groups onto preprocessing pools.
        inference: Spec for the inference pool, carrying its model factory and resources.

    Returns:
        The bound ingress application ready for serve.run or the serve run CLI.
    """
    pools = [
        serve.deployment(StagePool).options(**deployment_options(spec)).bind(spec.stages)
        for spec in grouping
    ]
    inference_pool = (
        serve.deployment(InferencePool)
        .options(**deployment_options(inference))
        .bind(inference.model_factory)
    )
    return serve.deployment(Ingress).options(name="ingress").bind(pools, inference_pool)

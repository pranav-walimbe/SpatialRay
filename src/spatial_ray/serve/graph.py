"""
The configurable-granularity builder that maps a stage-to-pool grouping onto the Serve graph.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ray import serve

from spatial_ray.serve.deployments import InferencePool, Ingress, StagePool
from spatial_ray.workload.cost import decoded_bytes, predicted_pixel_bands, predicted_tiles
from spatial_ray.workload.profiler import Stage
from spatial_ray.workload.stages import decode, normalize, reproject_stage, tile

DEFAULT_NUM_REPLICAS = 1

# zero writes the ongoing-request gauge inline rather than caching it on the replica's event loop
REPLICA_ENV_VARS = {"RAY_SERVE_METRICS_EXPORT_INTERVAL_MS": "0"}


@dataclass(frozen=True)
class PoolSpec:
    name: str  # Serve deployment name for the pool
    stages: tuple[Stage, ...]  # phase-1 stage functions the pool runs in order
    num_replicas: int = DEFAULT_NUM_REPLICAS  # static replica count when not autoscaling
    max_ongoing_requests: int | None = None  # per-replica request cap that None leaves to Serve
    max_concurrency: int | None = None  # stage chains one replica runs at once
    ray_actor_options: Mapping[str, Any] = field(default_factory=dict)  # per-replica resources
    autoscaling_config: Mapping[str, Any] | None = None  # Serve autoscaling overriding num_replicas
    work_unit: str | None = None  # work-in-flight gauge unit of bytes or tiles that None disables
    work_estimator: Callable[[Any], float] | None = None  # submitted request to pool work units


@dataclass(frozen=True)
class InferenceSpec:
    model_factory: Callable[[], object]  # zero-arg factory building the model on each replica
    name: str = "inference"  # Serve deployment name for the inference pool
    num_replicas: int = DEFAULT_NUM_REPLICAS  # static replica count when not autoscaling
    max_ongoing_requests: int | None = None  # per-replica request cap that None leaves to Serve
    max_concurrency: int | None = None  # forward passes one replica runs at once
    ray_actor_options: Mapping[str, Any] = field(default_factory=dict)  # per-replica resources
    autoscaling_config: Mapping[str, Any] | None = None  # Serve autoscaling overriding num_replicas
    work_unit: str | None = "tiles"  # work-in-flight gauge unit that None disables
    work_estimator: Callable[[Any], float] | None = predicted_tiles  # submitted request work


DISAGGREGATED: tuple[PoolSpec, ...] = (
    PoolSpec(name="decode", stages=(decode,), work_unit="bytes", work_estimator=decoded_bytes),
    PoolSpec(
        name="transform",
        stages=(reproject_stage, normalize, tile),
        work_unit="pixel_bands",
        work_estimator=predicted_pixel_bands,
    ),
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
        "ray_actor_options": {
            "runtime_env": {"env_vars": REPLICA_ENV_VARS},
            **spec.ray_actor_options,
        },
    }
    if spec.max_ongoing_requests is not None:
        options["max_ongoing_requests"] = spec.max_ongoing_requests
    if spec.autoscaling_config is not None:
        options["autoscaling_config"] = dict(spec.autoscaling_config)
    else:
        options["num_replicas"] = spec.num_replicas
    return options


def ingress_deployment_options(
    options: Mapping[str, Any] | None = None,
    *,
    fixed_autoscaling: bool = False,
) -> dict[str, Any]:
    """Render ingress options and optionally expose it as a fixed autoscaling context.

    Args:
        options: Serve options overriding the ingress defaults, none keeps them.
        fixed_autoscaling: Whether to express the replica count as a fixed autoscaling range.

    Returns:
        Kwargs for the ingress deployment's Serve options.
    """
    rendered = {"num_replicas": DEFAULT_NUM_REPLICAS, **dict(options or {})}
    rendered["ray_actor_options"] = {
        "runtime_env": {"env_vars": REPLICA_ENV_VARS},
        **rendered.get("ray_actor_options", {}),
    }
    if fixed_autoscaling:
        if "autoscaling_config" in rendered:
            raise ValueError(
                "ingress_options cannot set autoscaling_config with an application policy"
            )
        replicas = rendered.pop("num_replicas")
        rendered["autoscaling_config"] = {
            "min_replicas": replicas,
            "initial_replicas": replicas,
            "max_replicas": replicas,
        }
    return rendered


def build_graph(
    grouping: Sequence[PoolSpec] = DISAGGREGATED,
    *,
    inference: InferenceSpec,
    ingress_options: Mapping[str, Any] | None = None,
):
    """Bind the preprocessing pools and the inference pool into one Serve application.

    Args:
        grouping: Ordered pool specs mapping stage groups onto preprocessing pools.
        inference: Spec for the inference pool, carrying its model factory and resources.
        ingress_options: Serve options overriding the ingress defaults, none keeps them.

    Returns:
        The bound ingress application ready for serve.run or the serve run CLI.
    """
    pools = [
        serve.deployment(StagePool)
        .options(**deployment_options(spec))
        .bind(spec.stages, spec.work_unit, spec.max_concurrency)
        for spec in grouping
    ]
    inference_pool = (
        serve.deployment(InferencePool)
        .options(**deployment_options(inference))
        .bind(inference.model_factory, inference.work_unit, inference.max_concurrency)
    )
    options = ingress_deployment_options(ingress_options)
    ingress = serve.deployment(Ingress).options(name="ingress", **options)
    estimators = {
        spec.name: spec.work_estimator
        for spec in (*grouping, inference)
        if spec.work_estimator is not None
    }
    return ingress.bind(pools, inference_pool, estimators)

"""
Maps the cloud config onto the core detached controller that autoscales the deployed app's pools.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from perf.cloud.app import APP_NAME, sized_specs
from perf.cloud.utils import load_config
from spatial_ray.control.actor import launch_detached, stop_detached
from spatial_ray.control.bounds import PoolBounds
from spatial_ray.control.controller import DEFAULT_TICK_S
from spatial_ray.policy.dynamic import DynamicPolicy, disaggregated_dynamic_policy
from spatial_ray.serve.serve_config import compile_serve_config

_IMPORT_PATH = "perf.cloud.app:app"


def pool_bounds(pools_cfg: Mapping[str, Any]) -> dict[str, PoolBounds]:
    """Map each pool's config replica budget onto the controller's clamp bounds.

    Args:
        pools_cfg: The pools section of the cloud config, one entry per pool.

    Returns:
        Pool name to its min and max replica bounds.
    """
    return {
        name: PoolBounds(min_replicas=cfg["min_replicas"], max_replicas=cfg["max_replicas"])
        for name, cfg in pools_cfg.items()
    }


def build_policy(controller_cfg: Mapping[str, Any]) -> DynamicPolicy:
    """Map the controller config's signal capacities and setpoints onto the dynamic policy.

    Args:
        controller_cfg: The controller section of the cloud config.

    Returns:
        A dynamic policy pairing each pool with its own bottleneck signal.
    """
    return disaggregated_dynamic_policy(
        decode_bytes_per_replica=controller_cfg["decode_bytes_per_replica"],
        inference_queue_per_replica=controller_cfg["inference_queue_per_replica"],
        transform_util_target=controller_cfg["transform_util_target"],
        inference_util_target=controller_cfg["inference_util_target"],
    )


def start_controller(model_name: str, hardware: str):
    """Start the detached controller autoscaling the already-deployed app's pools.

    Args:
        model_name: Model module name under perf.common.models for the inference pool.
        hardware: Target hardware, cpu or gpu, selecting the inference replica's device.

    Returns:
        The running detached controller actor handle.
    """
    config = load_config()
    pools, inference = sized_specs(model_name, hardware)
    return launch_detached(
        build_policy(config["controller"]),
        pool_bounds(config["pools"]),
        compile_serve_config(pools, inference, import_path=_IMPORT_PATH, app_name=APP_NAME),
        app_name=APP_NAME,
        tick_s=config["controller"].get("tick_s", DEFAULT_TICK_S),
    )


def stop_controller() -> None:
    """Stop and remove the detached controller actor if it is running."""
    stop_detached()

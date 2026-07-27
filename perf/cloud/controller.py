"""
Maps the cloud config onto the core detached controller that autoscales the deployed app's pools.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from perf.cloud.app import from_config
from perf.cloud.utils import load_config
from spatial_ray.control.actor import launch_detached, stop_detached
from spatial_ray.control.bounds import PoolBounds
from spatial_ray.control.controller import DEFAULT_TICK_S
from spatial_ray.policy.dynamic import disaggregated_dynamic_policy, inference_queue_setpoint
from spatial_ray.policy.interfaces import Policy
from spatial_ray.policy.types import Action, Observation
from spatial_ray.serve.application import Application


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


@dataclass(frozen=True)
class _LoggingPolicy:
    """A policy wrapper printing each tick's per-pool util and proposed target for debugging."""

    inner: Policy  # wrapped policy whose decision is logged then returned

    def decide(self, observation: Observation) -> Action:
        """Log each pool's util, replicas, and proposed target then return the inner decision.

        Args:
            observation: The latest per-pool signals to decide from.

        Returns:
            The action the wrapped policy chose for this observation.
        """
        action = self.inner.decide(observation)
        parts = []
        for name, pool in observation.pools.items():
            want = action.targets.get(name)
            parts.append(f"{name}(util={pool.utilization:.2f} rep={pool.replicas} want={want})")
        print("[controller] " + " ".join(parts))
        return action


def build_policy(controller_cfg: Mapping[str, Any], application: Application) -> Policy:
    """Map the controller config and application specs onto the dynamic policy.

    Args:
        controller_cfg: The controller section of the cloud config.
        application: The deployed application whose pool specs set the decode and queue setpoints.

    Returns:
        A logging-wrapped dynamic policy pairing each pool with its own bottleneck signal.
    """
    decode = next(spec for spec in application.grouping if spec.name == "decode")
    policy = disaggregated_dynamic_policy(
        decode_max_ongoing_requests=decode.max_ongoing_requests,
        inference_queue_per_replica=inference_queue_setpoint(
            application.inference.max_ongoing_requests
        ),
        transform_util_target=controller_cfg["transform_util_target"],
        inference_util_target=controller_cfg["inference_util_target"],
    )
    return _LoggingPolicy(inner=policy)


def start_controller(model_name: str, hardware: str):
    """Start the detached controller autoscaling the already-deployed app's pools.

    Args:
        model_name: Model module name under perf.common.models for the inference pool.
        hardware: Target hardware, cpu or gpu, selecting the inference replica's device.

    Returns:
        The running detached controller actor handle.
    """
    config = load_config()
    application = from_config(model_name, hardware)
    return launch_detached(
        build_policy(config["controller"], application),
        pool_bounds(config["pools"]),
        application.serve_config,
        app_name=application.app_name,
        tick_s=config["controller"].get("tick_s", DEFAULT_TICK_S),
    )


def stop_controller() -> None:
    """Stop and remove the detached controller actor if it is running."""
    stop_detached()

"""
The detached controller host building the observe-decide-apply loop and running it on a thread.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import Any

import ray

from spatial_ray.control.bounds import PoolBounds
from spatial_ray.control.controller import DEFAULT_TICK_S, Controller
from spatial_ray.control.ray_actuator import RayActuator
from spatial_ray.control.ray_source import ray_observation_source
from spatial_ray.policy.interfaces import Policy

_STOP_JOIN_TIMEOUT_S = 30.0
DEFAULT_ACTOR_NAME = "spatialray-controller"


class ControllerHost:
    """Builds the control loop from picklable parts and runs it detached on a daemon thread."""

    def __init__(
        self,
        policy: Policy,
        bounds: Mapping[str, PoolBounds],
        actuator_config: dict[str, Any],
        *,
        app_name: str = "spatialray",
        util_kinds: Mapping[str, str | None] | None = None,
        tick_s: float = DEFAULT_TICK_S,
    ) -> None:
        # build the loop cluster-side so no live source, actuator, or lock is ever pickled to here
        source = (
            ray_observation_source(app_name=app_name)
            if util_kinds is None
            else ray_observation_source(util_kinds, app_name=app_name)
        )
        actuator = RayActuator(actuator_config)
        self._controller = Controller(source, policy, actuator, bounds, tick_s=tick_s)
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the control loop on a daemon thread and return at once, ignoring a second call."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._controller.run, name="spatialray-controller", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the loop to stop after its current tick and wait for the thread to exit."""
        self._controller.stop()
        if self._thread is not None:
            self._thread.join(timeout=_STOP_JOIN_TIMEOUT_S)
            self._thread = None


def launch_detached(
    policy: Policy,
    bounds: Mapping[str, PoolBounds],
    actuator_config: dict[str, Any],
    *,
    app_name: str = "spatialray",
    util_kinds: Mapping[str, str | None] | None = None,
    tick_s: float = DEFAULT_TICK_S,
    actor_name: str = DEFAULT_ACTOR_NAME,
):
    """Spawn the controller host as a named detached actor and start its loop ticking.

    Args:
        policy: The decision policy the loop applies each tick.
        bounds: Per-pool replica clamp bounds the controller reconciles within.
        actuator_config: The serveConfigV2 the actuator re-submits to rescale replicas.
        app_name: Name of the running Serve application to observe and rescale.
        util_kinds: Utilization source per pool, defaulting to the disaggregated grouping.
        tick_s: Control loop period in seconds.
        actor_name: Name the detached actor is registered under.

    Returns:
        The running detached controller actor handle.
    """
    host = ray.remote(ControllerHost).options(
        name=actor_name, lifetime="detached", get_if_exists=True
    )
    actor = host.remote(
        policy, bounds, actuator_config, app_name=app_name, util_kinds=util_kinds, tick_s=tick_s
    )
    actor.start.remote()
    return actor


def stop_detached(actor_name: str = DEFAULT_ACTOR_NAME) -> None:
    """Stop and remove the detached controller actor if it is registered.

    Args:
        actor_name: Name the detached actor was registered under.
    """
    try:
        actor = ray.get_actor(actor_name)
    except ValueError:
        return
    ray.get(actor.stop.remote())
    ray.kill(actor)

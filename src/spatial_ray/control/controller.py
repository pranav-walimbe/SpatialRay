"""
The control loop owning the clock and tying observe to decide to apply with anti-flap clamping.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping

from spatial_ray.control.bounds import PoolBounds
from spatial_ray.policy.interfaces import Actuator, ObservationSource, Policy
from spatial_ray.policy.types import Action, Observation

DEFAULT_TICK_S = 12.0
DEFAULT_DEADBAND = 0.10
DEFAULT_SCALE_UP_COOLDOWN_S = 0.0
DEFAULT_SCALE_DOWN_COOLDOWN_S = 60.0


class Controller:
    def __init__(
        self,
        source: ObservationSource,
        policy: Policy,
        actuator: Actuator,
        bounds: Mapping[str, PoolBounds],
        *,
        tick_s: float = DEFAULT_TICK_S,
        deadband: float = DEFAULT_DEADBAND,
        scale_up_cooldown_s: float = DEFAULT_SCALE_UP_COOLDOWN_S,
        scale_down_cooldown_s: float = DEFAULT_SCALE_DOWN_COOLDOWN_S,
    ) -> None:
        self._source = source
        self._policy = policy
        self._actuator = actuator
        self._bounds = dict(bounds)
        self._tick_s = tick_s
        self._deadband = deadband
        self._scale_up_cooldown_s = scale_up_cooldown_s
        self._scale_down_cooldown_s = scale_down_cooldown_s
        self._last_change_t: dict[str, float] = {}
        self._stop = threading.Event()

    def tick(self) -> Action:
        """Run one observe then decide then reconcile then apply cycle.

        Returns:
            The reconciled action applied or the raw proposal while a pool is still cold-starting.
        """
        observation = self._source.observe()
        proposed = self._policy.decide(observation)
        if not self._ready(observation):
            return proposed
        applied = self.reconcile(observation, proposed)
        self._actuator.apply(applied)
        return applied

    def _ready(self, observation: Observation) -> bool:
        # a pool below its replica floor is still cold-starting so hold off scaling until it is up
        for name, bounds in self._bounds.items():
            pool = observation.pools.get(name)
            if pool is None or pool.replicas < bounds.min_replicas:
                return False
        return True

    def reconcile(self, observation: Observation, proposed: Action) -> Action:
        """Clamp a policy's raw targets through the deadband, step cap, cooldowns, and bounds.

        Args:
            observation: The observation decided from, holding live replica counts and t_s.
            proposed: The policy's raw per-pool replica targets.

        Returns:
            An action whose targets honor each pool's anti-flap tuning and hard budget bounds.
        """
        targets: dict[str, int] = {}
        for name, desired in proposed.targets.items():
            current = observation.pools[name].replicas
            targets[name] = self._reconcile_pool(name, current, desired, observation.t_s)
        return Action(targets=targets)

    def run(self) -> None:
        """Loop observe then decide then apply on the tick cadence until stopped."""
        self._stop.clear()
        while not self._stop.is_set():
            self.tick()
            self._stop.wait(self._tick_s)

    def stop(self) -> None:
        """Signal the run loop to exit after its current tick."""
        self._stop.set()

    def _reconcile_pool(self, name: str, current: int, desired: int, t_s: float) -> int:
        # apply the deadband, per-tick step cap, direction cooldowns, and hard clamp in order
        bounds = self._bounds[name]
        if current > 0 and abs(desired - current) / current < self._deadband:
            return current
        delta = max(-bounds.max_step, min(bounds.max_step, desired - current))
        proposed = bounds.clamp(current + delta)
        if proposed == current:
            return current
        if not self._cooldown_ready(name, proposed > current, t_s):
            return current
        self._last_change_t[name] = t_s
        return proposed

    def _cooldown_ready(self, name: str, scaling_up: bool, t_s: float) -> bool:
        # scale up fast and scale down slow, gating each direction on its own cooldown
        last = self._last_change_t.get(name)
        if last is None:
            return True
        cooldown = self._scale_up_cooldown_s if scaling_up else self._scale_down_cooldown_s
        return t_s - last >= cooldown

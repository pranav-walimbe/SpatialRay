"""
The three protocols that make the simulator and the Ray deployment interchangeable.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from spatial_ray.policy.types import Action, Observation


@runtime_checkable
class Policy(Protocol):
    """A stateless mapping from an observation to a per-pool replica action."""

    def decide(self, observation: Observation) -> Action:
        """Choose each pool's desired replica count from the current observation.

        Args:
            observation: The latest per-pool signals and system arrival rate.

        Returns:
            An action carrying the desired num_replicas for each pool.
        """
        ...


@runtime_checkable
class ObservationSource(Protocol):
    """The read side, scraping live signals into an observation each tick."""

    def observe(self) -> Observation:
        """Read the current per-pool signals into one observation.

        Returns:
            An observation snapshot for the policy to decide on.
        """
        ...


@runtime_checkable
class Actuator(Protocol):
    """The write side, applying a policy's replica targets to the pools."""

    def apply(self, action: Action) -> None:
        """Apply the action's per-pool replica targets.

        Args:
            action: The desired num_replicas for each pool to reconcile toward.
        """
        ...

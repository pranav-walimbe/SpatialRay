"""
The dynamic policy scaling each pool on its own honest signal toward that signal's setpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from spatial_ray.policy.signals import MaxOf, Signal, TotalBacklog, Utilization
from spatial_ray.policy.types import Action, Observation

DEFAULT_TARGET_ONGOING_REQUESTS = 8.0


@dataclass(frozen=True)
class DynamicPolicy:
    """A stateless policy mapping each pool's signal demand to a whole replica target."""

    signals: dict[str, Signal]  # per-pool load signal keyed by Serve deployment name

    def decide(self, observation: Observation) -> Action:
        """Size each configured pool to the ceiling of its signal's replica demand.

        Args:
            observation: The latest per-pool signals to size each pool from.

        Returns:
            An action whose targets ceil each pool's demand, left for the controller to floor.
        """
        targets: dict[str, int] = {}
        for name, signal in self.signals.items():
            pool = observation.pools.get(name)
            if pool is None:
                continue
            demand = signal.demand(pool)
            if demand is not None:
                targets[name] = ceil(demand)
        return Action(targets=targets)


def disaggregated_dynamic_policy(
    *,
    decode_target_ongoing_requests: float,
    inference_target_ongoing_requests: float,
    transform_util_target: float = 0.7,
    inference_util_target: float = 0.7,
) -> DynamicPolicy:
    """Wire the decode, transform, and inference pools to their decided dynamic signals.

    Args:
        decode_target_ongoing_requests: Requests one decode replica is provisioned to hold at once.
        inference_target_ongoing_requests: Requests one inference replica is provisioned to hold.
        transform_util_target: CPU utilization setpoint the transform pool is sized to.
        inference_util_target: GPU utilization setpoint the inference pool is sized to.

    Returns:
        A dynamic policy pairing each pool with its own bottleneck signal.
    """
    return DynamicPolicy(
        signals={
            "decode": TotalBacklog(target_ongoing_requests=decode_target_ongoing_requests),
            "transform": Utilization(target=transform_util_target),
            "inference": MaxOf(
                (
                    Utilization(target=inference_util_target),
                    TotalBacklog(target_ongoing_requests=inference_target_ongoing_requests),
                )
            ),
        }
    )

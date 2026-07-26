"""
The dynamic policy scaling each pool on its own honest signal toward that signal's setpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from spatial_ray.policy.signals import Backlog, MaxOf, Signal, Utilization
from spatial_ray.policy.types import Action, Observation

_SERVE_DEFAULT_MAX_ONGOING = 5  # Ray Serve's per-replica request cap when a pool leaves it unset


def inference_queue_setpoint(max_ongoing_requests: int | None) -> float:
    """Derive the per-replica queue setpoint as double the replica's batch concurrency.

    Args:
        max_ongoing_requests: The inference deployment's per-replica request cap, or None if unset.

    Returns:
        The queued-request setpoint one inference replica is sized to hold.
    """
    resolved = (
        max_ongoing_requests if max_ongoing_requests is not None else _SERVE_DEFAULT_MAX_ONGOING
    )
    return 2 * resolved


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
            if pool is not None:
                targets[name] = ceil(signal.demand(pool))
        return Action(targets=targets)


def disaggregated_dynamic_policy(
    *,
    decode_bytes_per_replica: float,
    inference_queue_per_replica: float,
    transform_util_target: float = 0.7,
    inference_util_target: float = 0.7,
) -> DynamicPolicy:
    """Wire the decode, transform, and inference pools to their decided dynamic signals.

    Args:
        decode_bytes_per_replica: Work-in-flight bytes one decode replica is provisioned to hold.
        inference_queue_per_replica: Queued requests one inference replica is provisioned to hold.
        transform_util_target: CPU utilization setpoint the transform pool is sized to.
        inference_util_target: GPU utilization setpoint the inference pool is sized to.

    Returns:
        A dynamic policy pairing each pool with its own bottleneck signal.
    """
    return DynamicPolicy(
        signals={
            "decode": Backlog(per_replica=decode_bytes_per_replica, metric="work_in_flight"),
            "transform": Utilization(target=transform_util_target),
            "inference": MaxOf(
                (
                    Utilization(target=inference_util_target),
                    Backlog(per_replica=inference_queue_per_replica, metric="queue_depth"),
                )
            ),
        }
    )

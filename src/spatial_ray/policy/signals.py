"""
The per-pool load signals a dynamic policy reads to size each pool to its own bottleneck.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from spatial_ray.policy.types import PoolObservation


class Signal(Protocol):
    """A mapping from a pool's observation to the replica count that signal alone demands."""

    def demand(self, pool: PoolObservation) -> float:
        """Compute the replicas this signal would request for the pool.

        Args:
            pool: The pool's latest per-replica signals.

        Returns:
            The fractional replica count that holds the signal at its setpoint.
        """
        ...


@dataclass(frozen=True)
class Utilization:
    """A saturation-ratio signal for pools whose native utilization honestly tracks load."""

    target: float  # setpoint saturation fraction the pool is provisioned to hold

    def demand(self, pool: PoolObservation) -> float:
        """Scale replicas by the ratio of observed utilization to the target.

        Args:
            pool: The pool whose EWMA-smoothed utilization drives the ratio.

        Returns:
            The replicas that pull utilization back down to the target.
        """
        return pool.replicas * pool.utilization / self.target


@dataclass(frozen=True)
class Backlog:
    """An absolute-backlog signal for pools whose native utilization understates load."""

    per_replica: float  # backlog units one replica is provisioned to drain
    metric: str  # PoolObservation field read by name, work_in_flight or queue_depth

    def demand(self, pool: PoolObservation) -> float:
        """Divide the observed backlog by the per-replica capacity.

        Args:
            pool: The pool whose backlog metric is read by name.

        Returns:
            The replicas that keep the backlog at the per-replica capacity.
        """
        return getattr(pool, self.metric) / self.per_replica


@dataclass(frozen=True)
class MaxOf:
    """A combinator sizing a pool to whichever of its component signals is most saturated."""

    signals: tuple[Signal, ...]  # component signals whose demands are reduced by max

    def demand(self, pool: PoolObservation) -> float:
        """Take the largest demand across the component signals.

        Args:
            pool: The pool passed to each component signal.

        Returns:
            The maximum replica demand so the hottest bottleneck is satisfied.
        """
        return max(signal.demand(pool) for signal in self.signals)

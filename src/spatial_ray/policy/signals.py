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
class TotalBacklog:
    """A running-plus-queued signal sizing a pool the way Ray's own autoscaler sizes one."""

    target_ongoing_requests: float  # requests one replica is provisioned to hold at once

    def demand(self, pool: PoolObservation) -> float:
        """Divide the pool's running and router-queued requests by the per-replica target.

        Args:
            pool: The pool whose replica-side and router-side request counts are read.

        Returns:
            The replicas that hold every replica at the target ongoing requests.
        """
        return (pool.queue_depth + pool.queued_depth) / self.target_ongoing_requests


@dataclass(frozen=True)
class AdaptiveBacklog:
    """A byte-backlog signal sizing a pool from its live mean request size and batch concurrency."""

    max_ongoing_requests: int  # per-replica request cap the byte setpoint scales with

    def demand(self, pool: PoolObservation) -> float:
        """Divide bytes in flight by the concurrency-scaled per-replica byte setpoint.

        Args:
            pool: The pool whose bytes in flight and smoothed mean request size are read.

        Returns:
            The replicas that hold each at max_ongoing_requests mean-sized requests, or zero
            before any request has set the mean.
        """
        if pool.mean_decoded_bytes <= 0.0:
            return 0.0
        return pool.work_in_flight / (self.max_ongoing_requests * pool.mean_decoded_bytes)


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

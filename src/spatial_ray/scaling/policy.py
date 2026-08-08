"""
The capacity policy mapping normalized per-pool work onto replica demand.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite


@dataclass(frozen=True)
class PoolCapacity:
    work_per_s: float  # profiled work throughput of one replica
    target_utilization: float = 0.7  # fraction of profiled throughput provisioned for steady load
    target_queue_s: float = (
        2.0  # time budget in which provisioned replicas should drain pending work
    )

    def __post_init__(self) -> None:
        # validate the profile before it reaches a live scaling decision
        if not isfinite(self.work_per_s) or self.work_per_s <= 0.0:
            raise ValueError(f"work_per_s must be positive, got {self.work_per_s}")
        if not isfinite(self.target_utilization) or not 0.0 < self.target_utilization <= 1.0:
            raise ValueError(f"target_utilization must be in (0, 1], got {self.target_utilization}")
        if not isfinite(self.target_queue_s) or self.target_queue_s <= 0.0:
            raise ValueError(f"target_queue_s must be positive, got {self.target_queue_s}")


@dataclass(frozen=True)
class PoolLoad:
    work_per_s: float  # recent submitted work rate
    pending_work: float = 0.0  # work waiting for a replica

    def __post_init__(self) -> None:
        # reject corrupted telemetry instead of turning it into a scale-down decision
        if not isfinite(self.work_per_s) or self.work_per_s < 0.0:
            raise ValueError(f"work_per_s must be nonnegative, got {self.work_per_s}")
        if not isfinite(self.pending_work) or self.pending_work < 0.0:
            raise ValueError(f"pending_work must be nonnegative, got {self.pending_work}")


@dataclass(frozen=True)
class CapacityPolicy:
    """Scale pools from recent work rate with a pending-work burst correction."""

    capacities: dict[str, PoolCapacity]  # pool name to its profiled replica capacity

    def demand(self, pool: str, load: PoolLoad) -> int | None:
        """Compute the whole-replica demand for one configured pool.

        Args:
            pool: Pool whose capacity profile to use.
            load: Pool's recent work rate and pending work.

        Returns:
            Replica demand or None when the pool has no capacity profile.
        """
        capacity = self.capacities.get(pool)
        if capacity is None:
            return None
        usable_work_per_s = capacity.target_utilization * capacity.work_per_s
        rate_demand = load.work_per_s / usable_work_per_s
        queue_demand = load.pending_work / (usable_work_per_s * capacity.target_queue_s)
        return ceil(max(rate_demand, queue_demand))

    def decide(self, loads: dict[str, PoolLoad]) -> dict[str, int]:
        """Compute replica demand for every configured pool with telemetry.

        Args:
            loads: Normalized load keyed by pool name.

        Returns:
            Whole-replica demand keyed by configured pool name.
        """
        decisions = {}
        for name, load in loads.items():
            demand = self.demand(name, load)
            if demand is not None:
                decisions[name] = demand
        return decisions

"""
The per-pool replica budget bounds and the anti-flap tuning the controller clamps against.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PoolBounds:
    min_replicas: int  # hard floor, at least 1 so a pool never drains to zero
    max_replicas: int  # hard ceiling encoding the pool's fixed node budget
    max_step: int = 1  # most replicas the controller adds or removes in one tick

    def __post_init__(self) -> None:
        # a pool must keep at least one replica and admit a non-empty budget range
        if self.min_replicas < 1:
            raise ValueError(f"min_replicas must be at least 1, got {self.min_replicas}")
        if self.max_replicas < self.min_replicas:
            raise ValueError(
                f"max_replicas {self.max_replicas} is below min_replicas {self.min_replicas}"
            )
        if self.max_step < 1:
            raise ValueError(f"max_step must be at least 1, got {self.max_step}")

    def clamp(self, replicas: int) -> int:
        """Clamp a replica count into this pool's hard budget range.

        Args:
            replicas: Proposed replica count for the pool.

        Returns:
            The count clamped to [min_replicas, max_replicas].
        """
        return max(self.min_replicas, min(self.max_replicas, replicas))

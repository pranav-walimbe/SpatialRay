"""
The read/write data contract every policy, observation source, and actuator shares.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PoolObservation:
    name: str  # Serve deployment name of the pool
    replicas: int  # live replica count reconciled by Serve
    queue_depth: float  # requests waiting or in flight across the pool's replicas
    work_in_flight: (
        float  # spatialray_work_in_flight gauge sum, bytes for decode and tiles elsewhere
    )
    utilization: float  # EWMA-smoothed saturation fraction, CPU for transform and GPU for inference


@dataclass(frozen=True)
class Observation:
    t_s: float  # monotonic timestamp of the scrape, seconds
    arrival_rate: float  # system ingress request rate, requests per second
    pools: dict[str, PoolObservation] = field(
        default_factory=dict
    )  # per-pool signals keyed by name


@dataclass(frozen=True)
class Action:
    targets: dict[str, int] = field(default_factory=dict)  # desired num_replicas keyed by pool name

"""
The read/write data contract every policy, observation source, and actuator shares.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PoolObservation:
    name: str  # Serve deployment name of the pool
    replicas: int  # live replica count reconciled by Serve
    queue_depth: float | None  # requests processing across the replicas or None when unreported
    work_in_flight: (
        float | None  # work gauge sum of bytes for decode and tiles elsewhere or None when absent
    )
    utilization: float | None  # EWMA saturation fraction of its node or None when unreported
    mean_decoded_bytes: float | None = (
        None  # EWMA decoded bytes per request sizing decode's backlog
    )
    queued_depth: float | None = None  # requests waiting at the routers for a replica of this pool
    stale_s: float = 0.0  # seconds since these gauges last came from a whole scrape


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

"""
The Ray observation source reducing scraped Prometheus gauges and Serve status into an observation.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, MutableMapping

from ray import serve

from spatial_ray.control.ray_metrics import MetricsView, read_metrics_view
from spatial_ray.policy.types import Observation, PoolObservation

DEFAULT_EWMA_ALPHA = 0.3
CPU = "cpu"
GPU = "gpu"

# Utilization source per disaggregated pool
DISAGGREGATED_UTIL_KINDS: dict[str, str | None] = {
    "decode": None,
    "transform": CPU,
    "inference": GPU,
}


class _Ewma:
    """A one-pole EWMA smoother holding the last smoothed value for one pool's utilization."""

    def __init__(self, alpha: float) -> None:
        self._alpha = alpha
        self._value: float | None = None

    def update(self, sample: float | None) -> float | None:
        """Fold a new sample into the running average and return the smoothed value.

        Args:
            sample: The latest raw utilization fraction, or None when the scrape missed it.

        Returns:
            The updated EWMA, held unchanged across a missed sample and None before the first one.
        """
        if sample is None:
            return self._value
        if self._value is None:
            self._value = sample
        else:
            self._value = self._alpha * sample + (1.0 - self._alpha) * self._value
        return self._value


def _pool_utilization(
    pool: str, kind: str | None, view: MetricsView, replicas: int
) -> float | None:
    # the pool's summed node utilization per replica as a fraction or None when no node reported
    if kind is None or replicas <= 0:
        return None
    source = view.node_gpu if kind == GPU else view.node_cpu
    readings = [source[ip] for ip, role in view.roles.items() if role == pool and ip in source]
    if not readings:
        return None
    return sum(readings) / replicas / 100.0


def _carry_forward(held, scraped, t_s):
    # serve each gauge from this scrape or from the last one that reported it, with its age
    values: dict[str, float | None] = {}
    ages = []
    for field, value in scraped.items():
        if value is not None:
            held[field] = (value, t_s)
            values[field] = value
        elif field in held:
            carried, seen = held[field]
            values[field] = carried
            ages.append(t_s - seen)
        else:
            values[field] = None
    return values, max(ages, default=0.0)


def build_observation(
    t_s: float,
    view: MetricsView,
    replicas: Mapping[str, int],
    util_kinds: Mapping[str, str | None],
    ewmas: Mapping[str, _Ewma],
    arrival_rate: float = 0.0,
    held: MutableMapping[str, dict[str, tuple[float, float]]] | None = None,
    peak_handles: MutableMapping[str, int] | None = None,
) -> Observation:
    """Reduce one metrics scrape and replica census into a per-pool observation.

    Args:
        t_s: Monotonic timestamp of the scrape in seconds.
        view: The parsed per-node and per-deployment gauges from Prometheus.
        replicas: Live replica count keyed by pool name.
        util_kinds: Utilization source per pool, cpu or gpu or None to skip.
        ewmas: Per-pool EWMA smoother for the pools that carry a utilization signal.
        arrival_rate: System ingress request rate, left at zero until the predictive policy uses it.
        held: Per-pool gauge readings and the time each was last seen, updated in place.
        peak_handles: Per-pool high-water mark of routers seen reporting, updated in place.

    Returns:
        An observation carrying live replicas, backlog, work in flight, utilization, and staleness.
    """
    store = {} if held is None else held
    peaks = {} if peak_handles is None else peak_handles
    pools: dict[str, PoolObservation] = {}
    for name, count in replicas.items():
        raw = _pool_utilization(name, util_kinds.get(name), view, count)
        per_replica = view.queue.get(name)
        per_router = view.queued.get(name)
        if per_router is not None:
            peaks[name] = max(peaks.get(name, 0), view.queued_handles.get(name, 1))
        scraped = {
            "queue_depth": None if per_replica is None else per_replica * count,
            "queued_depth": None if per_router is None else per_router * peaks.get(name, 1),
            "work_in_flight": view.work.get(name),
            "mean_decoded_bytes": view.mean_bytes.get(name),
        }
        gauges, stale_s = _carry_forward(store.setdefault(name, {}), scraped, t_s)
        smoothed = ewmas[name].update(raw) if name in ewmas else raw
        pools[name] = PoolObservation(
            name=name, replicas=count, utilization=smoothed, stale_s=stale_s, **gauges
        )
    return Observation(t_s=t_s, arrival_rate=arrival_rate, pools=pools)


def serve_replica_counts(app_name: str = "spatialray") -> dict[str, int]:
    """Read each deployment's live running replica count from Serve status.

    Args:
        app_name: Name of the running Serve application to census.

    Returns:
        Running replica count keyed by deployment name, empty if the app is not present.
    """
    status = serve.status()
    application = status.applications.get(app_name)
    if application is None:
        return {}
    return {
        name: deployment.replica_states.get("RUNNING", 0)
        for name, deployment in application.deployments.items()
    }


class RayObservationSource:
    """The read side, folding a Prometheus scrape and Serve census into an observation each tick."""

    def __init__(
        self,
        read_metrics: Callable[[], MetricsView],
        read_replicas: Callable[[], dict[str, int]],
        util_kinds: Mapping[str, str | None],
        *,
        ewma_alpha: float = DEFAULT_EWMA_ALPHA,
    ) -> None:
        self._read_metrics = read_metrics
        self._read_replicas = read_replicas
        self._util_kinds = dict(util_kinds)
        # one smoother per pool that actually reports utilization
        self._ewmas = {
            name: _Ewma(ewma_alpha) for name, kind in util_kinds.items() if kind is not None
        }
        self._held: dict[str, dict[str, tuple[float, float]]] = {}
        self._peak_handles: dict[str, int] = {}

    def observe(self) -> Observation:
        """Scrape the current metrics and replica census into one observation.

        Returns:
            An observation snapshot for the policy to decide on.
        """
        return build_observation(
            time.monotonic(),
            self._read_metrics(),
            self._read_replicas(),
            self._util_kinds,
            self._ewmas,
            held=self._held,
            peak_handles=self._peak_handles,
        )


def ray_observation_source(
    util_kinds: Mapping[str, str | None] = DISAGGREGATED_UTIL_KINDS,
    *,
    app_name: str = "spatialray",
    ewma_alpha: float = DEFAULT_EWMA_ALPHA,
) -> RayObservationSource:
    """Wire an observation source to the live cluster's Prometheus scrape and Serve census.

    Args:
        util_kinds: Utilization source per pool, defaulting to the disaggregated grouping.
        app_name: Name of the running Serve application to census replicas from.
        ewma_alpha: Smoothing factor for each pool's utilization EWMA.

    Returns:
        A RayObservationSource reading live node and Serve gauges each tick.
    """
    return RayObservationSource(
        read_metrics=read_metrics_view,
        read_replicas=lambda: serve_replica_counts(app_name),
        util_kinds=util_kinds,
        ewma_alpha=ewma_alpha,
    )

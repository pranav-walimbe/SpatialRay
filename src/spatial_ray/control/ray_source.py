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

# gauge fields a pool carries forward from its last whole scrape
_GAUGE_FIELDS = ("queue_depth", "queued_depth", "work_in_flight", "mean_decoded_bytes")

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


def build_observation(
    t_s: float,
    view: MetricsView,
    replicas: Mapping[str, int],
    util_kinds: Mapping[str, str | None],
    ewmas: Mapping[str, _Ewma],
    arrival_rate: float = 0.0,
    last_good: MutableMapping[str, dict[str, float]] | None = None,
    last_whole_t: float | None = None,
    peak_handles: MutableMapping[str, int] | None = None,
) -> Observation:
    """Reduce one metrics scrape and replica census into a per-pool observation.

    A partial scrape leaves a pool's gauges absent rather than zero, so the last good reading is
    carried forward and stamped with its age instead of being read as an idle pool.

    Args:
        t_s: Monotonic timestamp of the scrape in seconds.
        view: The parsed per-node and per-deployment gauges from Prometheus.
        replicas: Live replica count keyed by pool name.
        util_kinds: Utilization source per pool, cpu or gpu or None to skip.
        ewmas: Per-pool EWMA smoother for the pools that carry a utilization signal.
        arrival_rate: System ingress request rate, left at zero until the predictive policy uses it.
        last_good: Per-pool gauge readings from the last whole scrape, updated in place.
        last_whole_t: Timestamp of the last whole scrape, or None if none has landed yet.
        peak_handles: Per-pool high-water mark of routers seen reporting, updated in place.

    Returns:
        An observation carrying live replicas, backlog, work in flight, utilization, and staleness.
    """
    held = {} if last_good is None else last_good
    peaks = {} if peak_handles is None else peak_handles
    if view.complete:
        stale_s = 0.0
    else:
        # before any whole scrape lands there is no last good reading to fall back on
        stale_s = float("inf") if last_whole_t is None else t_s - last_whole_t
    pools: dict[str, PoolObservation] = {}
    for name, count in replicas.items():
        raw = _pool_utilization(name, util_kinds.get(name), view, count)
        if view.complete:
            per_replica = view.queue.get(name)
            peaks[name] = max(peaks.get(name, 0), view.queued_handles.get(name, 1))
            per_router = view.queued.get(name)
            gauges = {
                "queue_depth": None if per_replica is None else per_replica * count,
                "queued_depth": None if per_router is None else per_router * peaks[name],
                "work_in_flight": view.work.get(name),
                "mean_decoded_bytes": view.mean_bytes.get(name),
            }
            held[name] = gauges
        else:
            gauges = held.get(name, dict.fromkeys(_GAUGE_FIELDS, None))
            raw = None
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
        # gauges from the last whole scrape carried forward across a partial one
        self._last_good: dict[str, dict[str, float]] = {}
        self._last_whole_t: float | None = None
        self._peak_handles: dict[str, int] = {}

    def observe(self) -> Observation:
        """Scrape the current metrics and replica census into one observation.

        Returns:
            An observation snapshot for the policy to decide on.
        """
        t_s = time.monotonic()
        view = self._read_metrics()
        observation = build_observation(
            t_s,
            view,
            self._read_replicas(),
            self._util_kinds,
            self._ewmas,
            last_good=self._last_good,
            last_whole_t=self._last_whole_t,
            peak_handles=self._peak_handles,
        )
        if view.complete:
            self._last_whole_t = t_s
        return observation


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

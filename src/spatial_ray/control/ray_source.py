"""
The Ray observation source reducing scraped Prometheus gauges and Serve status into an observation.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping

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

    def update(self, sample: float) -> float:
        """Fold a new sample into the running average and return the smoothed value.

        Args:
            sample: The latest raw utilization fraction.

        Returns:
            The updated EWMA, seeded to the first sample so it does not ramp from zero.
        """
        if self._value is None:
            self._value = sample
        else:
            self._value = self._alpha * sample + (1.0 - self._alpha) * self._value
        return self._value


def _pool_utilization(pool: str, kind: str | None, view: MetricsView, replicas: int) -> float:
    # the pool's summed node utilization divided per replica and normalized to a 0-to-1 fraction
    if kind is None or replicas <= 0:
        return 0.0
    source = view.node_gpu if kind == GPU else view.node_cpu
    ips = [ip for ip, role in view.roles.items() if role == pool]
    return sum(source.get(ip, 0.0) for ip in ips) / replicas / 100.0


def build_observation(
    t_s: float,
    view: MetricsView,
    replicas: Mapping[str, int],
    util_kinds: Mapping[str, str | None],
    ewmas: Mapping[str, _Ewma],
    arrival_rate: float = 0.0,
) -> Observation:
    """Reduce one metrics scrape and replica census into a per-pool observation.

    Args:
        t_s: Monotonic timestamp of the scrape in seconds.
        view: The parsed per-node and per-deployment gauges from Prometheus.
        replicas: Live replica count keyed by pool name.
        util_kinds: Utilization source per pool, cpu or gpu or None to skip.
        ewmas: Per-pool EWMA smoother for the pools that carry a utilization signal.
        arrival_rate: System ingress request rate, left at zero until the predictive policy uses it.

    Returns:
        An observation carrying live replicas, queue depth, work in flight, and utilization.
    """
    pools: dict[str, PoolObservation] = {}
    for name, count in replicas.items():
        raw = _pool_utilization(name, util_kinds.get(name), view, count)
        smoothed = ewmas[name].update(raw) if name in ewmas else raw
        pools[name] = PoolObservation(
            name=name,
            replicas=count,
            queue_depth=view.queue.get(name, 0.0),
            work_in_flight=view.work.get(name, 0.0),
            utilization=smoothed,
            mean_decoded_bytes=view.mean_bytes.get(name, 0.0),
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

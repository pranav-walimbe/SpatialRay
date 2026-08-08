"""
Reduces the cloud harness scrapes into time-series memory snapshots and per-deployment latency.
"""

from __future__ import annotations

import urllib.request
from collections import defaultdict
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import ray
from prometheus_client.parser import text_string_to_metric_families

NODE_CPU = "ray_node_cpu_utilization"
NODE_GPU = "ray_node_gpus_utilization"
NODE_GRAM = "ray_node_gram_used"
NODE_MEM = "ray_node_mem_used"
WORK = "ray_spatialray_work_in_flight"
QUEUE = "ray_serve_replica_processing_queries"
QUEUED = "ray_serve_deployment_queued_queries"

_LATENCY = "ray_serve_deployment_processing_latency_ms"
_NODE_RESOURCE_SUFFIX = "_node"
_SCRAPE_TIMEOUT_S = 2

# per-family reduction the plotted snapshot reads, each family with its label and aggregation
_SNAPSHOT_SPECS = {
    NODE_CPU: ("ip", "last"),
    NODE_GPU: ("ip", "sum"),
    NODE_GRAM: ("ip", "sum"),
    NODE_MEM: ("ip", "last"),
    WORK: ("deployment", "sum"),
    QUEUE: ("deployment", "sum"),
    QUEUED: ("deployment", "sum"),
}


@dataclass(frozen=True)
class Scrape:
    texts: tuple[str, ...]  # exposition document from each endpoint that answered
    failed: tuple[str, ...]  # endpoints that did not answer


def metrics_endpoints() -> list[str]:
    """Return the Prometheus metrics endpoint for every alive Ray node.

    Returns:
        One metrics URL per alive node in the current Ray cluster.
    """
    return [
        f"http://{node['NodeManagerAddress']}:{node['MetricsExportPort']}/metrics"
        for node in ray.nodes()
        if node.get("alive")
    ]


def node_roles() -> dict[str, str]:
    """Map Ray node addresses to their configured SpatialRay pool roles.

    Returns:
        Node address to pool name for nodes carrying a pool resource.
    """
    roles = {}
    for node in ray.nodes():
        address = node["NodeManagerAddress"]
        for resource in node.get("Resources", {}):
            if resource.endswith(_NODE_RESOURCE_SUFFIX):
                roles[address] = resource[: -len(_NODE_RESOURCE_SUFFIX)]
    return roles


def scrape(endpoints: list[str]) -> Scrape:
    """Fetch Prometheus exposition documents from Ray nodes concurrently.

    Args:
        endpoints: Ray node metrics URLs to fetch.

    Returns:
        The documents that answered and the endpoints that failed.
    """
    if not endpoints:
        return Scrape(texts=(), failed=())
    with ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
        fetched = tuple(pool.map(_fetch, endpoints))
    return Scrape(
        texts=tuple(text for text in fetched if text is not None),
        failed=tuple(url for url, text in zip(endpoints, fetched, strict=True) if text is None),
    )


def _fetch(url: str) -> str | None:
    # fetch one exposition document while treating network failures as missing samples
    try:
        with urllib.request.urlopen(url, timeout=_SCRAPE_TIMEOUT_S) as response:
            return response.read().decode()
    except OSError:
        return None


def reduce_families(
    texts: list[str], specs: Mapping[str, tuple[str, str]]
) -> dict[str, dict[str, float]]:
    """Reduce requested Prometheus families for benchmark reporting.

    Args:
        texts: Prometheus exposition documents from the sampled nodes.
        specs: Family name to grouping label and reduction of last, sum, or mean.

    Returns:
        Family name to grouped reduced values.
    """
    reduced: dict[str, dict[str, float]] = {name: {} for name in specs}
    reporters: dict[str, dict[str, int]] = {name: {} for name in specs}
    for text in texts:
        for family in text_string_to_metric_families(text):
            spec = specs.get(family.name)
            if spec is None:
                continue
            label, aggregation = spec
            bucket = reduced[family.name]
            counted = reporters[family.name]
            for sample in family.samples:
                key = sample.labels.get(label)
                if key is None:
                    continue
                if aggregation in ("sum", "mean") and key in bucket:
                    bucket[key] += sample.value
                else:
                    bucket[key] = sample.value
                counted[key] = counted.get(key, 0) + 1
    for name, (_, aggregation) in specs.items():
        if aggregation == "mean":
            reduced[name] = {
                key: total / reporters[name][key] for key, total in reduced[name].items()
            }
    return reduced


@dataclass(frozen=True)
class Snapshot:
    t_s: float  # seconds since the run started
    node_cpu: dict[str, float]  # node ip to CPU utilization percent
    node_gpu: dict[str, float]  # node ip to summed GPU utilization percent
    node_gram: dict[str, float]  # node ip to GPU memory used in bytes
    node_mem: dict[str, float]  # node ip to system memory used in bytes
    work: dict[str, float]  # deployment to work units in flight from our custom gauge
    queue: dict[str, float]  # deployment to queries being processed across its replicas
    queued: dict[str, float]  # deployment to queries waiting at the routers for a replica


def parse_snapshot(scraped: Scrape, t_s: float) -> Snapshot:
    """Reduce one scrape into a timestamped snapshot of the hardware and Serve gauges we plot.

    Args:
        scraped: The exposition documents that answered plus the endpoints that did not.
        t_s: Seconds since the run started, stamped onto the snapshot.

    Returns:
        A Snapshot holding per-node hardware gauges and per-deployment work and backlog.
    """
    reduced = reduce_families(list(scraped.texts), _SNAPSHOT_SPECS)
    return Snapshot(
        t_s=t_s,
        node_cpu=reduced[NODE_CPU],
        node_gpu=reduced[NODE_GPU],
        node_gram=reduced[NODE_GRAM],
        node_mem=reduced[NODE_MEM],
        work=reduced[WORK],
        queue=reduced[QUEUE],
        queued=reduced[QUEUED],
    )


@dataclass(frozen=True)
class _LatencyCounters:
    count: float  # requests the histogram has observed
    total: float  # summed processing latency in ms across those requests
    buckets: dict[float, float]  # upper bound in ms to the cumulative count at or below it


class LatencyAccumulator:
    """Folds each scrape's latency counters forward so replicas killed mid-run stay counted."""

    def __init__(self) -> None:
        self._banked: dict[str, _LatencyCounters] = {}
        self._previous: dict[str, _LatencyCounters] = {}

    def update(self, texts: list[str]) -> None:
        """Fold one scrape's cumulative latency histogram into the running totals.

        Args:
            texts: Per-node Prometheus exposition documents scraped from the cluster.
        """
        for deployment, scraped in _latency_counters(texts).items():
            self._banked[deployment] = _advance(
                self._banked.get(deployment), self._previous.get(deployment), scraped
            )
            self._previous[deployment] = scraped

    def stats(self) -> dict[str, dict]:
        """Reduce the accumulated counters to per-deployment latency stats.

        Returns:
            Deployment name to its request count, mean, p50, and p99 latency in ms.
        """
        return {deployment: _stats(counters) for deployment, counters in self._banked.items()}


def _advance(banked, previous, scraped):
    # bank only each counter's positive delta so a dying replica's drop keeps what it already did
    if banked is None or previous is None:
        return scraped
    keys = set(banked.buckets) | set(scraped.buckets)
    return _LatencyCounters(
        count=banked.count + max(0.0, scraped.count - previous.count),
        total=banked.total + max(0.0, scraped.total - previous.total),
        buckets={
            le: banked.buckets.get(le, 0.0)
            + max(0.0, scraped.buckets.get(le, 0.0) - previous.buckets.get(le, 0.0))
            for le in keys
        },
    )


def _latency_counters(texts):
    # one scrape's cumulative latency histogram per deployment, summed over the replicas reporting
    counts: dict[str, float] = defaultdict(float)
    sums: dict[str, float] = defaultdict(float)
    buckets: dict[str, dict[float, float]] = defaultdict(lambda: defaultdict(float))
    for text in texts:
        for family in text_string_to_metric_families(text):
            if family.name != _LATENCY:
                continue
            for sample in family.samples:
                deployment = sample.labels.get("deployment")
                if deployment is None:
                    continue
                if sample.name.endswith("_count"):
                    counts[deployment] += sample.value
                elif sample.name.endswith("_sum"):
                    sums[deployment] += sample.value
                elif sample.name.endswith("_bucket"):
                    buckets[deployment][float(sample.labels["le"])] += sample.value
    return {
        deployment: _LatencyCounters(
            count=count, total=sums[deployment], buckets=dict(buckets[deployment])
        )
        for deployment, count in counts.items()
    }


def _stats(counters):
    # one deployment's accumulated counters reduced to the report's latency columns
    ordered = sorted(counters.buckets.items())
    return {
        "n_requests": int(counters.count),
        "latency_mean_ms": counters.total / counters.count if counters.count else 0.0,
        "latency_p50_ms": _histogram_quantile(ordered, 0.50),
        "latency_p99_ms": _histogram_quantile(ordered, 0.99),
    }


def _histogram_quantile(buckets: list[tuple[float, float]], quantile: float) -> float:
    # Linear-interpolated Prometheus histogram quantile over cumulative (le, count) buckets
    if not buckets:
        return 0.0
    total = buckets[-1][1]
    if total <= 0:
        return 0.0
    rank = quantile * total
    prev_le, prev_count = 0.0, 0.0
    for le, count in buckets:
        if count >= rank:
            if le == float("inf"):
                return prev_le
            span = count - prev_count
            return le if span <= 0 else prev_le + (le - prev_le) * (rank - prev_count) / span
        prev_le, prev_count = le, count
    return buckets[-1][0]

"""
Scrapes Ray's Prometheus endpoints into time-series snapshots and per-deployment latency stats.
"""

from __future__ import annotations

import urllib.request
from collections import defaultdict
from dataclasses import dataclass

import ray
from prometheus_client.parser import text_string_to_metric_families

_SCRAPE_TIMEOUT_S = 5
_NODE_CPU = "ray_node_cpu_utilization"
_NODE_GPU = "ray_node_gpus_utilization"
_NODE_GRAM = "ray_node_gram_used"
_NODE_MEM = "ray_node_mem_used"
_WORK = "ray_spatialray_work_in_flight"
_QUEUE = "ray_serve_replica_processing_queries"
_LATENCY = "ray_serve_deployment_processing_latency_ms"


@dataclass(frozen=True)
class Snapshot:
    t_s: float  # seconds since the run started
    node_cpu: dict[str, float]  # node ip to CPU utilization percent
    node_gpu: dict[str, float]  # node ip to summed GPU utilization percent
    node_gram: dict[str, float]  # node ip to GPU memory used in bytes
    node_mem: dict[str, float]  # node ip to system memory used in bytes
    work: dict[str, float]  # deployment to work units in flight from our custom gauge
    queue: dict[str, float]  # deployment to queries being processed across its replicas


def metrics_endpoints() -> list[str]:
    """Return the Prometheus /metrics URL of every alive Ray node.

    Returns:
        One scrape URL per alive node in the current Ray cluster.
    """
    return [
        f"http://{node['NodeManagerAddress']}:{node['MetricsExportPort']}/metrics"
        for node in ray.nodes()
        if node.get("alive")
    ]


def node_roles() -> dict[str, str]:
    """Map each node ip to the stage it hosts, read from its per-stage node resource.

    Returns:
        Node ip to role name for nodes carrying a <role>_node custom resource.
    """
    roles = {}
    for node in ray.nodes():
        ip = node["NodeManagerAddress"]
        for resource in node.get("Resources", {}):
            if resource.endswith("_node"):
                roles[ip] = resource[: -len("_node")]
    return roles


def scrape(endpoints: list[str]) -> str:
    """Fetch and concatenate the Prometheus exposition text from each endpoint.

    Args:
        endpoints: Node /metrics URLs to scrape, unreachable ones are skipped.

    Returns:
        The concatenated exposition text across all reachable endpoints.
    """
    chunks = []
    for url in endpoints:
        try:
            with urllib.request.urlopen(url, timeout=_SCRAPE_TIMEOUT_S) as response:
                chunks.append(response.read().decode())
        except OSError:
            continue
    return "\n".join(chunks)


def parse_snapshot(text: str, t_s: float) -> Snapshot:
    """Reduce one scrape into a timestamped snapshot of the gauges we plot.

    Args:
        text: Prometheus exposition text scraped from the cluster.
        t_s: Seconds since the run started, stamped onto the snapshot.

    Returns:
        A Snapshot holding per-node hardware gauges and per-deployment work and queue depth.
    """
    node_cpu, node_gpu, node_gram, node_mem = {}, {}, {}, {}
    work: dict[str, float] = defaultdict(float)
    queue: dict[str, float] = defaultdict(float)
    for family in text_string_to_metric_families(text):
        if family.name == _NODE_CPU:
            _fill_by_ip(node_cpu, family.samples)
        elif family.name == _NODE_GPU:
            _accumulate_by_ip(node_gpu, family.samples)
        elif family.name == _NODE_GRAM:
            _accumulate_by_ip(node_gram, family.samples)
        elif family.name == _NODE_MEM:
            _fill_by_ip(node_mem, family.samples)
        elif family.name == _WORK:
            _accumulate_by_deployment(work, family.samples)
        elif family.name == _QUEUE:
            _accumulate_by_deployment(queue, family.samples)
    return Snapshot(t_s, node_cpu, node_gpu, node_gram, node_mem, dict(work), dict(queue))


def deployment_latency(text: str) -> dict[str, dict]:
    """Reduce the cumulative processing-latency histogram to per-deployment latency stats.

    Args:
        text: Prometheus exposition text scraped from the cluster.

    Returns:
        Deployment name to its request count, mean, p50, and p99 latency in ms.
    """
    counts: dict[str, float] = defaultdict(float)
    sums: dict[str, float] = defaultdict(float)
    buckets: dict[str, dict[float, float]] = defaultdict(lambda: defaultdict(float))
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
    stats = {}
    for deployment, count in counts.items():
        ordered = sorted(buckets[deployment].items())
        stats[deployment] = {
            "n_requests": int(count),
            "latency_mean_ms": sums[deployment] / count if count else 0.0,
            "latency_p50_ms": _histogram_quantile(ordered, 0.50),
            "latency_p99_ms": _histogram_quantile(ordered, 0.99),
        }
    return stats


def work_units(text: str) -> dict[str, str]:
    """Read each deployment's work-unit label off the work-in-flight gauge.

    Args:
        text: Prometheus exposition text scraped from the cluster.

    Returns:
        Deployment name to its work-unit label, bytes or tiles.
    """
    units = {}
    for family in text_string_to_metric_families(text):
        if family.name != _WORK:
            continue
        for sample in family.samples:
            deployment = sample.labels.get("deployment")
            if deployment is not None and "work_unit" in sample.labels:
                units[deployment] = sample.labels["work_unit"]
    return units


def _fill_by_ip(target, samples):
    # Set each node's gauge value, keyed by the ip label
    for sample in samples:
        target[sample.labels.get("ip", "?")] = sample.value


def _accumulate_by_ip(target, samples):
    # Sum a per-node gauge over its sub-series, for example one series per GPU on the node
    for sample in samples:
        ip = sample.labels.get("ip", "?")
        target[ip] = target.get(ip, 0.0) + sample.value


def _accumulate_by_deployment(target, samples):
    # Sum a Serve gauge across a deployment's replicas, keyed by the deployment label
    for sample in samples:
        deployment = sample.labels.get("deployment")
        if deployment is not None:
            target[deployment] += sample.value


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

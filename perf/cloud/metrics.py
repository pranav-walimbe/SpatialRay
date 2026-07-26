"""
Reduces the cloud harness scrapes into time-series memory snapshots and per-deployment latency.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from prometheus_client.parser import text_string_to_metric_families

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


def parse_snapshot(texts: list[str], t_s: float) -> Snapshot:
    """Reduce one scrape into a timestamped snapshot of the hardware and Serve gauges we plot.

    Args:
        texts: Per-node Prometheus exposition documents scraped from the cluster.
        t_s: Seconds since the run started, stamped onto the snapshot.

    Returns:
        A Snapshot holding per-node hardware gauges and per-deployment work and queue depth.
    """
    node_cpu: dict[str, float] = {}
    node_gpu: dict[str, float] = {}
    node_gram: dict[str, float] = {}
    node_mem: dict[str, float] = {}
    work: dict[str, float] = defaultdict(float)
    queue: dict[str, float] = defaultdict(float)
    for text in texts:
        for family in text_string_to_metric_families(text):
            for sample in family.samples:
                ip = sample.labels.get("ip", "?")
                deployment = sample.labels.get("deployment")
                if family.name == _NODE_CPU:
                    node_cpu[ip] = sample.value
                elif family.name == _NODE_GPU:
                    node_gpu[ip] = node_gpu.get(ip, 0.0) + sample.value
                elif family.name == _NODE_GRAM:
                    node_gram[ip] = node_gram.get(ip, 0.0) + sample.value
                elif family.name == _NODE_MEM:
                    node_mem[ip] = sample.value
                elif family.name == _WORK and deployment is not None:
                    work[deployment] += sample.value
                elif family.name == _QUEUE and deployment is not None:
                    queue[deployment] += sample.value
    return Snapshot(t_s, node_cpu, node_gpu, node_gram, node_mem, dict(work), dict(queue))


def deployment_latency(texts: list[str]) -> dict[str, dict]:
    """Reduce the cumulative processing-latency histogram to per-deployment latency stats.

    Args:
        texts: Per-node Prometheus exposition documents scraped from the cluster.

    Returns:
        Deployment name to its request count, mean, p50, and p99 latency in ms.
    """
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

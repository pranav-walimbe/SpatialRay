"""
The general Ray Prometheus reader scraping node and Serve gauges into an observation MetricsView.
"""

from __future__ import annotations

import urllib.request
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

import ray
from prometheus_client.parser import text_string_to_metric_families

_SCRAPE_TIMEOUT_S = 5
_NODE_CPU = "ray_node_cpu_utilization"
_NODE_GPU = "ray_node_gpus_utilization"
_WORK = "ray_spatialray_work_in_flight"
_QUEUE = "ray_serve_replica_processing_queries"


@dataclass(frozen=True)
class MetricsView:
    node_cpu: dict[str, float]  # node ip to CPU utilization percent
    node_gpu: dict[str, float]  # node ip to summed GPU utilization percent
    work: dict[str, float]  # deployment to work units in flight from our custom gauge
    queue: dict[str, float]  # deployment to queued and processing queries across its replicas
    roles: dict[str, str]  # node ip to the pool that node hosts


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
    """Map each node ip to the pool it hosts, read from its per-stage node resource.

    Returns:
        Node ip to pool name for nodes carrying a <pool>_node custom resource.
    """
    roles = {}
    for node in ray.nodes():
        ip = node["NodeManagerAddress"]
        for resource in node.get("Resources", {}):
            if resource.endswith("_node"):
                roles[ip] = resource[: -len("_node")]
    return roles


def scrape(endpoints: list[str]) -> list[str]:
    """Fetch the Prometheus exposition text from each reachable endpoint.

    Args:
        endpoints: Node /metrics URLs to scrape, unreachable ones are skipped.

    Returns:
        One exposition document per reachable endpoint, kept separate so each parses on its own.
    """
    texts = []
    for url in endpoints:
        try:
            with urllib.request.urlopen(url, timeout=_SCRAPE_TIMEOUT_S) as response:
                texts.append(response.read().decode())
        except OSError:
            continue
    return texts


def parse_metrics_view(texts: list[str], roles: Mapping[str, str]) -> MetricsView:
    """Reduce one scrape into the per-node and per-deployment gauges an observation reads.

    Node gauges take the last value seen per ip while GPU sub-series and Serve replicas sum,
    so a multi-GPU node and a multi-replica deployment each roll up to one figure.

    Args:
        texts: Per-node Prometheus exposition documents scraped from the cluster.
        roles: Node ip to the pool it hosts, carried through onto the view.

    Returns:
        A MetricsView holding per-node CPU and GPU utilization and per-deployment work and queue.
    """
    node_cpu: dict[str, float] = {}
    node_gpu: dict[str, float] = {}
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
                elif family.name == _WORK and deployment is not None:
                    work[deployment] += sample.value
                elif family.name == _QUEUE and deployment is not None:
                    queue[deployment] += sample.value
    return MetricsView(node_cpu, node_gpu, dict(work), dict(queue), dict(roles))


def read_metrics_view() -> MetricsView:
    """Scrape the live cluster into one MetricsView of node and Serve gauges.

    Returns:
        A MetricsView parsed from every alive node, tagged with each node's pool role.
    """
    return parse_metrics_view(scrape(metrics_endpoints()), node_roles())

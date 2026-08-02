"""
The general Ray Prometheus reader scraping node and Serve gauges into an observation MetricsView.
"""

from __future__ import annotations

import urllib.request
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import ray
from prometheus_client.parser import text_string_to_metric_families

_SCRAPE_TIMEOUT_S = 2
_NODE_RESOURCE_SUFFIX = "_node"

# Prometheus family names shared by the control loop and the perf harness
NODE_CPU = "ray_node_cpu_utilization"
NODE_GPU = "ray_node_gpus_utilization"
NODE_GRAM = "ray_node_gram_used"
NODE_MEM = "ray_node_mem_used"
WORK = "ray_spatialray_work_in_flight"
QUEUE = "ray_serve_replica_processing_queries"
QUEUED = "ray_serve_deployment_queued_queries"
MEAN_BYTES = "ray_spatialray_mean_decoded_bytes"

# per-family reduction the observation view reads, each family with its label and aggregation
_VIEW_SPECS = {
    NODE_CPU: ("ip", "last"),
    NODE_GPU: ("ip", "sum"),
    WORK: ("deployment", "sum"),
    QUEUE: ("deployment", "sum"),
    QUEUED: ("deployment", "sum"),
    MEAN_BYTES: ("deployment", "last"),
}


@dataclass(frozen=True)
class Scrape:
    texts: tuple[str, ...]  # exposition document from each endpoint that answered
    failed: tuple[str, ...]  # endpoints that did not answer so their gauges are absent not zero

    @property
    def complete(self) -> bool:
        """Whether every endpoint answered so a missing gauge really means zero.

        Returns:
            True when no endpoint was skipped.
        """
        return not self.failed


@dataclass(frozen=True)
class MetricsView:
    node_cpu: dict[str, float]  # node ip to CPU utilization percent
    node_gpu: dict[str, float]  # node ip to summed GPU utilization percent
    work: dict[str, float]  # deployment to work units in flight from our custom gauge
    queue: dict[str, float]  # deployment to queries being processed across its replicas
    roles: dict[str, str]  # node ip to the pool that node hosts
    mean_bytes: dict[str, float] = field(
        default_factory=dict
    )  # deployment to its EWMA decoded bytes per request
    queued: dict[str, float] = field(
        default_factory=dict
    )  # deployment to queries waiting at the routers for a replica
    complete: bool = True  # whether every endpoint answered so a missing gauge means zero load


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


def node_resource(pool: str) -> str:
    """Return the custom Ray resource pinning a pool's replicas to its per-stage node.

    Args:
        pool: Pool name, e.g. decode, transform, or inference.

    Returns:
        The <pool>_node custom resource key the pool's replicas and its node share.
    """
    return f"{pool}{_NODE_RESOURCE_SUFFIX}"


def node_roles() -> dict[str, str]:
    """Map each node ip to the pool it hosts, read from its per-stage node resource.

    Returns:
        Node ip to pool name for nodes carrying a <pool>_node custom resource.
    """
    roles = {}
    for node in ray.nodes():
        ip = node["NodeManagerAddress"]
        for resource in node.get("Resources", {}):
            if resource.endswith(_NODE_RESOURCE_SUFFIX):
                roles[ip] = resource[: -len(_NODE_RESOURCE_SUFFIX)]
    return roles


def scrape(endpoints: list[str]) -> Scrape:
    """Fetch the Prometheus exposition text from every endpoint at once.

    Args:
        endpoints: Node /metrics URLs to scrape.

    Returns:
        A Scrape holding the documents that answered and the endpoints that did not.
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
    # one endpoint's exposition text or None when it times out or refuses
    try:
        with urllib.request.urlopen(url, timeout=_SCRAPE_TIMEOUT_S) as response:
            return response.read().decode()
    except OSError:
        return None


def reduce_families(
    texts: list[str], specs: Mapping[str, tuple[str, str]]
) -> dict[str, dict[str, float]]:
    """Reduce scrapes into each requested family's label-keyed gauge values.

    A key no endpoint reported is left out of its family rather than defaulted to zero, so a
    caller can tell an absent gauge from an idle one.

    Args:
        texts: Per-node Prometheus exposition documents scraped from the cluster.
        specs: Family name to a (label, aggregation) pair.

    Returns:
        Family name to its label value to the reduced gauge.
    """
    reduced: dict[str, dict[str, float]] = {name: {} for name in specs}
    for text in texts:
        for family in text_string_to_metric_families(text):
            spec = specs.get(family.name)
            if spec is None:
                continue
            label, aggregation = spec
            bucket = reduced[family.name]
            for sample in family.samples:
                key = sample.labels.get(label)
                if key is None:
                    continue
                if aggregation == "sum" and key in bucket:
                    bucket[key] += sample.value
                else:
                    bucket[key] = sample.value
    return reduced


def parse_metrics_view(
    scraped: Scrape, roles: Mapping[str, str], *, complete: bool | None = None
) -> MetricsView:
    """Reduce one scrape into the per-node and per-deployment gauges an observation reads.

    Args:
        scraped: The documents that answered plus the endpoints that did not.
        roles: Node ip to the pool it hosts, carried through onto the view.
        complete: Override for whether the scrape was whole, defaulting to the scrape's own verdict.

    Returns:
        A MetricsView of the node and deployment gauges, flagged with whether the scrape was whole.
    """
    reduced = reduce_families(list(scraped.texts), _VIEW_SPECS)
    return MetricsView(
        node_cpu=reduced[NODE_CPU],
        node_gpu=reduced[NODE_GPU],
        work=reduced[WORK],
        queue=reduced[QUEUE],
        roles=dict(roles),
        mean_bytes=reduced[MEAN_BYTES],
        queued=reduced[QUEUED],
        complete=scraped.complete if complete is None else complete,
    )


def read_metrics_view() -> MetricsView:
    """Scrape the live cluster into one MetricsView of node and Serve gauges.

    Returns:
        A MetricsView parsed from every alive node, tagged with each node's pool role.
    """
    return parse_metrics_view(scrape(metrics_endpoints()), node_roles())

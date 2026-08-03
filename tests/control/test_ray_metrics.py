"""
Tests the Prometheus reader reduces node and Serve gauges into a MetricsView.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Gauge, generate_latest

from spatial_ray.control.ray_metrics import Scrape, parse_metrics_view


def _node_a() -> str:
    # scrape for a node with two transform CPUs and the inference GPU
    registry = CollectorRegistry()
    cpu = Gauge("ray_node_cpu_utilization", "cpu", ["ip"], registry=registry)
    cpu.labels(ip="t1").set(40.0)
    cpu.labels(ip="t2").set(60.0)
    gpu = Gauge("ray_node_gpus_utilization", "gpu", ["ip", "GpuIndex"], registry=registry)
    gpu.labels(ip="g1", GpuIndex="0").set(30.0)
    gpu.labels(ip="g1", GpuIndex="1").set(50.0)
    queue = Gauge("ray_serve_replica_processing_queries", "q", ["deployment"], registry=registry)
    queue.labels(deployment="inference").set(4.0)
    return generate_latest(registry).decode()


def _node_b() -> str:
    # second node's scrape with decode work and an inference queue
    registry = CollectorRegistry()
    work = Gauge(
        "ray_spatialray_work_in_flight", "w", ["deployment", "work_unit"], registry=registry
    )
    work.labels(deployment="decode", work_unit="bytes").set(1024.0)
    mean = Gauge("ray_spatialray_mean_decoded_bytes", "m", ["deployment"], registry=registry)
    mean.labels(deployment="decode").set(512.0)
    queue = Gauge("ray_serve_replica_processing_queries", "q", ["deployment"], registry=registry)
    queue.labels(deployment="inference").set(3.0)
    queued = Gauge(
        "ray_serve_deployment_queued_queries", "qd", ["deployment", "handle"], registry=registry
    )
    queued.labels(deployment="decode", handle="h1").set(12.0)
    queued.labels(deployment="decode", handle="h2").set(8.0)
    return generate_latest(registry).decode()


def _scrape(*texts: str) -> Scrape:
    # a whole scrape of the given exposition documents
    return Scrape(texts=texts, failed=())


def test_parse_metrics_view():
    """Per-node CPU is kept, GPUs on a node sum, and the replica and router queues mean."""
    roles = {"t1": "transform", "t2": "transform", "g1": "inference"}
    view = parse_metrics_view(_scrape(_node_a(), _node_b()), roles)
    assert view.node_cpu == {"t1": 40.0, "t2": 60.0}
    assert view.node_gpu == {"g1": 80.0}
    assert view.work == {"decode": 1024.0}
    assert view.mean_bytes == {"decode": 512.0}
    assert view.queue == {"inference": 3.5}
    assert view.queued == {"decode": 10.0}
    assert view.queued_handles == {"decode": 2}
    assert view.roles == roles


def test_queue_gauges_mean_over_the_sources_that_reported():
    """Both queue gauges average their reporters so a partial export is not an undercount."""
    both = parse_metrics_view(_scrape(_node_a(), _node_b()), {})
    one = parse_metrics_view(_scrape(_node_a()), {})
    assert (both.queue, both.queued) == ({"inference": 3.5}, {"decode": 10.0})
    assert (one.queue, one.queued) == ({"inference": 4.0}, {})


def test_parse_metrics_view_partial():
    """An unanswered endpoint leaves its gauges absent rather than reducing them to zero."""
    view = parse_metrics_view(Scrape(texts=(_node_a(),), failed=("http://b:8080/metrics",)), {})
    assert view.queued == {}
    assert view.work == {}
    assert view.mean_bytes == {}

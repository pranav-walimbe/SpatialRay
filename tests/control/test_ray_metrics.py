"""
Tests the Prometheus reader reduces node and Serve gauges into a MetricsView.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Gauge, generate_latest

from spatial_ray.control.ray_metrics import parse_metrics_view


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
    queue = Gauge("ray_serve_replica_processing_queries", "q", ["deployment"], registry=registry)
    queue.labels(deployment="inference").set(3.0)
    return generate_latest(registry).decode()


def test_parse_metrics_view():
    """Per-node CPU is kept, GPUs on a node sum, a deployment sums across nodes."""
    roles = {"t1": "transform", "t2": "transform", "g1": "inference"}
    view = parse_metrics_view([_node_a(), _node_b()], roles)
    assert view.node_cpu == {"t1": 40.0, "t2": 60.0}
    assert view.node_gpu == {"g1": 80.0}
    assert view.work == {"decode": 1024.0}
    assert view.queue == {"inference": 7.0}
    assert view.roles == roles

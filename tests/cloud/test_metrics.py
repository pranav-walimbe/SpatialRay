"""
Tests the latency accumulator banks a killed replica's histogram instead of losing it.
"""

from __future__ import annotations

from perf.cloud.metrics import LatencyAccumulator

_BUCKETS = (100.0, 1000.0, float("inf"))


def _exposition(deployment: str, count: float, total: float, buckets: dict[float, float]) -> str:
    # one replica's processing-latency histogram as Prometheus exposition text
    name = "ray_serve_deployment_processing_latency_ms"
    lines = [f"# TYPE {name} histogram"]
    for le, value in buckets.items():
        label = "+Inf" if le == float("inf") else repr(le)
        lines.append(f'{name}_bucket{{deployment="{deployment}",le="{label}"}} {value}')
    lines.append(f'{name}_count{{deployment="{deployment}"}} {count}')
    lines.append(f'{name}_sum{{deployment="{deployment}"}} {total}')
    return "\n".join(lines) + "\n"


def _histogram(count: float, total: float) -> str:
    # a histogram whose whole count sits in the slowest finite bucket
    return _exposition("decode", count, total, dict.fromkeys(_BUCKETS, count))


def test_accumulates_monotonic_counters():
    """Successive scrapes of a growing counter are banked once, not double counted."""
    accumulator = LatencyAccumulator()
    accumulator.update([_histogram(10, 5000.0)])
    accumulator.update([_histogram(25, 12500.0)])

    stats = accumulator.stats()["decode"]
    assert stats["n_requests"] == 25
    assert stats["latency_mean_ms"] == 500.0


def test_banks_counts_from_a_killed_replica():
    """A drop in the summed counter keeps the dead replica's requests and rebaselines the rest."""
    accumulator = LatencyAccumulator()
    # two replicas reporting 40 requests between them, then one is killed taking its 30 with it
    accumulator.update([_histogram(40, 20000.0)])
    accumulator.update([_histogram(10, 5000.0)])
    accumulator.update([_histogram(18, 9000.0)])

    stats = accumulator.stats()["decode"]
    assert stats["n_requests"] == 48
    assert stats["latency_mean_ms"] == 500.0


def test_missing_deployment_leaves_totals_untouched():
    """A scrape that reports nothing for a deployment neither drops nor rebanks its totals."""
    accumulator = LatencyAccumulator()
    accumulator.update([_histogram(10, 5000.0)])
    accumulator.update([])

    assert accumulator.stats()["decode"]["n_requests"] == 10

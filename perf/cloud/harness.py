"""
Deploys the disaggregated Ray Serve graph and paces a Poisson trace while sampling its metrics.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import ray
from ray import serve

from perf.cloud.app import from_config
from perf.cloud.metrics import (
    LatencyAccumulator,
    Snapshot,
    metrics_endpoints,
    node_roles,
    parse_snapshot,
    scrape,
)
from perf.common.models import load
from perf.common.trace import build_default_trace

_SAMPLE_INTERVAL_S = 1.0  # metrics scrape period during the run


@dataclass(frozen=True)
class Report:
    model_name: str  # model module the inference pool served
    hardware: str  # cpu or gpu inference target
    n_requests: int  # requests the trace drove through the graph
    rate_per_s: float  # mean Poisson arrival rate the trace was driven at
    wall_s: float  # total wall-clock of the run
    samples: tuple[Snapshot, ...]  # metrics snapshots sampled across the run
    latency: dict[str, dict]  # deployment to its cumulative latency stats
    roles: dict[str, str]  # node ip to the stage it hosts


def run(*, model_name: str, hardware: str, n_requests: int, rate_per_s: float) -> Report:
    """Deploy the disaggregated graph and drive a Poisson trace while sampling metrics.

    Args:
        model_name: Model module name under perf.common.models for the inference pool.
        hardware: Target hardware, cpu or gpu, selecting the inference replica's device.
        n_requests: Number of requests the Poisson trace generates.
        rate_per_s: Mean Poisson arrival rate in requests per second.

    Returns:
        A Report with the run's metrics time-series and per-deployment latency stats.
    """
    model = load(model_name)
    trace = build_default_trace(model, n=n_requests, rate_per_s=rate_per_s)
    application = from_config(model_name, hardware)
    ray.init(ignore_reinit_error=True)
    try:
        handle = serve.run(application.graph, name=application.app_name)
        endpoints = metrics_endpoints()
        roles = node_roles()
        samples: list[Snapshot] = []
        latency = LatencyAccumulator()
        wall_s = asyncio.run(_run_load(handle, trace, endpoints, samples, latency))
        latency.update(list(scrape(endpoints).texts))
    finally:
        serve.shutdown()
        ray.shutdown()

    return Report(
        model_name=model_name,
        hardware=hardware,
        n_requests=len(trace),
        rate_per_s=rate_per_s,
        wall_s=wall_s,
        samples=tuple(samples),
        latency=latency.stats(),
        roles=roles,
    )


async def _run_load(handle, trace, endpoints, samples, latency):
    # pace the trace with an async sampler ticking in the same loop
    start = time.perf_counter()
    sampler = asyncio.create_task(_sampler(endpoints, samples, start, latency))
    try:
        await _drive(handle, trace, start)
    finally:
        sampler.cancel()
        try:
            await sampler
        except asyncio.CancelledError:
            pass
    return time.perf_counter() - start


async def _sampler(endpoints, samples, start, latency):
    # scrape off the event loop on a fixed deadline
    deadline = time.perf_counter()
    while True:
        try:
            samples.append(await asyncio.to_thread(_take_snapshot, endpoints, start, latency))
        except Exception as error:
            print(f"metrics sample skipped: {error}")
        deadline = max(deadline + _SAMPLE_INTERVAL_S, time.perf_counter())
        await asyncio.sleep(deadline - time.perf_counter())


def _take_snapshot(endpoints, start, latency):
    # one blocking scrape folded into the latency totals and parsed into a timestamped snapshot
    scraped = scrape(endpoints)
    latency.update(list(scraped.texts))
    return parse_snapshot(scraped, time.perf_counter() - start)


async def _drive(handle, trace, start):
    # fire and await each request at its arrival time so one failure cannot cancel the others
    async def _fire(entry):
        delay = (start + entry.arrival_s) - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        await handle.remote(entry.request)

    results = await asyncio.gather(
        *(asyncio.create_task(_fire(entry)) for entry in trace), return_exceptions=True
    )
    failures = [r for r in results if isinstance(r, Exception)]
    if failures:
        print(f"{len(failures)}/{len(results)} requests failed, first: {failures[0]!r}")

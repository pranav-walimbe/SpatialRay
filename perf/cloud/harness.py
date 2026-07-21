"""
Deploys the disaggregated Ray Serve graph and paces a Poisson trace while sampling its metrics.
"""

from __future__ import annotations

import asyncio
import dataclasses
import threading
import time
from dataclasses import dataclass

import ray
from ray import serve

from perf.cloud.metrics import (
    Snapshot,
    deployment_latency,
    metrics_endpoints,
    node_roles,
    parse_snapshot,
    scrape,
    work_units,
)
from perf.cloud.utils import load_config
from perf.common.models import load
from perf.common.trace import build_default_trace
from spatial_ray.serve.graph import DISAGGREGATED, InferenceSpec, build_graph

_SAMPLE_INTERVAL_S = 1.0  # metrics scrape period during the run


@dataclass(frozen=True)
class Report:
    model_name: str  # model module the inference pool served
    hardware: str  # cpu or gpu inference target
    n_requests: int  # requests the trace drove through the graph
    wall_s: float  # total wall-clock of the run
    samples: tuple[Snapshot, ...]  # metrics snapshots sampled across the run
    latency: dict[str, dict]  # deployment to its cumulative latency stats
    roles: dict[str, str]  # node ip to the stage it hosts
    work_units: dict[str, str]  # deployment to its work-in-flight unit label


def run(*, model_name: str, hardware: str, n_requests: int, rate_per_s: float) -> Report:
    """Deploy the disaggregated graph, drive a Poisson trace, and sample its Ray metrics.

    Args:
        model_name: Model module name under perf.common.models for the inference pool.
        hardware: Target hardware, cpu or gpu, selecting the inference replica's device.
        n_requests: Number of requests the Poisson trace generates.
        rate_per_s: Mean Poisson arrival rate in requests per second.

    Returns:
        A Report with the run's metrics time-series and per-deployment latency stats.
    """
    pools_cfg = load_config()["pools"]
    model = load(model_name)
    trace = build_default_trace(model, n=n_requests, rate_per_s=rate_per_s)
    app = build_graph(
        _sized_pools(pools_cfg),
        inference=_inference_spec(pools_cfg, model_name, hardware),
    )
    ray.init(ignore_reinit_error=True)
    try:
        handle = serve.run(app)
        endpoints = metrics_endpoints()
        roles = node_roles()
        stop = threading.Event()
        samples: list[Snapshot] = []
        start = time.perf_counter()
        sampler = threading.Thread(
            target=_sample_loop, args=(stop, endpoints, samples, start), daemon=True
        )
        sampler.start()
        wall_s = asyncio.run(_drive(handle, trace))
        stop.set()
        sampler.join()
        final = scrape(endpoints)
        latency = deployment_latency(final)
        units = work_units(final)
    finally:
        serve.shutdown()
        ray.shutdown()

    return Report(
        model_name=model_name,
        hardware=hardware,
        n_requests=len(trace),
        wall_s=wall_s,
        samples=tuple(samples),
        latency=latency,
        roles=roles,
        work_units=units,
    )


def _sized_pools(pools_cfg):
    # Size each pool from config and pin it to its per-stage node resource
    pools = []
    for spec in DISAGGREGATED:
        pool_cfg = pools_cfg[spec.name]
        options = {"num_cpus": pool_cfg["num_cpus"], "resources": {f"{spec.name}_node": 0.01}}
        pools.append(
            dataclasses.replace(spec, num_replicas=pool_cfg["replicas"], ray_actor_options=options)
        )
    return tuple(pools)


def _inference_spec(pools_cfg, model_name, hardware):
    # Build the inference pool spec from config with its replica pinned to the inference node
    inference_cfg = pools_cfg["inference"]
    variant = inference_cfg[hardware]
    options = {"resources": {"inference_node": 0.01}}
    for key in ("num_cpus", "num_gpus"):
        if key in variant:
            options[key] = variant[key]
    return InferenceSpec(
        model_factory=_model_factory(model_name),
        num_replicas=inference_cfg["replicas"],
        ray_actor_options=options,
    )


def _model_factory(model_name):
    # Zero-arg factory that Serve cloudpickles to rebuild the model on each inference replica
    def build():
        return load(model_name).build()

    return build


def _sample_loop(stop, endpoints, samples, start):
    # Scrape and parse a metrics snapshot every interval until told to stop
    while not stop.is_set():
        samples.append(parse_snapshot(scrape(endpoints), time.perf_counter() - start))
        stop.wait(_SAMPLE_INTERVAL_S)


async def _drive(handle, trace):
    # Fire each request at its trace arrival time and await every response before returning
    start = time.perf_counter()

    async def _fire(entry):
        delay = (start + entry.arrival_s) - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        await handle.remote(entry.request)

    await asyncio.gather(*(asyncio.create_task(_fire(entry)) for entry in trace))
    return time.perf_counter() - start

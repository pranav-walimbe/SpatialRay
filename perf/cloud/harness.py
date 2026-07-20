"""
Persistent, multiprocess load-test harness replaying a Poisson trace, one OS process per pool.
"""

from __future__ import annotations

import multiprocessing as mp
import statistics
import threading
import time
from dataclasses import dataclass

from perf.common.measure import peak_rss_bytes
from perf.common.models import load
from perf.common.report import RunStats, StageLoadStats
from perf.common.trace import build_default_trace
from spatial_ray.serve.deployments import StagePool
from spatial_ray.serve.graph import DISAGGREGATED
from spatial_ray.workload.metadata import RasterPayload

_STOP = None  # poison pill signaling a stage process no more requests are coming
_UTIL_SAMPLE_S = 0.5  # utilization sampler poll interval


@dataclass(frozen=True)
class _RequestRecord:
    latency_s: float  # this stage's wall time for the request
    n_tiles: int | None  # tiles produced, when the stage's payload carries them


def run(*, model_name: str, hardware: str, n_requests: int, rate_per_s: float) -> RunStats:
    """Replay an n-request Poisson trace through a persistent multiprocess pipeline.

    Args:
        model_name: Model module name under perf.common.models for the inference stage.
        hardware: Target hardware, cpu or gpu, selecting the inference device.
        n_requests: Number of requests the Poisson trace generates.
        rate_per_s: Mean Poisson arrival rate in requests per second.

    Returns:
        Whole-run utilization plus per-stage throughput/latency/memory stats.
    """
    model = load(model_name)
    trace = build_default_trace(model, n=n_requests, rate_per_s=rate_per_s)
    pool_stages = {spec.name: spec.stages for spec in DISAGGREGATED}

    ctx = mp.get_context("spawn")
    decode_q, transform_q, inference_q, results_q = (ctx.Queue() for _ in range(4))
    processes = [
        ctx.Process(
            target=_stage_worker,
            args=("decode", pool_stages["decode"], decode_q, transform_q, results_q),
        ),
        ctx.Process(
            target=_stage_worker,
            args=("transform", pool_stages["transform"], transform_q, inference_q, results_q),
        ),
        ctx.Process(target=_inference_worker, args=(model_name, hardware, inference_q, results_q)),
    ]
    for process in processes:
        process.start()

    stop_sampling = threading.Event()
    cpu_samples: list[float] = []
    gpu_samples: list[float] = []
    sampler = threading.Thread(
        target=_sample_utilization,
        args=(stop_sampling, hardware == "gpu", cpu_samples, gpu_samples),
        daemon=True,
    )
    sampler.start()

    start = time.perf_counter()
    for entry in trace:
        delay = (start + entry.arrival_s) - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        decode_q.put(RasterPayload(request=entry.request))
    decode_q.put(_STOP)

    stage_stats = dict(results_q.get() for _ in range(3))
    wall_s = time.perf_counter() - start

    stop_sampling.set()
    sampler.join()
    for process in processes:
        process.join()

    return RunStats(
        n_requests=n_requests,
        wall_s=wall_s,
        cpu_util_mean=statistics.fmean(cpu_samples) if cpu_samples else 0.0,
        cpu_util_peak=max(cpu_samples) if cpu_samples else 0.0,
        gpu_util_mean=statistics.fmean(gpu_samples) if gpu_samples else None,
        gpu_util_peak=max(gpu_samples) if gpu_samples else None,
        stages=(stage_stats["decode"], stage_stats["transform"], stage_stats["inference"]),
    )


def _stage_worker(name, stages, in_q, out_q, results_q) -> None:
    # Persistent stage process: StagePool owns any shared resources (e.g. decode's IO pool)
    stage_pool = StagePool(stages)
    records = []
    start = time.perf_counter()
    while (payload := in_q.get()) is not _STOP:
        t0 = time.perf_counter()
        payload = stage_pool.run(payload)
        n_tiles = int(payload.tiles.shape[0]) if payload.tiles is not None else None
        records.append(_RequestRecord(latency_s=time.perf_counter() - t0, n_tiles=n_tiles))
        out_q.put(payload)
    wall_s = time.perf_counter() - start
    stage_pool.shutdown()
    out_q.put(_STOP)
    results_q.put((name, _summarize(name, "cpu", records, wall_s)))


def _inference_worker(model_name, hardware, in_q, results_q) -> None:
    # Persistent inference process: model loaded once, forward pass reused across every request
    import torch

    model = load(model_name).build()
    device = "cuda" if hardware == "gpu" and torch.cuda.is_available() else "cpu"
    if hardware == "gpu" and device == "cpu":
        raise RuntimeError("hardware=gpu requested but torch.cuda.is_available() is False")

    records = []
    start = time.perf_counter()
    while (payload := in_q.get()) is not _STOP:
        t0 = time.perf_counter()
        model(payload.tiles)
        records.append(
            _RequestRecord(latency_s=time.perf_counter() - t0, n_tiles=int(payload.tiles.shape[0]))
        )
    wall_s = time.perf_counter() - start
    results_q.put(("inference", _summarize("inference", device, records, wall_s)))


def _summarize(name, device, records, wall_s) -> StageLoadStats:
    # Reduce a stage process's per-request records into throughput/latency/memory stats
    n = len(records)
    latencies_ms = [r.latency_s * 1e3 for r in records]
    tile_counts = [r.n_tiles for r in records if r.n_tiles is not None]
    return StageLoadStats(
        name=name,
        device=device,
        n_requests=n,
        throughput_req_s=n / wall_s if wall_s else 0.0,
        throughput_tiles_s=sum(tile_counts) / wall_s if tile_counts and wall_s else None,
        latency_mean_ms=statistics.fmean(latencies_ms) if latencies_ms else 0.0,
        latency_median_ms=statistics.median(latencies_ms) if latencies_ms else 0.0,
        latency_p99_ms=_p99(latencies_ms),
        peak_rss_b=peak_rss_bytes(),
    )


def _p99(values: list[float]) -> float:
    # 99th percentile, falling back to the lone/no sample for runs too small for quantiles
    if len(values) < 2:
        return values[0] if values else 0.0
    return statistics.quantiles(values, n=100, method="inclusive")[98]


def _sample_utilization(stop, has_gpu, cpu_samples, gpu_samples) -> None:
    # Poll system CPU% and, on gpu runs, GPU% every _UTIL_SAMPLE_S until told to stop
    import psutil

    torch = None
    if has_gpu:
        import torch as _torch

        torch = _torch
    psutil.cpu_percent(interval=None)  # prime the baseline, the first real reading follows it
    while not stop.is_set():
        stop.wait(_UTIL_SAMPLE_S)
        cpu_samples.append(psutil.cpu_percent(interval=None))
        if torch is not None:
            gpu_samples.append(float(torch.cuda.utilization()))

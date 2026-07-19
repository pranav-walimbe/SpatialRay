"""
Cloud subprocess-per-stage orchestrator that measures one trace disjointly, stage by stage.
"""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from perf.common.measure import RequestMeasurement, StageMeasurement
from perf.common.trace import TraceEntry
from spatial_ray.workload.metadata import RasterPayload

# Stage names in execution order, decode through the terminal inference stage
STAGE_SEQUENCE = ("decode", "reproject_stage", "normalize", "tile", "inference")
_INFERENCE_STAGE = "inference"
_RUNNER_MODULE = "perf.cloud._stage_runner"


def measure_trace(
    trace: Sequence[TraceEntry],
    *,
    model: str,
    hardware: str,
) -> list[RequestMeasurement]:
    """Measure each request in the trace with every stage run in its own subprocess.

    Args:
        trace: Requests to measure in arrival order.
        model: Model module name under perf.common.models for the inference stage.
        hardware: Target hardware, cpu or gpu, selecting the inference device.

    Returns:
        One measurement record per trace entry, in arrival order.
    """
    with tempfile.TemporaryDirectory(prefix="spatialray-perf-") as scratch:
        root = Path(scratch)
        return [
            _measure_request(entry.request, model=model, hardware=hardware, scratch=root, index=i)
            for i, entry in enumerate(trace)
        ]


def _measure_request(request, *, model, hardware, scratch, index):
    # Chain the stages through pickled payload files, one isolated subprocess per stage
    paths = [scratch / f"req{index}_stage{s}.pkl" for s in range(len(STAGE_SEQUENCE) + 1)]
    paths[0].write_bytes(pickle.dumps(RasterPayload(request=request)))
    measurements = []
    n_tiles = 0
    for i, stage in enumerate(STAGE_SEQUENCE):
        output = None if stage == _INFERENCE_STAGE else paths[i + 1]
        record = _run_stage(stage, paths[i], output, model, hardware)
        measurements.append(
            StageMeasurement(
                name=record["name"],
                wall_s=record["wall_s"],
                rss_peak_b=record["rss_peak_b"],
                vram_peak_b=record["vram_peak_b"],
            )
        )
        if record["n_tiles"] is not None:
            n_tiles = record["n_tiles"]
    return RequestMeasurement(
        scene_id=request.scene.item_id, n_tiles=n_tiles, stages=tuple(measurements)
    )


def _run_stage(stage, input_path, output_path, model, hardware):
    # Invoke the stage runner subprocess and parse its JSON cost line
    cmd = [
        sys.executable,
        "-m",
        _RUNNER_MODULE,
        "--stage",
        stage,
        "--input",
        str(input_path),
        "--model",
        model,
        "--hardware",
        hardware,
    ]
    if output_path is not None:
        cmd += ["--output", str(output_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"stage {stage} failed:\n{proc.stderr}")
    return json.loads(proc.stdout.strip().splitlines()[-1])

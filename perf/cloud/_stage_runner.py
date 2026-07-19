"""
Subprocess worker that runs one pipeline stage in isolation and reports its measured cost.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

from perf.common.measure import StageMeasurement, peak_rss_bytes
from spatial_ray.workload.stages import decode, normalize, reproject_stage, tile

_PAYLOAD_STAGES = {
    "decode": decode,
    "reproject_stage": reproject_stage,
    "normalize": normalize,
    "tile": tile,
}
INFERENCE_STAGE = "inference"


def main() -> None:
    """Run one named stage on a pickled payload and print its measured cost as a JSON line."""
    parser = argparse.ArgumentParser(description="Run one isolated pipeline stage.")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    parser.add_argument("--model", required=True)
    parser.add_argument("--hardware", required=True, choices=("cpu", "gpu"))
    args = parser.parse_args()

    payload = pickle.loads(Path(args.input).read_bytes())
    if args.stage == INFERENCE_STAGE:
        measurement, n_tiles = _run_inference(payload, args.model, args.hardware)
    else:
        measurement, n_tiles = _run_payload_stage(payload, args.stage, args.output)
    print(
        json.dumps(
            {
                "name": measurement.name,
                "wall_s": measurement.wall_s,
                "rss_delta_b": measurement.rss_delta_b,
                "vram_peak_b": measurement.vram_peak_b,
                "n_tiles": n_tiles,
            }
        )
    )


def _run_payload_stage(payload, name, output_path):
    # Run a RasterPayload stage, measure its time and RSS, then dump the resulting payload
    stage = _PAYLOAD_STAGES[name]
    rss_before = peak_rss_bytes()
    start = time.perf_counter()
    payload = stage(payload)
    wall_s = time.perf_counter() - start
    rss_delta = peak_rss_bytes() - rss_before
    Path(output_path).write_bytes(pickle.dumps(payload))
    return StageMeasurement(name=name, wall_s=wall_s, rss_delta_b=rss_delta, vram_peak_b=0), None


def _run_inference(payload, model_name, hardware):
    # Run the model forward on the tile batch, measuring time, RSS, and CUDA peak on gpu
    import torch  # lazy so the cpu preprocessing stages never import torch

    from perf.common.models import load

    model = load(model_name).build()
    tiles = payload.tiles
    on_gpu = hardware == "gpu" and torch.cuda.is_available()
    if on_gpu:
        torch.cuda.reset_peak_memory_stats()
    rss_before = peak_rss_bytes()
    start = time.perf_counter()
    model(tiles)
    wall_s = time.perf_counter() - start
    rss_delta = peak_rss_bytes() - rss_before
    vram = int(torch.cuda.max_memory_allocated()) if on_gpu else 0
    measurement = StageMeasurement(
        name=INFERENCE_STAGE, wall_s=wall_s, rss_delta_b=rss_delta, vram_peak_b=vram
    )
    return measurement, int(tiles.shape[0])


if __name__ == "__main__":
    main()

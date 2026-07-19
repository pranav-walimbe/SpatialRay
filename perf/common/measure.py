"""
Per-stage time and memory measurement types plus a cross-platform peak-RSS reader.
"""

from __future__ import annotations

import resource
import sys
from dataclasses import dataclass

# ru_maxrss is bytes on macOS but kibibytes on Linux, so normalize to bytes
_RU_MAXRSS_TO_BYTES = 1 if sys.platform == "darwin" else 1024


@dataclass(frozen=True)
class StageMeasurement:
    name: str  # stage name
    wall_s: float  # wall-clock seconds spent in the stage
    rss_peak_b: int  # peak resident set size of the stage subprocess, in bytes
    vram_peak_b: int  # peak CUDA memory during the stage in bytes, 0 when not on gpu


@dataclass(frozen=True)
class RequestMeasurement:
    scene_id: str  # scene the request read from
    n_tiles: int  # tiles produced for the request
    stages: tuple[StageMeasurement, ...]  # per-stage measurements in execution order


def peak_rss_bytes() -> int:
    """Return the current process peak resident set size in bytes.

    Returns:
        Peak RSS high-water mark normalized to bytes across platforms.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * _RU_MAXRSS_TO_BYTES

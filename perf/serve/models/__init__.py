"""
Registry that loads a perf inference model module by name.
"""

from __future__ import annotations

import importlib
from types import ModuleType

DEFAULT_MODEL = "sample_vit_small"


def load(name: str) -> ModuleType:
    """Import a perf model module by name.

    Args:
        name: Module name under perf.serve.models, for example prithvi_eo_v1_100m.

    Returns:
        The model module exposing BAND_NAMES, TILE_SIZE, and build.
    """
    return importlib.import_module(f"perf.serve.models.{name}")

"""
Tests the STAC proj:code parser extracts the integer EPSG code.
"""

from __future__ import annotations

from spatial_ray.workload.catalog import _parse_epsg


def test_parse_epsg_extracts_code():
    """_parse_epsg reads the integer code from a STAC proj:code string."""
    assert _parse_epsg("EPSG:32610") == 32610

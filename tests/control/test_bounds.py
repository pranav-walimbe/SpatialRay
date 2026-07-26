"""
Tests the per-pool bounds reject bad ranges and clamp replica counts into the budget.
"""

from __future__ import annotations

import pytest

from spatial_ray.control.bounds import PoolBounds


def test_clamp():
    """Counts below the floor lift to it and above the ceiling drop to it."""
    bounds = PoolBounds(min_replicas=1, max_replicas=3)
    assert bounds.clamp(0) == 1
    assert bounds.clamp(2) == 2
    assert bounds.clamp(9) == 3


def test_rejects_bad_ranges():
    """Refuses a below-one floor, an inverted range, and a zero step."""
    with pytest.raises(ValueError):
        PoolBounds(min_replicas=0, max_replicas=3)
    with pytest.raises(ValueError):
        PoolBounds(min_replicas=3, max_replicas=1)
    with pytest.raises(ValueError):
        PoolBounds(min_replicas=1, max_replicas=3, max_step=0)

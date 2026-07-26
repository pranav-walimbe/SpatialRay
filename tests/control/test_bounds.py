"""
Tests the per-pool bounds reject bad ranges and clamp replica counts into the budget.
"""

from __future__ import annotations

import pytest

from spatial_ray.control.bounds import PoolBounds


def test_clamp_holds_count_inside_the_budget():
    """Counts below the floor lift up to it and counts above the ceiling drop down to it."""
    bounds = PoolBounds(min_replicas=1, max_replicas=3)
    assert bounds.clamp(0) == 1
    assert bounds.clamp(2) == 2
    assert bounds.clamp(9) == 3


def test_rejects_zero_floor_and_inverted_range_and_zero_step():
    """The bounds refuse a below-one floor, a ceiling under the floor, and a zero step."""
    with pytest.raises(ValueError):
        PoolBounds(min_replicas=0, max_replicas=3)
    with pytest.raises(ValueError):
        PoolBounds(min_replicas=3, max_replicas=1)
    with pytest.raises(ValueError):
        PoolBounds(min_replicas=1, max_replicas=3, max_step=0)

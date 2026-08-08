"""
Tests the workload meter reports stage-specific rates over a sliding window.
"""

from __future__ import annotations

import pytest

from spatial_ray.scaling.meter import WorkloadMeter
from spatial_ray.scaling.metrics import work_rate_metric


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_reports_work_rates_per_pool():
    """Each pool receives its own weighted rate over the same request stream."""
    clock = _Clock()
    meter = WorkloadMeter(
        {"decode": lambda item: item["bytes"], "inference": lambda item: item["tiles"]},
        window_s=10.0,
        time_fn=clock,
    )
    assert meter.record({"bytes": 100.0, "tiles": 2.0}) == {
        "decode": 100.0,
        "inference": 2.0,
    }
    clock.now = 2.0
    meter.record({"bytes": 300.0, "tiles": 6.0})
    metrics = meter.snapshot()
    assert metrics[work_rate_metric("decode")] == 200.0
    assert metrics[work_rate_metric("inference")] == 4.0


def test_expires_old_work():
    """Events outside the window stop contributing to the reported rates."""
    clock = _Clock()
    meter = WorkloadMeter({"decode": lambda item: float(item)}, window_s=5.0, time_fn=clock)
    meter.record(100.0)
    clock.now = 6.0
    metrics = meter.snapshot()
    assert metrics[work_rate_metric("decode")] == 0.0


def test_rejects_negative_work():
    """A broken estimator cannot turn invalid work into a scaling signal."""
    meter = WorkloadMeter({"decode": lambda item: -1.0})
    try:
        meter.record(object())
    except ValueError as error:
        assert "nonnegative" in str(error)
    else:
        raise AssertionError("negative work was accepted")


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_rejects_nonfinite_work(value):
    """A nonfinite estimate cannot poison later capacity decisions."""
    meter = WorkloadMeter({"decode": lambda item: value})
    with pytest.raises(ValueError, match="nonnegative"):
        meter.record(object())

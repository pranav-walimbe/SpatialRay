"""
Tests the controller tick log separates an unreported gauge from a reported zero.
"""

from __future__ import annotations

from perf.cloud.controller import _LoggingPolicy
from spatial_ray.policy.dynamic import DynamicPolicy
from spatial_ray.policy.signals import TotalBacklog
from spatial_ray.policy.types import Observation, PoolObservation


def _observation(**gauges) -> Observation:
    # one decode pool whose gauge presence the caller sets
    fields = {"queue_depth": 0.0, "queued_depth": 0.0, "work_in_flight": 0.0, "utilization": None}
    fields.update(gauges)
    return Observation(
        t_s=0.0,
        arrival_rate=0.0,
        pools={"decode": PoolObservation(name="decode", replicas=3, **fields)},
    )


def _policy() -> _LoggingPolicy:
    return _LoggingPolicy(inner=DynamicPolicy(signals={"decode": TotalBacklog(8.0)}))


def test_absent_gauge_logs_a_dash_and_holds(capsys):
    """An unreported gauge logs as a dash with the pool held rather than given a target."""
    _policy().decide(_observation(queue_depth=None, queued_depth=None, work_in_flight=None))
    line = capsys.readouterr().out
    assert "run=- q=- work=-" in line
    assert "want=hold" in line


def test_reported_zero_logs_a_zero_and_scales(capsys):
    """A gauge reporting a genuine zero logs as zero and still drives a target."""
    _policy().decide(_observation())
    line = capsys.readouterr().out
    assert "run=0 q=0 work=0" in line
    assert "want=0" in line

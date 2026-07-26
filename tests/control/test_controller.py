"""
Tests the controller clamps policy targets through the deadband, step cap, cooldowns, and bounds.
"""

from __future__ import annotations

from spatial_ray.control.bounds import PoolBounds
from spatial_ray.control.controller import Controller
from spatial_ray.policy.types import Action, Observation, PoolObservation


def _obs(replicas, t_s=0.0):
    # a one-pool observation named "p" with the given live replica count
    pool = PoolObservation(
        name="p", replicas=replicas, queue_depth=0.0, work_in_flight=0.0, utilization=0.0
    )
    return Observation(t_s=t_s, arrival_rate=0.0, pools={"p": pool})


def _controller(max_step=1, **kwargs):
    # a controller over a single pool "p" with an eager, do-nothing source, policy, and actuator
    bounds = {"p": PoolBounds(min_replicas=1, max_replicas=5, max_step=max_step)}
    return Controller(_FakeSource(), _FakePolicy(), _FakeActuator(), bounds, **kwargs)


class _FakeSource:
    def observe(self):
        return _obs(2)


class _FakePolicy:
    def decide(self, observation):
        return Action(targets={"p": 5})


class _FakeActuator:
    def __init__(self):
        self.applied = None

    def apply(self, action):
        self.applied = action


def test_deadband_ignores_small_moves_near_the_setpoint():
    """A one-replica move off twenty is under the ten percent deadband so nothing changes."""
    controller = _controller()
    assert controller.reconcile(_obs(20), Action(targets={"p": 21})).targets["p"] == 20


def test_step_cap_ramps_one_replica_per_tick():
    """A jump from one to five is capped to a single-replica step."""
    controller = _controller(max_step=1)
    assert controller.reconcile(_obs(1), Action(targets={"p": 5})).targets["p"] == 2


def test_clamp_holds_targets_inside_the_budget():
    """A target above the ceiling is clamped down to the pool max."""
    controller = _controller(max_step=10)
    assert controller.reconcile(_obs(4), Action(targets={"p": 9})).targets["p"] == 5


def test_scale_down_waits_for_the_cooldown_but_scale_up_does_not():
    """An up move records the change and a down move is blocked until its cooldown elapses."""
    controller = _controller(max_step=4, scale_up_cooldown_s=0.0, scale_down_cooldown_s=60.0)
    assert controller.reconcile(_obs(2, t_s=0.0), Action(targets={"p": 5})).targets["p"] == 5
    assert controller.reconcile(_obs(5, t_s=10.0), Action(targets={"p": 1})).targets["p"] == 5
    assert controller.reconcile(_obs(5, t_s=70.0), Action(targets={"p": 1})).targets["p"] == 1


def test_tick_applies_the_reconciled_action():
    """One tick observes two replicas, decides five, and applies the step-capped target."""
    controller = _controller(max_step=1)
    applied = controller.tick()
    assert applied.targets["p"] == 3
    assert controller._actuator.applied is applied

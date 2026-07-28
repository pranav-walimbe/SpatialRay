"""
Tests the controller clamps policy targets through the deadband, step cap, cooldowns, and bounds.
"""

from __future__ import annotations

from spatial_ray.control.bounds import PoolBounds
from spatial_ray.control.controller import Controller
from spatial_ray.policy.types import Action, Observation, PoolObservation


def _obs(replicas, t_s=0.0):
    # one-pool observation with the given replica count
    pool = PoolObservation(
        name="p", replicas=replicas, queue_depth=0.0, work_in_flight=0.0, utilization=0.0
    )
    return Observation(t_s=t_s, arrival_rate=0.0, pools={"p": pool})


def _controller(max_step=1, **kwargs):
    # controller over one pool with fake source, policy, and actuator
    bounds = {"p": PoolBounds(min_replicas=1, max_replicas=5, max_step=max_step)}
    return Controller(_FakeSource(), _FakePolicy(), _FakeActuator(), bounds, **kwargs)


class _FakeSource:
    def observe(self):
        return _obs(2)


class _ColdSource:
    def observe(self):
        return _obs(0)


class _FakePolicy:
    def decide(self, observation):
        return Action(targets={"p": 5})


class _FakeActuator:
    def __init__(self):
        self.applied = None

    def apply(self, action):
        self.applied = action


def test_deadband():
    """A one-replica move inside the deadband changes nothing."""
    controller = _controller()
    assert controller.reconcile(_obs(20), Action(targets={"p": 21})).targets["p"] == 20


def test_step_cap():
    """A jump from one to five caps to a single step."""
    controller = _controller(max_step=1)
    assert controller.reconcile(_obs(1), Action(targets={"p": 5})).targets["p"] == 2


def test_clamp():
    """A target above the ceiling clamps to the pool max."""
    controller = _controller(max_step=10)
    assert controller.reconcile(_obs(4), Action(targets={"p": 9})).targets["p"] == 5


def test_scale_down_cooldown():
    """Scale-up applies at once and scale-down waits for its cooldown."""
    controller = _controller(max_step=4, scale_up_cooldown_s=0.0, scale_down_cooldown_s=60.0)
    assert controller.reconcile(_obs(2, t_s=0.0), Action(targets={"p": 5})).targets["p"] == 5
    assert controller.reconcile(_obs(5, t_s=10.0), Action(targets={"p": 1})).targets["p"] == 5
    assert controller.reconcile(_obs(5, t_s=70.0), Action(targets={"p": 1})).targets["p"] == 1


def test_tick_applies():
    """One tick reconciles the decision and applies the step-capped target."""
    controller = _controller(max_step=1)
    applied = controller.tick()
    assert applied.targets["p"] == 3
    assert controller._actuator.applied is applied


def test_cold_start_tick_skips_apply():
    """A pool below its replica floor holds off scaling and applies nothing."""
    bounds = {"p": PoolBounds(min_replicas=1, max_replicas=5, max_step=1)}
    controller = Controller(_ColdSource(), _FakePolicy(), _FakeActuator(), bounds)
    proposed = controller.tick()
    assert proposed.targets["p"] == 5
    assert controller._actuator.applied is None

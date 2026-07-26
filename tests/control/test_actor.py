"""
Tests the controller host builds its loop offline without pickling live resources.
"""

from __future__ import annotations

from spatial_ray.control.actor import ControllerHost
from spatial_ray.control.bounds import PoolBounds
from spatial_ray.policy.dynamic import disaggregated_dynamic_policy


def test_controller_host_builds_offline_and_stop_before_start_is_a_noop():
    """Constructing the host wires the loop without a cluster and stopping before start is safe."""
    host = ControllerHost(
        disaggregated_dynamic_policy(decode_bytes_per_replica=1.0, inference_queue_per_replica=1.0),
        {"decode": PoolBounds(min_replicas=1, max_replicas=8)},
        {"applications": []},
    )
    host.stop()

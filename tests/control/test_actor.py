"""
Tests the controller host builds its loop offline without pickling live resources.
"""

from __future__ import annotations

from spatial_ray.control.actor import ControllerHost
from spatial_ray.control.bounds import PoolBounds
from spatial_ray.policy.dynamic import disaggregated_dynamic_policy


def test_host_builds_offline():
    """The host wires its loop without a cluster and stops safely before start."""
    host = ControllerHost(
        disaggregated_dynamic_policy(
            decode_max_ongoing_requests=1, inference_queue_per_replica=1.0
        ),
        {"decode": PoolBounds(min_replicas=1, max_replicas=8)},
        {"applications": []},
    )
    host.stop()

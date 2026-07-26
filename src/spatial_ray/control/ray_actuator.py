"""
The Ray actuator applying a policy's replica targets by re-submitting the patched Serve config.
"""

from __future__ import annotations

import copy
from typing import Any

from ray.serve.context import _get_global_client
from ray.serve.schema import ServeDeploySchema

from spatial_ray.policy.types import Action


def patch_replica_targets(config: dict[str, Any], targets: dict[str, int]) -> dict[str, Any]:
    """Copy a serveConfigV2 and override num_replicas on each named deployment.

    Args:
        config: A serveConfigV2 dict as compiled by compile_serve_config.
        targets: Desired num_replicas keyed by deployment name.

    Returns:
        A deep-copied config with matching num_replicas overridden and autoscaling dropped.
    """
    patched = copy.deepcopy(config)
    for application in patched.get("applications", []):
        for deployment in application.get("deployments", []):
            target = targets.get(deployment["name"])
            if target is not None:
                deployment["num_replicas"] = target
                deployment.pop("autoscaling_config", None)
    return patched


class RayActuator:
    """Applies replica targets by patching and re-submitting the live serveConfigV2."""

    def __init__(self, config: dict[str, Any], *, client: Any = None) -> None:
        # holds the last submitted config so each apply patches forward from the current desire
        self._config = copy.deepcopy(config)
        self._client = client

    def apply(self, action: Action) -> None:
        """Patch the held config with the action's targets and submit it for incremental reconcile.

        Args:
            action: The desired num_replicas for each pool to reconcile toward.
        """
        self._config = patch_replica_targets(self._config, action.targets)
        client = self._client if self._client is not None else _get_global_client()
        client.deploy_apps(ServeDeploySchema.model_validate(self._config), _blocking=False)

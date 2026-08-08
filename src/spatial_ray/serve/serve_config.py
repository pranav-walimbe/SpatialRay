"""
Compiles a pool grouping and inference spec into a plain serveConfigV2 dict.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from spatial_ray.serve.graph import (
    InferenceSpec,
    PoolSpec,
    deployment_options,
    ingress_deployment_options,
)


def compile_serve_config(
    grouping: Sequence[PoolSpec],
    inference: InferenceSpec,
    *,
    import_path: str,
    app_name: str = "spatialray",
    ingress_options: Mapping[str, Any] | None = None,
    autoscaling_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile a pool grouping and inference spec into a serveConfigV2 application.

    Args:
        grouping: Ordered pool specs mapping stage groups onto preprocessing pools.
        inference: Spec for the inference pool.
        import_path: Module path to the bound Serve application, e.g. perf.cloud.app:app.
        app_name: Name of the compiled application.
        ingress_options: Serve options overriding the ingress defaults, none keeps them.
        autoscaling_policy: Optional Ray Serve application-level policy configuration.

    Returns:
        A serveConfigV2 dict, ready for yaml.safe_dump and serve deploy.
    """
    deployments = [deployment_options(spec) for spec in (*grouping, inference)]
    if autoscaling_policy is not None:
        static_deployments = [
            deployment["name"]
            for deployment in deployments
            if "autoscaling_config" not in deployment
        ]
        if static_deployments:
            names = ", ".join(static_deployments)
            raise ValueError(
                f"application autoscaling requires autoscaling_config for deployments: {names}"
            )
        deployments.append(
            {
                "name": "ingress",
                **ingress_deployment_options(ingress_options, fixed_autoscaling=True),
            }
        )
    application = {
        "name": app_name,
        "import_path": import_path,
        "deployments": deployments,
    }
    if autoscaling_policy is not None:
        application["autoscaling_policy"] = dict(autoscaling_policy)
    return {"applications": [application]}

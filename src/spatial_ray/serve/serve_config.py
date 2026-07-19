"""
Compiles a pool grouping and inference spec into a plain serveConfigV2 dict.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from spatial_ray.serve.graph import InferenceSpec, PoolSpec, deployment_options


def compile_serve_config(
    grouping: Sequence[PoolSpec],
    inference: InferenceSpec,
    *,
    import_path: str,
    app_name: str = "spatialray",
) -> dict[str, Any]:
    """Compile a pool grouping and inference spec into a serveConfigV2 application.

    Args:
        grouping: Ordered pool specs mapping stage groups onto preprocessing pools.
        inference: Spec for the inference pool.
        import_path: Module path to the bound Serve application, e.g. perf.cluster.app:app.
        app_name: Name of the compiled application.

    Returns:
        A serveConfigV2 dict, ready for yaml.safe_dump and serve deploy.
    """
    deployments = [deployment_options(spec) for spec in (*grouping, inference)]
    return {
        "applications": [
            {
                "name": app_name,
                "import_path": import_path,
                "deployments": deployments,
            }
        ]
    }

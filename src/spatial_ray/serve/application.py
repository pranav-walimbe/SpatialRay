"""
One pool grouping and inference spec rendered as both a bound graph and a matching serveConfigV2.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from spatial_ray.serve.graph import InferenceSpec, PoolSpec, build_graph
from spatial_ray.serve.serve_config import compile_serve_config

DEFAULT_APP_NAME = "spatialray"


@dataclass(frozen=True)
class Application:
    """One pool grouping and inference spec, rendered as both a bound graph and a serveConfigV2."""

    grouping: Sequence[PoolSpec]  # ordered preprocessing pools this application binds
    inference: InferenceSpec  # the inference pool spec
    import_path: str  # module path Ray re-imports to rebuild this graph, e.g. perf.cloud.app:app
    app_name: str = DEFAULT_APP_NAME  # name shared by the bound graph and the compiled config

    @property
    def graph(self):
        """Bind the grouping and inference spec into a runnable Serve application.

        Returns:
            The bound ingress application ready for serve.run or the serve run CLI.
        """
        return build_graph(self.grouping, inference=self.inference)

    @property
    def serve_config(self) -> dict[str, Any]:
        """Compile the grouping and inference spec into a serveConfigV2 dict.

        Returns:
            A serveConfigV2 dict whose import_path re-imports this same application.
        """
        return compile_serve_config(
            self.grouping,
            self.inference,
            import_path=self.import_path,
            app_name=self.app_name,
        )

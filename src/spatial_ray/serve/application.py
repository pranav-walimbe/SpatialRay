"""
One pool grouping and inference spec rendered as both a bound graph and a matching serveConfigV2.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from spatial_ray.scaling.ray import RayPoolScalingConfig, application_policy_config
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
    ingress_options: Mapping[str, Any] = field(default_factory=dict)  # ingress Serve options
    autoscaling_policy: Mapping[str, Any] | None = None  # application-level Ray scaling policy

    @property
    def graph(self):
        """Bind the grouping and inference spec into a runnable Serve application.

        Returns:
            The bound ingress application ready for serve.run or the serve run CLI.
        """
        return build_graph(
            self.grouping, inference=self.inference, ingress_options=self.ingress_options
        )

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
            ingress_options=self.ingress_options,
            autoscaling_policy=self.autoscaling_policy,
        )

    def with_workload_autoscaling(self, pools: Mapping[str, RayPoolScalingConfig]) -> Application:
        """Return this application configured for coordinated workload autoscaling.

        Args:
            pools: Ray scaling and capacity configuration keyed by every scalable pool name.

        Returns:
            A new application carrying per-pool Ray envelopes and the coordinated policy.
        """
        expected = {spec.name for spec in (*self.grouping, self.inference)}
        configured = set(pools)
        if configured != expected:
            missing = ", ".join(sorted(expected - configured)) or "none"
            unknown = ", ".join(sorted(configured - expected)) or "none"
            raise ValueError(
                f"pool scaling config mismatch, missing: {missing}, unknown: {unknown}"
            )
        grouping = tuple(
            replace(spec, autoscaling_config=pools[spec.name].deployment_config())
            for spec in self.grouping
        )
        inference = replace(
            self.inference,
            autoscaling_config=pools[self.inference.name].deployment_config(),
        )
        capacities = {name: config.capacity for name, config in pools.items()}
        return replace(
            self,
            grouping=grouping,
            inference=inference,
            autoscaling_policy=application_policy_config(capacities),
        )

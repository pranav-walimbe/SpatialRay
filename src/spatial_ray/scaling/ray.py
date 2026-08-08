"""
The Ray Serve adapter for coordinated stage-work autoscaling.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any

from spatial_ray.scaling.metrics import LEDGER_SNAPSHOT_AGE, pending_work_metric, work_rate_metric
from spatial_ray.scaling.policy import CapacityPolicy, PoolCapacity, PoolLoad

POLICY_IMPORT_PATH = "spatial_ray.scaling.ray:RayWorkloadAutoscalingPolicy"


@dataclass(frozen=True)
class RayPoolScalingConfig:
    """Configure one pool's capacity model and Ray Serve scaling envelope."""

    capacity: PoolCapacity
    min_replicas: int = 0
    max_replicas: int = 100
    initial_replicas: int | None = None
    upscale_delay_s: float = 0.0
    downscale_delay_s: float = 300.0

    def __post_init__(self) -> None:
        # validate Ray's replica envelope before rendering application configuration
        if self.min_replicas < 0:
            raise ValueError(f"min_replicas must be nonnegative, got {self.min_replicas}")
        if self.max_replicas < self.min_replicas:
            raise ValueError(f"max_replicas must be at least min_replicas, got {self.max_replicas}")
        if self.initial_replicas is not None and not (
            self.min_replicas <= self.initial_replicas <= self.max_replicas
        ):
            raise ValueError("initial_replicas must be within the configured replica range")
        if self.upscale_delay_s < 0.0 or self.downscale_delay_s < 0.0:
            raise ValueError("autoscaling delays must be nonnegative")

    def deployment_config(self) -> dict[str, Any]:
        """Render this pool's Ray Serve autoscaling configuration.

        Returns:
            A JSON-safe Ray Serve autoscaling_config mapping.
        """
        config: dict[str, Any] = {
            "min_replicas": self.min_replicas,
            "max_replicas": self.max_replicas,
            "upscale_delay_s": self.upscale_delay_s,
            "downscale_delay_s": self.downscale_delay_s,
        }
        if self.initial_replicas is not None:
            config["initial_replicas"] = self.initial_replicas
        return config


def application_policy_config(
    capacities: Mapping[str, PoolCapacity],
    *,
    ingress: str = "ingress",
    max_snapshot_age_s: float = 2.0,
) -> dict[str, Any]:
    """Render a JSON-safe Ray Serve application autoscaling policy.

    Args:
        capacities: Profiled capacity keyed by pool name.
        ingress: Deployment reporting application work arrival rates.
        max_snapshot_age_s: Oldest exact ledger snapshot accepted for a decision.

    Returns:
        Ray Serve autoscaling policy configuration for a compiled application.
    """
    _validate_snapshot_age(max_snapshot_age_s)
    return {
        "policy_function": POLICY_IMPORT_PATH,
        "policy_kwargs": {
            "capacities": {
                name: {
                    "work_per_s": profile.work_per_s,
                    "target_utilization": profile.target_utilization,
                    "target_queue_s": profile.target_queue_s,
                }
                for name, profile in capacities.items()
            },
            "ingress": ingress,
            "max_snapshot_age_s": max_snapshot_age_s,
        },
    }


class RayWorkloadAutoscalingPolicy:
    """Adapt Ray application contexts to the runtime-neutral capacity policy."""

    def __init__(
        self,
        capacities: Mapping[str, Mapping[str, float]],
        ingress: str = "ingress",
        max_snapshot_age_s: float = 2.0,
    ):
        self._policy = CapacityPolicy(
            capacities={name: PoolCapacity(**dict(profile)) for name, profile in capacities.items()}
        )
        self._ingress = ingress
        _validate_snapshot_age(max_snapshot_age_s)
        self._max_snapshot_age_s = max_snapshot_age_s

    def __call__(self, contexts: Mapping[Any, Any]):
        """Return coordinated Ray Serve replica targets for one application.

        Args:
            contexts: Deployment IDs mapped to Ray Serve autoscaling contexts.

        Returns:
            Replica targets and preserved per-deployment policy state.
        """
        by_name = {
            context.deployment_name: (deployment_id, context)
            for deployment_id, context in contexts.items()
        }
        ingress_entry = by_name.get(self._ingress)
        decisions = {
            deployment_id: context.target_num_replicas
            for deployment_id, context in contexts.items()
        }
        state = {
            deployment_id: dict(context.policy_state) for deployment_id, context in contexts.items()
        }
        if ingress_entry is None:
            return decisions, state
        _, ingress_context = ingress_entry
        loads = self._loads(by_name, ingress_context)
        for name, demand in self._policy.decide(loads).items():
            deployment_id, _ = by_name[name]
            decisions[deployment_id] = demand
        return decisions, state

    def _loads(self, by_name: Mapping[str, tuple[Any, Any]], ingress_context: Any):
        # combine ingress work rates with one fresh exact ledger snapshot
        loads = {}
        reporter = _freshest_ledger_reporter(ingress_context, self._max_snapshot_age_s)
        if reporter is None:
            return loads
        for name in self._policy.capacities:
            pool_entry = by_name.get(name)
            work_rate = _metric_sum(ingress_context, work_rate_metric(name))
            pending_work = _metric_for_reporter(
                ingress_context, pending_work_metric(name), reporter
            )
            if pool_entry is None or work_rate is None or pending_work is None:
                continue
            loads[name] = PoolLoad(
                work_per_s=work_rate,
                pending_work=max(0.0, pending_work),
            )
        return loads


def _metric_sum(context: Any, name: str) -> float | None:
    # sum one ingress metric across all reporting replicas
    metrics = context.aggregated_metrics or {}
    replicas = metrics.get(name)
    if replicas is None:
        return None
    return sum(replicas.values())


def _freshest_ledger_reporter(context: Any, max_age_s: float) -> Any | None:
    # choose one ingress replica so a shared ledger snapshot is never counted more than once
    metrics = context.aggregated_metrics or {}
    ages = metrics.get(LEDGER_SNAPSHOT_AGE)
    if not ages:
        return None
    reporter, age = min(ages.items(), key=lambda item: item[1])
    return reporter if 0.0 <= age <= max_age_s else None


def _metric_for_reporter(context: Any, name: str, reporter: Any) -> float | None:
    # read the metric belonging to the ingress replica chosen by snapshot freshness
    metrics = context.aggregated_metrics or {}
    return (metrics.get(name) or {}).get(reporter)


def _validate_snapshot_age(max_snapshot_age_s: float) -> None:
    # reject a policy window that could admit stale or nonsensical ledger state
    if not isfinite(max_snapshot_age_s) or max_snapshot_age_s <= 0.0:
        raise ValueError("max_snapshot_age_s must be finite and positive")

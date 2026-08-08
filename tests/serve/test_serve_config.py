"""
Tests compiling a pool grouping and inference spec into a serveConfigV2 dict.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from spatial_ray.serve.graph import DISAGGREGATED, InferenceSpec
from spatial_ray.serve.serve_config import compile_serve_config


def test_compile_serve_config():
    """Lists one deployment per pool and carries each pool's knobs."""
    inference = InferenceSpec(model_factory=lambda: None)
    decode, transform = DISAGGREGATED
    grouping = (replace(decode, max_ongoing_requests=12), transform)
    config = compile_serve_config(grouping, inference, import_path="perf.cloud.app:app")
    app = config["applications"][0]
    assert app["import_path"] == "perf.cloud.app:app"
    deployments = {d["name"]: d for d in app["deployments"]}
    assert list(deployments) == ["decode", "transform", "inference"]
    assert deployments["decode"]["max_ongoing_requests"] == 12


def test_every_deployment_writes_its_gauge_inline():
    """Each replica gets the env var switching Serve off its event-loop-cached gauge export."""
    config = compile_serve_config(
        DISAGGREGATED, InferenceSpec(model_factory=lambda: None), import_path="perf.cloud.app:app"
    )
    for deployment in config["applications"][0]["deployments"]:
        env_vars = deployment["ray_actor_options"]["runtime_env"]["env_vars"]
        assert env_vars["RAY_SERVE_METRICS_EXPORT_INTERVAL_MS"] == "0"


def test_compile_application_autoscaling_policy():
    """The compiled application carries a coordinated Ray policy configuration."""
    policy = {"policy_function": "pkg.policy:scale", "policy_kwargs": {"target": 0.7}}
    grouping = tuple(
        replace(spec, autoscaling_config={"min_replicas": 0, "max_replicas": 10})
        for spec in DISAGGREGATED
    )
    inference = InferenceSpec(
        model_factory=lambda: None,
        autoscaling_config={"min_replicas": 0, "max_replicas": 10},
    )
    config = compile_serve_config(
        grouping,
        inference,
        import_path="perf.cloud.app:app",
        autoscaling_policy=policy,
    )
    application = config["applications"][0]
    assert application["autoscaling_policy"] == policy
    ingress = application["deployments"][-1]
    assert ingress["name"] == "ingress"
    assert ingress["autoscaling_config"] == {
        "min_replicas": 1,
        "initial_replicas": 1,
        "max_replicas": 1,
    }


def test_application_policy_rejects_static_pools():
    """An application policy rejects pools that Ray omits from autoscaling contexts."""
    with pytest.raises(ValueError, match="decode, transform, inference"):
        compile_serve_config(
            DISAGGREGATED,
            InferenceSpec(model_factory=lambda: None),
            import_path="perf.cloud.app:app",
            autoscaling_policy={"policy_function": "pkg.policy:scale"},
        )

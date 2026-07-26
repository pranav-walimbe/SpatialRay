"""
Sizes the disaggregated Serve pools from the cloud config and binds a runnable graph.
"""

from __future__ import annotations

import dataclasses
import os

from perf.cloud.utils import load_config
from perf.common.models import DEFAULT_MODEL, load
from spatial_ray.serve.graph import DISAGGREGATED, InferenceSpec, build_graph

APP_NAME = "spatialray"
_HARDWARE_ENV = "SPATIALRAY_HARDWARE"
_MODEL_ENV = "SPATIALRAY_MODEL"


def sized_specs(model_name: str, hardware: str):
    """Size each pool from the cloud config and pin it to its per-stage node.

    Args:
        model_name: Model module name under perf.common.models for the inference pool.
        hardware: Target hardware, cpu or gpu, selecting the inference replica's device.

    Returns:
        The preprocessing pool specs and the inference spec, each sized and node-pinned.
    """
    pools_cfg = load_config()["pools"]
    pools = tuple(_sized_pool(spec, pools_cfg[spec.name]) for spec in DISAGGREGATED)
    return pools, _inference_spec(pools_cfg, model_name, hardware)


def build_app(model_name: str = DEFAULT_MODEL, hardware: str = "cpu"):
    """Size the pools and bind the disaggregated graph into a runnable Serve application.

    Args:
        model_name: Model module name under perf.common.models for the inference pool.
        hardware: Target hardware, cpu or gpu, selecting the inference replica's device.

    Returns:
        The bound ingress application ready for serve.run or the serve run CLI.
    """
    pools, inference = sized_specs(model_name, hardware)
    return build_graph(pools, inference=inference)


def _sized_pool(spec, pool_cfg):
    # replica count from config and a fractional node resource pinning the pool to its stage node
    options = {"num_cpus": pool_cfg["num_cpus"], "resources": {f"{spec.name}_node": 0.01}}
    return dataclasses.replace(spec, num_replicas=pool_cfg["replicas"], ray_actor_options=options)


def _inference_spec(pools_cfg, model_name, hardware):
    # inference pool sized from config with its replica pinned to the inference node
    inference_cfg = pools_cfg["inference"]
    variant = inference_cfg[hardware]
    options = {"resources": {"inference_node": 0.01}}
    for key in ("num_cpus", "num_gpus"):
        if key in variant:
            options[key] = variant[key]
    return InferenceSpec(
        model_factory=_model_factory(model_name),
        num_replicas=inference_cfg["replicas"],
        ray_actor_options=options,
    )


def _model_factory(model_name):
    # zero-arg factory that Serve cloudpickles to rebuild the model on each inference replica
    def build():
        return load(model_name).build()

    return build


app = build_app(os.environ.get(_MODEL_ENV, DEFAULT_MODEL), os.environ.get(_HARDWARE_ENV, "cpu"))

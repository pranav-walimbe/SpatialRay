"""
Sizes the disaggregated Serve pools from the cloud config into a runnable Application.
"""

from __future__ import annotations

import dataclasses
import os

from perf.cloud.utils import load_config
from perf.common.models import DEFAULT_MODEL, load
from spatial_ray.control.ray_metrics import node_resource
from spatial_ray.serve.application import Application
from spatial_ray.serve.graph import DISAGGREGATED, InferenceSpec

APP_NAME = "spatialray"
_IMPORT_PATH = "perf.cloud.app:app"
_HARDWARE_ENV = "SPATIALRAY_HARDWARE"
_MODEL_ENV = "SPATIALRAY_MODEL"


def from_config(model_name: str = DEFAULT_MODEL, hardware: str = "cpu") -> Application:
    """Size the disaggregated pools from the cloud config into a runnable Application.

    Args:
        model_name: Model module name under perf.common.models for the inference pool.
        hardware: Target hardware, cpu or gpu, selecting the inference replica's device.

    Returns:
        An Application binding the sized grouping and inference spec.
    """
    pools_cfg = load_config()["pools"]
    grouping = tuple(_sized_pool(spec, pools_cfg[spec.name]) for spec in DISAGGREGATED)
    inference = _inference_spec(pools_cfg, model_name, hardware)
    return Application(grouping, inference, import_path=_IMPORT_PATH, app_name=APP_NAME)


def _sized_pool(spec, pool_cfg):
    # replica count from config and a fractional node resource pinning the pool to its stage node
    options = {"num_cpus": pool_cfg["num_cpus"], "resources": {node_resource(spec.name): 0.01}}
    return dataclasses.replace(spec, num_replicas=pool_cfg["replicas"], ray_actor_options=options)


def _inference_spec(pools_cfg, model_name, hardware):
    # inference pool sized from config with its replica pinned to the inference node
    inference_cfg = pools_cfg["inference"]
    variant = inference_cfg[hardware]
    options = {"resources": {node_resource("inference"): 0.01}}
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


app = from_config(
    os.environ.get(_MODEL_ENV, DEFAULT_MODEL), os.environ.get(_HARDWARE_ENV, "cpu")
).graph

"""
Boots the disaggregated EC2 Ray cluster from config.yaml and collects the head's metrics figure.
"""

from __future__ import annotations

import time
import uuid
from argparse import ArgumentParser
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from perf.cloud.utils import load_config
from perf.common.models import DEFAULT_MODEL

_BOOTSTRAP = Path(__file__).parent / "bootstrap.sh"
_ASSETS_DIR = Path(__file__).parents[2] / "assets"
_RESULT_PREFIX = "spatialray-runs"
_POLL_SECONDS = 20


def main() -> None:
    """Boot the disaggregated cluster for the chosen hardware and fetch its metrics figure."""
    parser = ArgumentParser(description="Boot an EC2 Ray cluster and run the perf measurement.")
    parser.add_argument(
        "--hardware", default="cpu", choices=("cpu", "gpu"), help="cpu or single-T4 inference node"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="model module under perf.common.models"
    )
    parser.add_argument(
        "--requests", type=int, default=1000, help="number of requests the Poisson trace generates"
    )
    parser.add_argument(
        "--rate", type=float, default=1.0, help="mean Poisson arrival rate in requests/s"
    )
    args = parser.parse_args()

    cfg = load_config()
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    region = cfg["region"]
    ec2 = boto3.client("ec2", region_name=region)
    ssm = boto3.client("ssm", region_name=region)
    s3 = boto3.client("s3", region_name=region)

    figure_path = _ASSETS_DIR / f"perf-{args.hardware}-{args.model}.png"
    instance_ids = _launch_cluster(ec2, ssm, cfg, run_id, args)
    print(f"launched {len(instance_ids)} nodes ({args.hardware}) run {run_id}: {instance_ids}")
    try:
        _wait_for_success(s3, ec2, cfg, run_id, instance_ids)
        _download(s3, cfg, run_id, "result.png", figure_path)
        print(f"wrote {figure_path}")
    finally:
        ec2.terminate_instances(InstanceIds=instance_ids)
        print(f"terminated {instance_ids}")


def _cluster_nodes(cfg, hardware):
    # The decode and transform worker nodes plus the inference head node
    pools = cfg["pools"]
    inference = pools["inference"][hardware]
    return [
        ("decode", pools["decode"]["instance_type"], pools["decode"]["ami_ssm"], False),
        ("transform", pools["transform"]["instance_type"], pools["transform"]["ami_ssm"], False),
        ("inference", inference["instance_type"], inference["ami_ssm"], True),
    ]


def _launch_cluster(ec2, ssm, cfg, run_id, args):
    # Boot one tagged instance per cluster node and return their instance ids
    nodes = _cluster_nodes(cfg, args.hardware)
    expected = len(nodes)
    instance_ids = []
    for role, instance_type, ami_key, is_head in nodes:
        ami = _resolve_ami(ssm, cfg[ami_key])
        user_data = _render_bootstrap(cfg, run_id, args, role, is_head, expected)
        instance_ids.append(_run_instance(ec2, cfg, run_id, role, ami, instance_type, user_data))
    return instance_ids


def _run_instance(ec2, cfg, run_id, role, ami, instance_type, user_data):
    # Start one terminate-on-shutdown instance for a cluster role
    root_device = ec2.describe_images(ImageIds=[ami])["Images"][0]["RootDeviceName"]
    response = ec2.run_instances(
        ImageId=ami,
        InstanceType=instance_type,
        MinCount=1,
        MaxCount=1,
        UserData=user_data,
        InstanceInitiatedShutdownBehavior="terminate",
        IamInstanceProfile={"Name": cfg["instance_profile"]},
        BlockDeviceMappings=[
            {
                "DeviceName": root_device,
                "Ebs": {
                    "VolumeSize": cfg["volume_gb"],
                    "VolumeType": "gp3",
                    "DeleteOnTermination": True,
                },
            }
        ],
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Project", "Value": "spatialray-perf"},
                    {"Key": "RunId", "Value": run_id},
                    {"Key": "Role", "Value": role},
                ],
            }
        ],
    )
    return response["Instances"][0]["InstanceId"]


def _resolve_ami(ssm, parameter):
    # Look up the current AMI id behind an SSM public parameter path
    return ssm.get_parameter(Name=parameter)["Parameter"]["Value"]


def _render_bootstrap(cfg, run_id, args, role, is_head, expected_nodes):
    # Substitute the @@NAME@@ placeholders in bootstrap.sh with this node's values
    replacements = {
        "@@RUN_ID@@": run_id,
        "@@REGION@@": cfg["region"],
        "@@RESULT_BUCKET@@": cfg["result_bucket"],
        "@@RESULT_PREFIX@@": _RESULT_PREFIX,
        "@@REPO_URL@@": cfg["repo_url"],
        "@@REPO_BRANCH@@": cfg["repo_branch"],
        "@@MODEL@@": args.model,
        "@@HARDWARE@@": args.hardware,
        "@@REQUESTS@@": str(args.requests),
        "@@RATE@@": str(args.rate),
        "@@MAX_RUNTIME_MIN@@": str(cfg["max_runtime_mins"]),
        "@@ROLE@@": role,
        "@@IS_HEAD@@": "1" if is_head else "0",
        "@@EXPECTED_NODES@@": str(expected_nodes),
    }
    script = _BOOTSTRAP.read_text()
    for key, value in replacements.items():
        script = script.replace(key, value)
    return script


def _wait_for_success(s3, ec2, cfg, run_id, instance_ids):
    # Poll for the _SUCCESS marker while streaming progress and abort if a node dies first
    bucket = cfg["result_bucket"]
    success_key = f"{_RESULT_PREFIX}/{run_id}/_SUCCESS"
    progress_key = f"{_RESULT_PREFIX}/{run_id}/progress.log"
    deadline = time.monotonic() + (cfg["max_runtime_mins"] + 10) * 60
    seen = 0
    while time.monotonic() < deadline:
        seen = _emit_progress(s3, bucket, progress_key, seen)
        if _exists(s3, bucket, success_key):
            return
        dead = _dead_nodes(ec2, instance_ids)
        if dead:
            raise RuntimeError(f"run {run_id} aborted: nodes terminated before success: {dead}")
        time.sleep(_POLL_SECONDS)
    raise TimeoutError(f"run {run_id} did not finish within the deadline")


def _dead_nodes(ec2, instance_ids):
    # Instance ids no longer pending or running, a node that died before writing _SUCCESS
    reservations = ec2.describe_instances(InstanceIds=instance_ids)["Reservations"]
    return [
        instance["InstanceId"]
        for reservation in reservations
        for instance in reservation["Instances"]
        if instance["State"]["Name"] not in ("pending", "running")
    ]


def _emit_progress(s3, bucket, key, seen):
    # Print progress-log lines not yet seen and return the new line count
    if not _exists(s3, bucket, key):
        return seen
    lines = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode().splitlines()
    for line in lines[seen:]:
        print(line)
    return len(lines)


def _exists(s3, bucket, key):
    # Return whether an object exists while treating a missing-key error as absent
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as error:
        if error.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def _download(s3, cfg, run_id, name, dest):
    # Download one of the run's result files to dest overwriting any prior file for this run kind
    dest.parent.mkdir(parents=True, exist_ok=True)
    key = f"{_RESULT_PREFIX}/{run_id}/{name}"
    s3.download_file(cfg["result_bucket"], key, str(dest))


if __name__ == "__main__":
    main()

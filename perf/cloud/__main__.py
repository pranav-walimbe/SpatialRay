"""
Launches an EC2 instance that runs the on-box perf measurement and collects its result.
"""

from __future__ import annotations

import time
import uuid
from argparse import ArgumentParser
from pathlib import Path

import boto3
import yaml
from botocore.exceptions import ClientError

from perf.common.models import DEFAULT_MODEL

_CONFIG = Path(__file__).parent / "config.yaml"
_BOOTSTRAP = Path(__file__).parent / "bootstrap.sh"
_ASSETS_DIR = Path(__file__).parents[2] / "assets"
_RESULT_PREFIX = "spatialray-runs"
_POLL_SECONDS = 20

# Map each hardware target to its instance-type and AMI keys in config.yaml
_INSTANCE = {
    "cpu": ("cpu_instance_type", "cpu_ami_ssm"),
    "gpu": ("gpu_instance_type", "gpu_ami_ssm"),
}


def main() -> None:
    """Boot an EC2 box for the chosen hardware, run the perf measurement, and fetch the result."""
    parser = ArgumentParser(description="Run the perf measurement on an EC2 instance.")
    parser.add_argument(
        "--hardware", default="cpu", choices=tuple(_INSTANCE), help="cpu or single-T4 gpu box"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="model module under perf.common.models"
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(_CONFIG.read_text())
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    region = cfg["region"]
    ec2 = boto3.client("ec2", region_name=region)
    ssm = boto3.client("ssm", region_name=region)
    s3 = boto3.client("s3", region_name=region)

    result_path = _ASSETS_DIR / f"perf-{args.hardware}-{args.model}.txt"
    instance_id = _launch(ec2, ssm, cfg, run_id, args.hardware, args.model)
    print(f"launched {instance_id} ({args.hardware}) run {run_id}")
    try:
        _wait_for_success(s3, cfg, run_id)
        _download_result(s3, cfg, run_id, result_path)
        print(result_path.read_text())
        print(f"wrote {result_path}")
    finally:
        ec2.terminate_instances(InstanceIds=[instance_id])
        print(f"terminated {instance_id}")


def _launch(ec2, ssm, cfg, run_id, hardware, model):
    # Resolve the AMI, render the bootstrap, and start one terminate-on-shutdown instance
    type_key, ami_key = _INSTANCE[hardware]
    ami = _resolve_ami(ssm, cfg[ami_key])
    root_device = ec2.describe_images(ImageIds=[ami])["Images"][0]["RootDeviceName"]
    response = ec2.run_instances(
        ImageId=ami,
        InstanceType=cfg[type_key],
        MinCount=1,
        MaxCount=1,
        UserData=_render_bootstrap(cfg, run_id, hardware, model),
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
                ],
            }
        ],
    )
    return response["Instances"][0]["InstanceId"]


def _resolve_ami(ssm, parameter):
    # Look up the current AMI id behind an SSM public parameter path
    return ssm.get_parameter(Name=parameter)["Parameter"]["Value"]


def _render_bootstrap(cfg, run_id, hardware, model):
    # Substitute the @@NAME@@ placeholders in bootstrap.sh with this run's values
    replacements = {
        "@@RUN_ID@@": run_id,
        "@@REGION@@": cfg["region"],
        "@@RESULT_BUCKET@@": cfg["result_bucket"],
        "@@RESULT_PREFIX@@": _RESULT_PREFIX,
        "@@REPO_URL@@": cfg["repo_url"],
        "@@REPO_BRANCH@@": cfg["repo_branch"],
        "@@MODEL@@": model,
        "@@HARDWARE@@": hardware,
        "@@MAX_RUNTIME_MIN@@": str(cfg["max_runtime_min"]),
    }
    script = _BOOTSTRAP.read_text()
    for key, value in replacements.items():
        script = script.replace(key, value)
    return script


def _wait_for_success(s3, cfg, run_id):
    # Poll the run's _SUCCESS marker, streaming progress lines, until it appears or times out
    bucket = cfg["result_bucket"]
    success_key = f"{_RESULT_PREFIX}/{run_id}/_SUCCESS"
    progress_key = f"{_RESULT_PREFIX}/{run_id}/progress.log"
    deadline = time.monotonic() + (cfg["max_runtime_min"] + 10) * 60
    seen = 0
    while time.monotonic() < deadline:
        seen = _emit_progress(s3, bucket, progress_key, seen)
        if _exists(s3, bucket, success_key):
            return
        time.sleep(_POLL_SECONDS)
    raise TimeoutError(f"run {run_id} did not finish within the deadline")


def _emit_progress(s3, bucket, key, seen):
    # Print progress-log lines not yet seen and return the new line count
    if not _exists(s3, bucket, key):
        return seen
    lines = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode().splitlines()
    for line in lines[seen:]:
        print(line)
    return len(lines)


def _exists(s3, bucket, key):
    # Return whether an object exists, treating a missing-key error as absent
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as error:
        if error.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def _download_result(s3, cfg, run_id, dest):
    # Download the run's result.txt to dest, overwriting any prior result for this run kind
    dest.parent.mkdir(parents=True, exist_ok=True)
    key = f"{_RESULT_PREFIX}/{run_id}/result.txt"
    s3.download_file(cfg["result_bucket"], key, str(dest))


if __name__ == "__main__":
    main()

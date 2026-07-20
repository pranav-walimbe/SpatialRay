#!/usr/bin/env bash
# Builds SpatialRay, runs the on-box concurrent load-test harness over public Sentinel-2 COGs,
# captures its summary to result.txt.

set -uo pipefail

export HOME=/root

RUN_ID="@@RUN_ID@@"
REGION="@@REGION@@"
RESULT_BUCKET="@@RESULT_BUCKET@@"
RESULT_PREFIX="@@RESULT_PREFIX@@"
REPO_URL="@@REPO_URL@@"
REPO_BRANCH="@@REPO_BRANCH@@"
MODEL="@@MODEL@@"
HARDWARE="@@HARDWARE@@"
MAX_RUNTIME_MIN="@@MAX_RUNTIME_MIN@@"

S3_BASE="s3://${RESULT_BUCKET}/${RESULT_PREFIX}/${RUN_ID}"
RESULT=/data/result.txt
LOG=/var/log/spatialray-bootstrap.log
exec > >(tee -a "$LOG") 2>&1
log() { echo "[bootstrap] $*"; }

# Always ship the log and self-terminate, whether we succeed or fail
cleanup() { aws s3 cp "$LOG" "${S3_BASE}/bootstrap.log" --region "$REGION" || true; shutdown -h now; }
trap cleanup EXIT

# Hard cap: terminate even if a step wedges
( sleep $((MAX_RUNTIME_MIN * 60)); log "watchdog timeout"; shutdown -h now ) &

# Publish the log to S3 every 15s so the launcher can show live step progress
( while true; do
    aws s3 cp "$LOG" "${S3_BASE}/progress.log" --region "$REGION" >/dev/null 2>&1 || true
    sleep 15
  done ) &

set -e

# The gpu box is an Ubuntu Deep Learning AMI (apt) and the cpu box is AL2023 (dnf), so install
# git and a compiler through whichever package manager the AMI provides.
log "installing packages"
if command -v dnf >/dev/null 2>&1; then
  dnf install -y gcc git >/dev/null
else
  apt-get update -y >/dev/null && apt-get install -y gcc git >/dev/null
fi

log "installing uv"
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

# Pin uv to a managed 3.11 for every sync and run
export UV_PYTHON=3.11

log "cloning ${REPO_URL} @ ${REPO_BRANCH}"
git clone --depth 1 --branch "$REPO_BRANCH" "$REPO_URL" /opt/spatialray
cd /opt/spatialray

log "building SpatialRay"

uv venv
uv pip install -e '.[perf]'

mkdir -p /data/scratch

# /tmp is tmpfs (RAM) on these AMIs, so spill the per-stage scratch to the EBS data volume
export TMPDIR=/data/scratch

export AWS_DEFAULT_REGION="$REGION"
log "measuring model=${MODEL} hardware=${HARDWARE}"

# result.txt gets stdout only
uv run --no-sync python -m perf.cloud.onbox --model "$MODEL" --hardware "$HARDWARE" > "$RESULT"

log "done"
aws s3 cp "$RESULT" "${S3_BASE}/result.txt" --region "$REGION"
aws s3 cp "$LOG" "${S3_BASE}/progress.log" --region "$REGION" || true
echo ok | aws s3 cp - "${S3_BASE}/_SUCCESS" --region "$REGION"

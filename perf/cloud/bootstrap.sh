#!/usr/bin/env bash
# Builds SpatialRay and joins the Ray cluster for its role while the head runs perf.cloud

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
REQUESTS="@@REQUESTS@@"
RATE="@@RATE@@"
MAX_RUNTIME_MIN="@@MAX_RUNTIME_MIN@@"
ROLE="@@ROLE@@"
IS_HEAD="@@IS_HEAD@@"
EXPECTED_NODES="@@EXPECTED_NODES@@"

S3_BASE="s3://${RESULT_BUCKET}/${RESULT_PREFIX}/${RUN_ID}"
HEAD_KEY="${S3_BASE}/head_ip"
FIGURE=/data/result.png
LOG=/var/log/spatialray-bootstrap.log

# the head streams to progress.log for the launcher and each worker streams to its own key
if [ "$IS_HEAD" = "1" ]; then STREAM_KEY="${S3_BASE}/progress.log"; else STREAM_KEY="${S3_BASE}/log-${ROLE}.log"; fi

exec > >(tee -a "$LOG") 2>&1
log() { echo "[bootstrap $ROLE] $*"; }

# flush this node's log to S3 and self-terminate on exit or when systemd sends SIGTERM
cleanup() { aws s3 cp "$LOG" "$STREAM_KEY" --region "$REGION" || true; shutdown -h now; }
trap cleanup EXIT TERM

# hard cap terminate even if a step wedges
( sleep $((MAX_RUNTIME_MIN * 60)); log "watchdog timeout"; shutdown -h now ) &

# every node streams its log to S3 every 15s so each node is watchable live
( while true; do aws s3 cp "$LOG" "$STREAM_KEY" --region "$REGION" >/dev/null 2>&1 || true; sleep 15; done ) &

set -e

# the gpu head is Ubuntu (apt) and the cpu workers are AL2023 (dnf)
log "installing packages"
if command -v dnf >/dev/null 2>&1; then
  dnf install -y gcc git >/dev/null
else
  apt-get update -y >/dev/null && apt-get install -y gcc git >/dev/null
fi

log "installing uv"
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
export UV_PYTHON=3.11

log "cloning ${REPO_URL} @ ${REPO_BRANCH}"
git clone --depth 1 --branch "$REPO_BRANCH" "$REPO_URL" /opt/spatialray
cd /opt/spatialray

log "building SpatialRay"
uv venv
uv pip install -e '.[perf]'

mkdir -p /data/scratch
# /tmp is tmpfs (RAM) on these AMIs so spill the per-stage scratch to the EBS data volume
export TMPDIR=/data/scratch
export AWS_DEFAULT_REGION="$REGION"

# this node's private ip via IMDSv2
TOKEN=$(curl -sX PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 300")
LOCAL_IP=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/local-ipv4)
RESOURCES="{\"${ROLE}_node\": 1}"

if [ "$IS_HEAD" = "1" ]; then
  log "starting ray head at ${LOCAL_IP}"
  uv run --no-sync ray start --head --node-ip-address="$LOCAL_IP" --port=6379 --resources="$RESOURCES"
  echo "$LOCAL_IP" | aws s3 cp - "$HEAD_KEY" --region "$REGION"

  log "waiting for ${EXPECTED_NODES} nodes to join"
  RAY_ADDRESS=auto uv run --no-sync python - "$EXPECTED_NODES" <<'PY'
import sys, time, ray
ray.init(address="auto")
expected = int(sys.argv[1])
while len([n for n in ray.nodes() if n["alive"]]) < expected:
    time.sleep(5)
PY

  log "running perf.cloud model=${MODEL} hardware=${HARDWARE} requests=${REQUESTS} rate=${RATE}"
  RAY_ADDRESS=auto uv run --no-sync python -m perf.cloud \
    --model "$MODEL" --hardware "$HARDWARE" --requests "$REQUESTS" --rate "$RATE" --out "$FIGURE"

  log "done"
  aws s3 cp "$FIGURE" "${S3_BASE}/result.png" --region "$REGION"
  aws s3 cp "$LOG" "${S3_BASE}/progress.log" --region "$REGION" || true
  echo ok | aws s3 cp - "${S3_BASE}/_SUCCESS" --region "$REGION"
else
  log "waiting for the head ip"
  until aws s3 cp "$HEAD_KEY" /tmp/head_ip --region "$REGION" >/dev/null 2>&1; do sleep 5; done
  HEAD_IP=$(cat /tmp/head_ip)

  log "joining ray head at ${HEAD_IP}"
  uv run --no-sync ray start --address="${HEAD_IP}:6379" --node-ip-address="$LOCAL_IP" --resources="$RESOURCES"

  # hold the box up so its replicas keep serving until the launcher tears the cluster down
  log "joined, idling until terminated"
  sleep $((MAX_RUNTIME_MIN * 60))
fi

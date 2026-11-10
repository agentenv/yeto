#!/usr/bin/env bash
# First minutes on a fresh ROCm sampler node (runs on the node as the login
# user): host facts, docker, the NVMe directories, the SGLang image pull and
# the base model in the background.
#
#   scripts/sampler_node/bootstrap.sh --image IMAGE --model MODEL [--model-dir DIR]
#       [--nvme-root /mnt/nvme] [--peer user@host:/path/to/model] [--hf-token-file FILE]
#       [--r2-env-file FILE] [--no-pull]
#       [--release-store DIR --release-prefix RUN [--release-peers host:port,...] [--serve-port P]
#        [--yeto-dir DIR] [--python PY]]
#
#   MODEL      an HF repo id (downloaded with `hf download`), or an HF bucket
#              path `hf://buckets/<org>/<name>` (synced with `hf buckets sync`)
#   --peer     copy the model from another node that already has it (rsync
#              over ssh) instead of from the Hub; the peer's ssh key must be
#              set up already
#   --r2-env-file  a file with the object store credentials (AWS_ACCESS_KEY_ID,
#              AWS_SECRET_ACCESS_KEY, AWS_ENDPOINT_URL, AWS_REGION, R2_BUCKET);
#              only its permissions are fixed here (0600), nothing is read
#
#   --release-store  start the policy release puller (`python -m yeto.rl.spot.s3_delivery
#              pull-loop`) in the background: releases come from the peers of
#              --release-peers first (near-region fan-out, yeto.rl.spot.peer_fanout) and
#              from the object store only when no peer has them; --serve-port serves this
#              node's placed releases to the other nodes. The store settings come from
#              --r2-env-file (S3_BUCKET / R2_BUCKET, AWS_ENDPOINT_URL, AWS_REGION) and
#              --release-prefix (the run's prefix in the bucket); the file is read only for
#              this (0600 enforced) and nothing from it is printed.
#
# Background jobs log to $NVME_ROOT/logs/{pull,model,releases}.log; `tail -f` them,
# while they run. Idempotent: re-running resumes.
# amd-smi (ROCm 6.1+) is required on the node: the launcher's card preflight runs
# `amd-smi static --json` (rocm-smi is the fallback) and bootstrap refuses a node
# without either tool.
set -euo pipefail
IMAGE=""; MODEL=""; MODEL_DIR=""; NVME_ROOT=/mnt/nvme; PEER=""; HF_TOKEN_FILE=""; R2_ENV_FILE=""; PULL=1
HERE=$(cd "$(dirname "$0")" && pwd)
RELEASE_STORE=""; RELEASE_PREFIX=""; RELEASE_PEERS=""; SERVE_PORT=""; YETO_DIR=$(cd "$HERE/../.." && pwd); PY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --image) IMAGE=$2; shift 2;;
    --model) MODEL=$2; shift 2;;
    --model-dir) MODEL_DIR=$2; shift 2;;
    --nvme-root) NVME_ROOT=$2; shift 2;;
    --peer) PEER=$2; shift 2;;
    --hf-token-file) HF_TOKEN_FILE=$2; shift 2;;
    --r2-env-file) R2_ENV_FILE=$2; shift 2;;
    --no-pull) PULL=0; shift;;
    --release-store) RELEASE_STORE=$2; shift 2;;
    --release-prefix) RELEASE_PREFIX=$2; shift 2;;
    --release-peers) RELEASE_PEERS=$2; shift 2;;
    --serve-port) SERVE_PORT=$2; shift 2;;
    --yeto-dir) YETO_DIR=$2; shift 2;;
    --python) PY=$2; shift 2;;
    -h|--help) sed -n '2,36p' "$0"; exit 0;;
    *) echo "unknown option $1" >&2; exit 64;;
  esac
done
[ -n "$IMAGE" ] && [ -n "$MODEL" ] || { echo "usage: $0 --image IMAGE --model MODEL [...]" >&2; exit 64; }
model_name=$(basename "${MODEL%/}")
MODEL_DIR=${MODEL_DIR:-$NVME_ROOT/models/$model_name}
SUDO=""; [ "$(id -u)" = 0 ] || SUDO=sudo

echo "== host"
hostname; uname -r; nproc; free -g | head -2
cat /opt/rocm/.info/version 2>/dev/null || echo "no /opt/rocm/.info/version"
if command -v amd-smi >/dev/null; then
  echo "amd-smi cards: $(amd-smi static --json 2>/dev/null | grep -c market_name || true)"
elif command -v rocm-smi >/dev/null; then
  echo "warning: no amd-smi (the launcher preflight falls back to rocm-smi)" >&2
  rocm-smi --showproductname 2>/dev/null | grep -i "card series" || true
else
  echo "neither amd-smi nor rocm-smi on PATH: the launcher cannot check the cards on this node" >&2
  exit 69
fi
command -v docker >/dev/null || { echo "docker is not installed" >&2; exit 69; }
df -h "$NVME_ROOT" 2>/dev/null | tail -1 || echo "no $NVME_ROOT mount"

echo "== directories"
$SUDO mkdir -p "$NVME_ROOT/models" "$NVME_ROOT/cache" "$NVME_ROOT/logs" "$NVME_ROOT/yeto"
$SUDO chown -R "$(id -u):$(id -g)" "$NVME_ROOT/models" "$NVME_ROOT/cache" "$NVME_ROOT/logs" "$NVME_ROOT/yeto"
if [ -n "$R2_ENV_FILE" ]; then
  [ -f "$R2_ENV_FILE" ] || { echo "$R2_ENV_FILE does not exist" >&2; exit 66; }
  chmod 600 "$R2_ENV_FILE"
  echo "store env file $R2_ENV_FILE: mode $(stat -c %a "$R2_ENV_FILE" 2>/dev/null || stat -f %Lp "$R2_ENV_FILE")"
fi

if [ "$PULL" = 1 ]; then
  echo "== image pull (background): $IMAGE"
  nohup $SUDO docker pull "$IMAGE" > "$NVME_ROOT/logs/pull.log" 2>&1 &
fi

echo "== model: $MODEL -> $MODEL_DIR (background)"
mkdir -p "$MODEL_DIR"
if [ -n "$PEER" ]; then
  nohup rsync -a --partial --info=progress2 -e "ssh -o StrictHostKeyChecking=accept-new" \
    "$PEER/" "$MODEL_DIR/" > "$NVME_ROOT/logs/model.log" 2>&1 &
else
  if [ ! -x "$HOME/hfenv/bin/hf" ]; then
    python3 -m venv "$HOME/hfenv" 2>/dev/null \
      || { $SUDO apt-get update -qq && $SUDO apt-get install -y -qq python3-venv && python3 -m venv "$HOME/hfenv"; }
    "$HOME/hfenv/bin/pip" install -q 'huggingface_hub[hf_xet]>=1.0' 2>&1 | tail -1
  fi
  "$HOME/hfenv/bin/hf" version 2>&1 | head -1
  token_env=""
  if [ -n "$HF_TOKEN_FILE" ]; then
    [ -f "$HF_TOKEN_FILE" ] || { echo "$HF_TOKEN_FILE does not exist" >&2; exit 66; }
    chmod 600 "$HF_TOKEN_FILE"
    token_env="HF_TOKEN=$(tr -d '\n' < "$HF_TOKEN_FILE")"
  fi
  case "$MODEL" in
    hf://buckets/*)
      nohup env HF_XET_HIGH_PERFORMANCE=1 $token_env "$HOME/hfenv/bin/hf" buckets sync "$MODEL" "$MODEL_DIR" \
        > "$NVME_ROOT/logs/model.log" 2>&1 &;;
    *)
      nohup env HF_XET_HIGH_PERFORMANCE=1 $token_env "$HOME/hfenv/bin/hf" download "$MODEL" --local-dir "$MODEL_DIR" \
        > "$NVME_ROOT/logs/model.log" 2>&1 &;;
  esac
fi

if [ -n "$RELEASE_STORE" ]; then
  echo "== policy releases: $RELEASE_STORE (background)"
  [ -n "$RELEASE_PREFIX" ] || { echo "--release-store needs --release-prefix RUN" >&2; exit 64; }
  [ -n "$R2_ENV_FILE" ] || { echo "--release-store needs --r2-env-file (bucket, endpoint, credentials)" >&2; exit 64; }
  [ -d "$YETO_DIR/yeto" ] || { echo "$YETO_DIR is not a yeto checkout (--yeto-dir)" >&2; exit 66; }
  if [ -z "$PY" ]; then
    if [ -x "$YETO_DIR/.venv/bin/python" ]; then PY="$YETO_DIR/.venv/bin/python"; else PY=python3; fi
  fi
  command -v rclone >/dev/null || echo "warning: rclone not on PATH; the release puller needs it" >&2
  mkdir -p "$RELEASE_STORE"
  peers_args=(); [ -n "$RELEASE_PEERS" ] && peers_args=(--peers "$RELEASE_PEERS")
  serve_args=(); [ -n "$SERVE_PORT" ] && serve_args=(--serve-port "$SERVE_PORT")
  (
    set -a
    # shellcheck disable=SC1090
    . "$R2_ENV_FILE"
    set +a
    export S3_BUCKET="${S3_BUCKET:-${R2_BUCKET:-}}" S3_PREFIX="$RELEASE_PREFIX" S3_REGION="${S3_REGION:-${AWS_REGION:-auto}}"
    # s3_delivery builds an env-only rclone remote named "s3w" (RCLONE_CONFIG_S3W_*);
    # an S3-compatible endpoint (R2) reaches it through these variables
    if [ -n "${AWS_ENDPOINT_URL:-}" ]; then
      export RCLONE_CONFIG_S3W_ENDPOINT="${RCLONE_CONFIG_S3W_ENDPOINT:-$AWS_ENDPOINT_URL}"
      export RCLONE_S3_ENDPOINT="${RCLONE_S3_ENDPOINT:-$AWS_ENDPOINT_URL}"
    fi
    [ -n "$S3_BUCKET" ] || { echo "the env file sets neither S3_BUCKET nor R2_BUCKET" >&2; exit 64; }
    export PYTHONPATH="$YETO_DIR${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
    nohup "$PY" -m yeto.rl.spot.s3_delivery pull-loop --store "$RELEASE_STORE" "${peers_args[@]}" "${serve_args[@]}"       > "$NVME_ROOT/logs/releases.log" 2>&1 &
    echo $! > "$NVME_ROOT/logs/releases.pid"
  )
  echo "release puller pid $(cat "$NVME_ROOT/logs/releases.pid"), log $NVME_ROOT/logs/releases.log"
fi
echo "started: image pull -> $NVME_ROOT/logs/pull.log, model -> $NVME_ROOT/logs/model.log"

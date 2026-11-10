#!/usr/bin/env bash
# Start one SGLang server (ROCm image) on a sampler node, in docker.
#
#   scripts/sampler_node/run_sglang.sh --image IMAGE --model-dir DIR [--served-name NAME]
#       [--tp 8] [--gpus 0,1,...,7] [--port 30000] [--host 0.0.0.0] [--name sglang-0]
#       [--cache-dir /mnt/nvme/cache/sglang-0] [--r2-env-file FILE] [--no-routed-experts]
#       [--no-fp32-lm-head] [--attn-dp-size N] [--moe-a2a-backend deepep|none]
#       [--mem-fraction 0.85] [--shm-size 64g] [--foreground] [-- EXTRA SGLANG ARGS...]
#
# Defaults follow the Kimi K3.1 RL sampler contract (INTERFACES 2.6 / 2.8, S1):
#   --enable-return-routed-experts   the server returns the routed expert ids
#   --enable-fp32-lm-head            fp32 logits for score-centering top-k logprobs
#   --attn-dp-size TP                attention data parallel over the TP group
#                                    (--attn-dp-size 0 turns it off)
#   --moe-a2a-backend deepep         MoE all-to-all through DeepEP (none: off)
#   ROCm environment: ROCM_QUICK_REDUCE_QUANTIZATION=NONE SGLANG_ROCM_RUNTIME_RESERVE_GB=10
#                     AITER_SITUV2_A8W4=1 HSA_NO_SCRATCH_RECLAIM=1  (no MX_PATCH)
# Every value is a parameter; nothing here names a site, a host or a model.
# --r2-env-file passes the object store credentials into the container as its
# environment (docker --env-file); the file must be mode 0600 and is never
# printed. The model directory is mounted read-only.
set -euo pipefail
IMAGE=""; MODEL_DIR=""; SERVED=""; TP=8; GPUS=""; PORT=30000; HOST=0.0.0.0; NAME="sglang-0"
CACHE_DIR=""; R2_ENV_FILE=""; ROUTED=1; FP32_HEAD=1; ATTN_DP=""; MOE_A2A=deepep; MEM_FRACTION=""; SHM=64g; DETACH=1
while [ $# -gt 0 ]; do
  case "$1" in
    --image) IMAGE=$2; shift 2;;
    --model-dir) MODEL_DIR=$2; shift 2;;
    --served-name) SERVED=$2; shift 2;;
    --tp) TP=$2; shift 2;;
    --gpus) GPUS=$2; shift 2;;
    --port) PORT=$2; shift 2;;
    --host) HOST=$2; shift 2;;
    --name) NAME=$2; shift 2;;
    --cache-dir) CACHE_DIR=$2; shift 2;;
    --r2-env-file) R2_ENV_FILE=$2; shift 2;;
    --no-routed-experts) ROUTED=0; shift;;
    --no-fp32-lm-head) FP32_HEAD=0; shift;;
    --attn-dp-size) ATTN_DP=$2; shift 2;;
    --moe-a2a-backend) MOE_A2A=$2; shift 2;;
    --mem-fraction) MEM_FRACTION=$2; shift 2;;
    --shm-size) SHM=$2; shift 2;;
    --foreground) DETACH=0; shift;;
    --) shift; break;;
    -h|--help) sed -n '2,20p' "$0"; exit 0;;
    *) echo "unknown option $1" >&2; exit 64;;
  esac
done
[ -n "$IMAGE" ] && [ -n "$MODEL_DIR" ] || { echo "usage: $0 --image IMAGE --model-dir DIR [...]" >&2; exit 64; }
[ -d "$MODEL_DIR" ] || { echo "model dir $MODEL_DIR does not exist" >&2; exit 66; }
SERVED=${SERVED:-$(basename "${MODEL_DIR%/}")}
if [ -z "$GPUS" ]; then GPUS=$(seq -s, 0 $((TP - 1))); fi
CACHE_DIR=${CACHE_DIR:-$(dirname "$MODEL_DIR")/../cache/$NAME}
mkdir -p "$CACHE_DIR"
SUDO=""; [ "$(id -u)" = 0 ] || docker info >/dev/null 2>&1 || SUDO=sudo

env_file_args=()
if [ -n "$R2_ENV_FILE" ]; then
  [ -f "$R2_ENV_FILE" ] || { echo "$R2_ENV_FILE does not exist" >&2; exit 66; }
  mode=$(stat -c %a "$R2_ENV_FILE" 2>/dev/null || stat -f %Lp "$R2_ENV_FILE")
  [ "$mode" = 600 ] || { echo "$R2_ENV_FILE must be mode 600 (is $mode)" >&2; exit 77; }
  env_file_args=(--env-file "$R2_ENV_FILE")
fi

sglang_args=(--model-path /models/"$SERVED" --served-model-name "$SERVED"
             --host "$HOST" --port "$PORT" --tp "$TP" --trust-remote-code)
[ "$ROUTED" = 1 ] && sglang_args+=(--enable-return-routed-experts)
[ "$FP32_HEAD" = 1 ] && sglang_args+=(--enable-fp32-lm-head)
ATTN_DP=${ATTN_DP:-$TP}
[ "$ATTN_DP" != 0 ] && sglang_args+=(--attn-dp-size "$ATTN_DP")
[ "$MOE_A2A" != none ] && sglang_args+=(--moe-a2a-backend "$MOE_A2A")
[ -n "$MEM_FRACTION" ] && sglang_args+=(--mem-fraction-static "$MEM_FRACTION")
sglang_args+=("$@")

run_mode=(-d --restart unless-stopped); [ "$DETACH" = 1 ] || run_mode=(--rm -it)
$SUDO docker rm -f "$NAME" >/dev/null 2>&1 || true
$SUDO docker run "${run_mode[@]}" --name "$NAME" --network host --ipc host --shm-size "$SHM" \
  --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined \
  --cap-add SYS_PTRACE --ulimit memlock=-1 \
  -v "$MODEL_DIR":/models/"$SERVED":ro -v "$CACHE_DIR":/root/.cache \
  "${env_file_args[@]}" \
  -e HIP_VISIBLE_DEVICES="$GPUS" \
  -e ROCM_QUICK_REDUCE_QUANTIZATION=NONE -e SGLANG_ROCM_RUNTIME_RESERVE_GB=10 \
  -e AITER_SITUV2_A8W4=1 -e HSA_NO_SCRATCH_RECLAIM=1 \
  -e PYTHONUNBUFFERED=1 \
  "$IMAGE" python3 -m sglang.launch_server "${sglang_args[@]}"
echo "started $NAME: GPUs $GPUS, TP $TP, http://$HOST:$PORT (model $SERVED)"

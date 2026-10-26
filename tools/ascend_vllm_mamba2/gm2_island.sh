#!/usr/bin/env bash
# G-M2 stage 2: one yeto verl island (FSDP2 trainer + colocated vLLM rollout) on n4, no syncer.
# usage: gm2_island.sh <tag>   env: DEVS (0,1) MODEL ROUNDS GROUPS SAMPLES RESP EXTRA
set -u
TAG=$1; W=/data/work/s28-gm2; RUN=$W/runs/$TAG; DEVS=${DEVS:-0,1}
NCARD=$(printf "%s" "$DEVS" | tr "," "\n" | grep -c .)
PORT=${PORT:-6431}
mkdir -p $RUN/home
echo acad9875a8bdfc81afbcb0a50d146b2630f44093 > $RUN/verl_sha.txt
[ -n "$(docker ps -aq -f name=^gm2_isl_$TAG$)" ] && { echo "gm2_isl_$TAG exists, not starting twice"; exit 0; }
[ -e $W/logs/isl_$TAG.log ] && { echo "log for $TAG exists, pick a new tag"; exit 0; }
docker run --rm --name gm2_isl_$TAG \
  -e ASCEND_VISIBLE_DEVICES=$DEVS --shm-size=64g --network host --ipc host -v /data:/data \
  -v $RUN/verl_sha.txt:/workspace/verl_sha.txt:ro -w $RUN \
  -e HOME=$RUN/home -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e PYTHONPATH=$W/verl-fork:$W/src:$W/pysite \
  -e YETO_VERL_ROOT=$W/verl-fork \
  -e YETO_ISLAND_HMAC_KEY=2222222222222222222222222222222222222222222222222222222222222222 \
  -e YETO_RL_COMPAT_GROUP=ascend-910b3-64g \
  -e HCCL_SOCKET_IFNAME=bond0 -e GLOO_SOCKET_IFNAME=bond0 -e HCCL_CONNECT_TIMEOUT=1800 -e HCCL_EXEC_TIMEOUT=1800 \
  -e ASCEND_GLOBAL_LOG_LEVEL=3 -e ASCEND_SLOG_PRINT_TO_STDOUT=0 -e PYTHONUNBUFFERED=1 \
  -e PYTORCH_NPU_ALLOC_CONF=${ALLOC:-expandable_segments:True} \
  -e RAY_ADDRESS=127.0.0.1:$PORT -e YETO_M2_DEBUG_RELOAD=${DBG:-0} -e TOKENIZERS_PARALLELISM=false -e WANDB_MODE=disabled -e HYDRA_FULL_ERROR=1 \
  ${IMG:-q38/verl-loop:8.5.2} bash -lc "
ray start --head --node-ip-address=127.0.0.1 --port=$PORT --num-cpus=32 --resources='{\"NPU\": $NCARD}' \
  --include-dashboard=false --temp-dir=$RUN/ray > $RUN/ray-start.log 2>&1 || exit 97
python3 -m yeto.rl.adapters.verl.island_entry \
  --model ${MODEL:-$W/models/nano-mini} --model-revision ${REV:-7a25511e79c83b254d30b52091725c3181d62935} ${TRC:-} \
  --data $W/data/x-tasks-400.jsonl --rl-single-island-no-sync --learner-id 0 \
  --reward-function ${REWARD:-yeto.rl.bringup_reward:score} \
  --global-rounds ${ROUNDS:-4} --groups-per-round ${GROUPS_:-2} --samples-per-group ${SAMPLES:-4} \
  --rollout-max-response-len ${RESP:-128} --event-tape $RUN/tape.jsonl \
  --inner-lr ${LR:-1e-5} --seq-len 1024 --seed 1 --parameter-mode full \
  --actor-num-gpus-per-node $NCARD ${EXTRA:-}
rc=\$?; ray stop --force >/dev/null 2>&1; exit \$rc
" > $W/logs/isl_$TAG.log 2>&1
echo "rc=$?" > $W/out/isl_$TAG.rc

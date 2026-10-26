#!/usr/bin/env bash
# usage: gm2_run.sh <tag>   env: DEVS (default 0,1,2,3) TP MAXTOK NBATCH EAGER GMU
set -u
TAG=$1; W=/data/work/s28-gm2; IMG=${IMG:-q38/verl-loop:8.5.2}
mkdir -p $W/out $W/logs
docker run --rm --name "gm2_$TAG" \
  -e ASCEND_VISIBLE_DEVICES="${DEVS:-0,1,2,3}" \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_PLUGINS=ascend,yeto_ascend_mamba2 \
  -e PYTHONPATH=$W/src:$W/src/tools/ascend_vllm_mamba2/site \
  -e YETO_ASCEND_MAMBA2="${YM2:-1}" \
  -e TASK_QUEUE_ENABLE=1 -e CPU_AFFINITY_CONF=1 -e PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_USE_MODELSCOPE=False \
  -e MODEL="${MODEL:-$W/models/nano}" -e TP="${TP:-4}" -e MAXTOK="${MAXTOK:-64}" -e NBATCH="${NBATCH:-0}" \
  -e EAGER="${EAGER:-1}" -e LOADFMT="${LOADFMT:-auto}" -e SLEEP="${SLEEP:-0}" -e GMU="${GMU:-0.8}" -e LONG="${LONG:-1}" -e MAXSEQ="${MAXSEQ:-32}" -e OUT="$W/out/${TAG}.json" \
  --shm-size=32g --ipc host -v /data:/data -w $W \
  "$IMG" bash -lc "python $W/tools/${SCRIPT:-gm2_gen.py}" > "$W/logs/${TAG}.log" 2>&1
echo "rc=$?" | tee "$W/out/${TAG}.rc"

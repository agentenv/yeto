#!/usr/bin/env bash
# G-120 (S29 D20 copy in the repo): bring up a 3-node Ray cluster (n1 head + n2/n3 workers) on npu1 for the
# 120B verl Megatron buffer-trainer island, 24 x 910B3 (TP4/PP3/EP8).
#
# NOT docker (G-EZ3's 16-card island was docker FSDP2). This uses the bare
# sft92 conda env + the G-FJ2/G-FT pysite312 overlay, the same stack that ran
# the Megatron engine on a single node (VERL-ASCEND-MEGATRON.md, VERL-MEGATRON-PP.md).
#
# /data is per-node local disk, so the yeto source, pysite and model must exist
# on every node. The 120B text checkpoint is at:
#   n1: /data/work/s28-g120/models/nemotron-120b-text   (copied 16:09Z)
#   n2,n3: /data/work/s27-ez/models/nemotron-120b-text   (G-EZ3, read-only)
#
# Usage on npu1:
#   HEAD_IP=<n1 ip> ROLE=head bash ray_cluster.sh            # run on n1
#   HEAD_IP=<n1 ip> MY_IP=<n2 ip> ROLE=worker NODE=n2 bash ray_cluster.sh
#   HEAD_IP=<n1 ip> MY_IP=<n3 ip> ROLE=worker NODE=n3 bash ray_cluster.sh
# or drive all three from n1 over ssh (see g120_up.sh, written after W1 reports).
set -o pipefail
# G-120C: YETOSRC picks the deployed yeto tree (default $MINE/yetosrc). Ray workers
# import yeto from THIS path, so the driver and the cluster must use the same one.
MINE=/data/work/s28-g120
HEAD_IP=${HEAD_IP:?HEAD_IP=<head node intranet IP>}  # n1 (VERL-ASCEND-RL.md §2)
RAY_PORT=${RAY_PORT:-6394}  # S29: give each run its own port (and the ports below) when clusters share nodes
ROLE=${ROLE:?ROLE=head|worker}
RUN=${RUN:-g120-1}
OUT=$MINE/runs/$RUN
mkdir -p "$OUT/home" "$OUT/ray" "$MINE/triton_cache" "$MINE/stage"

source /data/work/sft-env/activate_sft92.sh >/dev/null 2>&1
# pysite312 (megatron-core 0.18.2, megatron-bridge 0.5.0, mindspeed 0.18, ray 2.59)
# and the verl fork live under G-FJ2's tree; they are read-only here.
export PYTHONPATH="$MINE/pysite:/data/work/s27-fj2/pysite312:/data/work/s27-fj2/verl-fork:${YETOSRC:-$MINE/yetosrc}:$PYTHONPATH"
export PATH="/data/work/s27-fj2/pysite312/bin:$PATH"
export HOME="$OUT/home"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_SOCKET_IFNAME=bond0 GLOO_SOCKET_IFNAME=bond0
export HCCL_CONNECT_TIMEOUT=1800 HCCL_EXEC_TIMEOUT=1800
export ASCEND_GLOBAL_LOG_LEVEL=3 ASCEND_SLOG_PRINT_TO_STDOUT=0
export CUDA_DEVICE_MAX_CONNECTIONS=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=$MINE/triton_cache

# S29 D20: stop only THIS run's Ray processes (those of $OUT/ray, or listening on $RAY_PORT).
# `ray stop --force` kills every Ray process of the user on the node, including another
# task's cluster on the same nodes (10-31: the idle lp-e1 cluster on n1-n3).
own_ray_pids() { ps -eo pid,args | awk -v d="$OUT/ray" -v p=":$RAY_PORT" \
  '($0 ~ d || $0 ~ ("port=" substr(p, 2) "( |$)") || $0 ~ ("gcs-address=[0-9.]+" p)) && $0 ~ /ray/ && !/awk/ {print $1}'; }
for p in $(own_ray_pids); do kill "$p" 2>/dev/null; done; sleep 2
for p in $(own_ray_pids); do kill -9 "$p" 2>/dev/null; done
if [ "$ROLE" = head ]; then
  ray start --head --node-ip-address="$HEAD_IP" --port="$RAY_PORT" --num-cpus=64 \
    --resources='{"NPU": 8}' --include-dashboard=false --temp-dir="$OUT/ray" --ray-client-server-port=10101 --dashboard-agent-listen-port=52465 --dashboard-agent-grpc-port=52466 --runtime-env-agent-port=52467 --metrics-export-port=52468 --min-worker-port=30000 --max-worker-port=31999 \
    >"$OUT/ray-head.log" 2>&1
  grep -q "Ray runtime started" "$OUT/ray-head.log" || { echo "HEAD-ABORT"; tail -5 "$OUT/ray-head.log"; exit 97; }
  echo "head up $(date -u +%H:%M:%SZ)"
else
  MY_IP=${MY_IP:?MY_IP=<intranet IP of this worker>}
  until ray start --address="$HEAD_IP:$RAY_PORT" --node-ip-address="$MY_IP" --num-cpus=64 --temp-dir="$OUT/ray" --dashboard-agent-listen-port=52465 --dashboard-agent-grpc-port=52466 --runtime-env-agent-port=52467 --metrics-export-port=52468 --min-worker-port=30000 --max-worker-port=31999 \
        --resources='{"NPU": 8}' >>"$OUT/ray-$NODE.log" 2>&1; do sleep 5; done
  echo "worker $NODE joined $(date -u +%H:%M:%SZ)"
fi

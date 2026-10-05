#!/bin/bash
# m5 post-steps on the kept 2x4 L40S cluster after the RL launch returned (s1run.sh m5 calls it; also runnable by hand):
#   1. wait until both nodes have no compute apps (RL job gone), 2. cross-node NCCL all-reduce probe (s1m5nccl.py, world 8),
#   3. generation-only sglang TP8 server across both nodes (--nnodes 2) + rank snapshot + greedy /generate (s1m5gen.py label tp8),
#   4. single-node TP4 reference server on the head (label ref), 5. pull evidence, 6. sky down (KEEP_M5=1 keeps the cluster).
# usage: s1m5post.sh <run dir>   evidence: <run>/pulled/{m5nccl.json,m5gen.json,m5ranks-gen.jsonl,m5post-*.log}
set -u
R=$1; D=$(cd "$(dirname "$0")" && pwd); CL=$(cat $R/cluster.txt); W=$CL-worker1
export HOME=/home/michael; SKY=/home/michael/work/gpu-head/venv/bin/sky
S="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20"
MODEL=${M5_MODEL:-Qwen/Qwen3-0.6B}; REV=${M5_REV:-c1899de289a04d12100db370d81485cdf75e47ca}
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $R/m5post.log; }
ship() { for n in $CL $W; do for f in s1m5nccl.py s1m5gen.py s1m5ranks.py; do b=$(base64 -w0 $D/$f); timeout 60 $S $n "mkdir -p ~/yeto-rl && echo $b | base64 -d > ~/yeto-rl/$f"; done; done; }
# remote prelude: python with torch+sglang, the island NIC (same rule as yeto.launcher._DETECT_IFACE), NCCL over sockets
PRE='PY=; for p in /opt/sglang/bin/python python3; do $p -c "import torch, sglang" 2>/dev/null && { PY=$p; break; }; done
YETO_IFACE=${NCCL_SOCKET_IFNAME:-$(for d in /sys/class/net/*; do n=$(basename "$d"); case "$n" in lo|docker*|veth*|br-*|virbr*) continue;; esac; [ "$(cat "$d/operstate" 2>/dev/null)" = up ] && { echo "$n"; break; }; done)}
export NCCL_IB_DISABLE=1 NCCL_DEBUG=${NCCL_DEBUG:-INFO} NCCL_SOCKET_IFNAME=$YETO_IFACE GLOO_SOCKET_IFNAME=$YETO_IFACE; cd ~/yeto-rl'
mkdir -p $R/pulled
log "m5post start cluster=$CL"
for i in $(seq 1 18); do
  busy=0; for n in $CL $W; do c=$(timeout 60 $S $n 'nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l' 2>/dev/null || echo 99); [ "$c" = 0 ] || busy=1; done
  [ $busy = 0 ] && break; sleep 10
done
log "gpus_free=$([ $busy = 0 ] && echo yes || echo no)"; [ $busy = 0 ] || for n in $CL $W; do timeout 60 $S $n 'pkill -f sglang; pkill -f MegatronTrain; ray stop --force' >/dev/null 2>&1; done
ship
IP=$(timeout 60 $S $CL 'hostname -I' | awk '{print $1}'); log "head_ip=$IP"
# ---- 2. NCCL all-reduce across nodes
timeout 60 $S $W "$PRE; (setsid nohup timeout 600 \$PY s1m5nccl.py 1 2 $IP 29511 /tmp/m5nccl.json > ~/yeto-rl/m5nccl-n1.log 2>&1 &)"
timeout 660 $S $CL "$PRE; timeout 600 \$PY s1m5nccl.py 0 2 $IP 29511 ~/yeto-rl/m5nccl.json > ~/yeto-rl/m5nccl-n0.log 2>&1"; log "nccl rc=$?"
timeout 60 $S $CL 'cat ~/yeto-rl/m5nccl.json' > $R/pulled/m5nccl.json 2>/dev/null
for n in 0 1; do h=$([ $n = 0 ] && echo $CL || echo $W); timeout 60 $S $h "tail -c 200000 ~/yeto-rl/m5nccl-n$n.log" > $R/pulled/m5post-nccl-n$n.log 2>/dev/null; done
# ---- 3/4. generation-only servers
srv() {  # srv <host> <node_rank> <tp> <nnodes> <tag>
  timeout 60 $S $1 "$PRE; (setsid nohup \$PY -m sglang.launch_server --model-path $MODEL --revision $REV --trust-remote-code --tp $3 --nnodes $4 --node-rank $2 --dist-init-addr $IP:29512 --host 127.0.0.1 --port 30000 --mem-fraction-static 0.6 --random-seed 17 > ~/yeto-rl/m5srv-$5.log 2>&1 &)"
}
wait_health() { for i in $(seq 1 60); do timeout 20 $S $CL 'curl -sf http://127.0.0.1:30000/health >/dev/null' && return 0; sleep 10; done; return 1; }
stop_srv() { for n in $CL $W; do timeout 60 $S $n 'pkill -f sglang.launch_server; sleep 5; pkill -9 -f "sglang::" ; true' >/dev/null 2>&1; done; sleep 10; }
srv $W 1 8 2 tp8-n1; srv $CL 0 8 2 tp8-n0
if wait_health; then
  log "tp8 server healthy"
  for k in 0 1; do h=$([ $k = 0 ] && echo $CL || echo $W); timeout 60 $S $h "$PRE; \$PY s1m5ranks.py $k" >> $R/pulled/m5ranks-gen.jsonl 2>/dev/null; done
  timeout 900 $S $CL "$PRE; \$PY s1m5gen.py http://127.0.0.1:30000 tp8 8 2 ~/yeto-rl/m5gen.json" | tee -a $R/m5post.log
else log "tp8 server NOT healthy"; fi
for n in 0 1; do h=$([ $n = 0 ] && echo $CL || echo $W); timeout 60 $S $h "tail -c 300000 ~/yeto-rl/m5srv-tp8-n$n.log" > $R/pulled/m5post-srv-tp8-n$n.log 2>/dev/null; done
stop_srv
srv $CL 0 4 1 ref
if wait_health; then log "ref server healthy"; timeout 900 $S $CL "$PRE; \$PY s1m5gen.py http://127.0.0.1:30000 ref 4 1 ~/yeto-rl/m5gen.json" | tee -a $R/m5post.log
else log "ref server NOT healthy"; fi
timeout 60 $S $CL "tail -c 300000 ~/yeto-rl/m5srv-ref.log" > $R/pulled/m5post-srv-ref.log 2>/dev/null
stop_srv
timeout 60 $S $CL 'cat ~/yeto-rl/m5gen.json' > $R/pulled/m5gen.json 2>/dev/null
log "m5post done"
if [ "${KEEP_M5:-0}" != 1 ]; then HOME=/home/michael $SKY down -y $CL > $R/m5post-down.out 2>&1; log "sky down rc=$?"; fi
touch $R/M5POST_DONE

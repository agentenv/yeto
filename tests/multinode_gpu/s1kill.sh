#!/bin/bash
# G3 fault injection (runs on the local machine): wait for rl_driver_phase train of rollout_id>=1 on the head, then kill the worker's island raylet
# (pkill -f "[m]iles-ray/" on <cluster>-worker1; the bracket keeps pkill from matching the ssh bash -c carrying this command line) = node loss; record the kill time (worker clock) and keep pulling journal/probe for >= 200 s (window >= 150 s).
# KILL_RID (env, default 1): minimum rollout_id of the train phase that triggers the kill (m4a: 2).
R=$1; K=${KILL_RID:-1}; RE="([$K-9]|[1-9][0-9])"; CL=$(cat $R/cluster.txt); W=$CL-worker1; export HOME=/home/michael
S="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20"
log() { echo "[s1kill $(date -u +%FT%TZ)] $*"; }
until [ -f $R/rc.txt ]; do
  if timeout 30 $S $CL 'grep -E "\"event\": ?\"rl_driver_phase\"" ~/yeto-output/rl-island-0.jsonl 2>/dev/null | grep -E "\"phase\": ?\"train\"" | grep -qE "\"rollout_id\": ?'"$RE"'[,} ]"'; then break; fi
  sleep 3
done
[ -f $R/rc.txt ] && { log "launcher ended before the trigger; no kill"; exit 1; }
log "trigger seen (train, rollout_id>=$K); killing island raylet on $W"
timeout 60 $S $W 'date -u +%FT%TZ; date +%s.%N; pgrep -af "[m]iles-ray/" | cut -c1-120 | head -5; pkill -f "[m]iles-ray/"; sleep 1; echo killed; date +%s.%N; pgrep -c -f "[m]iles-ray/"' > $R/kill.txt 2>&1
log "kill rc=$? $(sed -n 2p $R/kill.txt)"
for i in $(seq 1 40); do   # 200 s post-kill window, 5 s cadence (terminal-state evidence from inside the head container)
  timeout 30 $S $CL 'cat ~/yeto-rl/elastic-state/reconfig/journal.jsonl 2>/dev/null' > $R/pulled/.j2 2>/dev/null && [ -s $R/pulled/.j2 ] && mv $R/pulled/.j2 $R/pulled/journal.jsonl
  timeout 30 $S $CL 'cat ~/yeto-rl/s1probe.log 2>/dev/null' > $R/pulled/.p2 2>/dev/null && [ -s $R/pulled/.p2 ] && mv $R/pulled/.p2 $R/pulled/s1probe.log
  timeout 30 $S $CL 'cat ~/yeto-output/rl-island-0.jsonl 2>/dev/null' > $R/pulled/.e2 2>/dev/null && [ -s $R/pulled/.e2 ] && mv $R/pulled/.e2 $R/pulled/rl-island-0.jsonl
  sleep 5
done
log "post-kill window done"

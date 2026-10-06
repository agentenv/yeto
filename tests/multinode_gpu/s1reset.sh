#!/bin/bash
# usage: s1reset.sh <cluster> <log>   between g12 and g3 on the kept 2-node cluster: stop leftovers on BOTH nodes, remove state, require empty GPUs (1 per node).
CL=$1; LOG=$2; export HOME=/home/michael; ok=1; : > $LOG
for n in $CL $CL-worker1; do
  timeout 300 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 $n 'bash -s' >> $LOG 2>&1 <<'REMOTE'
echo "== $(hostname) reset $(date -u +%FT%TZ)"
pkill -9 -f "yeto-rl/s1probe.sh" 2>/dev/null; pkill -f "python3 -m yeto.rl.learner" 2>/dev/null
MR="$HOME/miles-ray"; pkill -f "$MR/" 2>/dev/null; for i in $(seq 1 10); do pgrep -f "$MR/" >/dev/null || break; sleep 1; done; pkill -KILL -f "$MR/" 2>/dev/null
pkill -KILL -f "yeto.rl.learner" 2>/dev/null; pkill -KILL -f "sglang.launch_server|sglang.srt|sglang::" 2>/dev/null; sleep 3
rm -rf ~/yeto-rl/elastic-state ~/yeto-output ~/yeto-rl/s1probe.log ~/yeto-rl/s1probe.out
for i in $(seq 1 24); do
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .); maxmem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
  echo "check $i: compute_apps=$apps max_mem_used_mib=${maxmem:-NA}"; [ "$apps" = 0 ] && [ -n "$maxmem" ] && [ "$maxmem" -lt 1500 ] && { echo RESET_OK; exit 0; }; pkill -KILL -f "$MR/" 2>/dev/null; sleep 5
done
echo RESET_NOT_CLEAN; exit 1
REMOTE
  [ $? = 0 ] || ok=0
done
[ $ok = 1 ] && echo "reset ok" >> $LOG; [ $ok = 1 ]

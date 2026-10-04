#!/bin/bash
# usage: reset_island.sh <cluster name> <log file>   (chain mode, between items; gpu-plan 9.23)
# Brings a kept island back to "fresh": stops leftover helpers and the island's own Ray (path-scoped, never SkyPilot's runtime Ray), removes the elastic state dir /
# tape / sampler files, then REQUIRES an empty GPU compute list and ~0 memory.used on every GPU.  Exit 0 = fresh; 1 = not clean (the chain then releases the cluster).
CL=$1; LOG=$2; SSH=${SSH:-ssh}; export HOME=${HOME:-/home/michael}
HOME=/home/michael timeout ${RESET_TIMEOUT_S:-300} $SSH -o StrictHostKeyChecking=no -o ConnectTimeout=20 $CL 'bash -s' > $LOG 2>&1 <<'REMOTE'
echo "reset start $(date -u +%FT%TZ)"
for p in router_sampler.py inwatch.py dkill.py dctl.py fork_probe.py probe_remote.sh; do pkill -9 -f "yeto-rl/$p" 2>/dev/null; done
pkill -f "python3 -m yeto.rl.learner" 2>/dev/null
MR="$HOME/miles-ray"
pkill -f "$MR/" 2>/dev/null; for i in 1 2 3 4 5 6 7 8 9 10; do pgrep -f "$MR/" >/dev/null 2>&1 || break; sleep 1; done; pkill -KILL -f "$MR/" 2>/dev/null
pkill -KILL -f "yeto.rl.learner" 2>/dev/null
pkill -KILL -f "sglang.launch_server|sglang.srt|sglang::" 2>/dev/null
sleep 3
rm -rf ~/yeto-rl/elastic-state ~/yeto-output ~/yeto-rl/router_samples.jsonl ~/yeto-rl/gpu_samples.jsonl ~/yeto-rl/sampler.out ~/yeto-rl/inwatch.log ~/yeto-rl/inwatch.out ~/yeto-rl/dkill.log ~/yeto-rl/dctl.log ~/yeto-rl/*.py.out ~/yeto-rl/router_sampler.py ~/yeto-rl/inwatch.py ~/yeto-rl/dkill.py ~/yeto-rl/dctl.py ~/yeto-rl/fork_probe.py ~/yeto-rl/probe_remote.sh
ok=0
for i in $(seq 1 24); do
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
  maxmem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
  ngpu=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | grep -c .)
  echo "check $i: compute_apps=$apps max_mem_used_mib=${maxmem:-NA} gpus=$ngpu"
  if [ "$apps" = 0 ] && [ -n "$maxmem" ] && [ "$maxmem" -lt 1500 ] && [ "$ngpu" = 8 ]; then ok=1; break; fi
  pkill -KILL -f "$MR/" 2>/dev/null; sleep 5
done
echo "--- ps (miles/sglang/ray/yeto leftovers)"; ps -eo pid,ppid,etime,stat,args --no-headers | grep -E "miles-ray|sglang|yeto|Megatron|ray::" | grep -v grep | cut -c1-200
echo "--- state dirs"; ls -la ~/yeto-rl ~/yeto-output 2>&1 | head -20
[ $ok = 1 ] && echo "RESET_OK $(date -u +%FT%TZ)" || echo "RESET_NOT_CLEAN $(date -u +%FT%TZ)"
[ $ok = 1 ]
REMOTE
rc=$?; echo "reset rc=$rc" >> $LOG; exit $rc

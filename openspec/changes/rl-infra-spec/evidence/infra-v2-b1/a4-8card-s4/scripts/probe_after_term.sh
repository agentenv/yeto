#!/bin/bash
# usage: probe_after_term.sh <run dir> <prefix> <n_terminal> <extra_wait_s> [e1b]
# Chain-safe after-hook (never stops anything): waits until the island journal shows n terminal phases, waits extra_wait_s
# (>= 60 s so the GPU sampler covers the wd criterion (4) window), runs the fork status probe -> probe_after.txt, and for e1b
# also the stale-ACK probe on the first NEW engine URL (journal member_engines, the fork's own identity of the new engines)
# -> probe_stale.txt and the old-epoch probe -> probe_oldepoch.txt. Finally pulls gpu_samples.jsonl. Exit 0 when done.
source /home/michael/work/gpu-b1-runs/probelib.sh
R=$1; P=$2; N=$3; W=$4; MODE=${5:-}; CL=$(cat $R/cluster.txt)
until [ -f $R/rc.txt ]; do
  if [ -s $R/pulled/elastic-state.tgz.b64 ]; then
    rm -rf $R/.st; mkdir -p $R/.st; base64 -d $R/pulled/elastic-state.tgz.b64 2>/dev/null | tar xz -C $R/.st 2>/dev/null
    J=$R/.st/elastic-state/reconfig/journal.jsonl
    n=$(grep -c '"phase": *"\(SUCCEEDED\|REBUILT_OLD\|CANCELLED\|RECOVERY_REQUIRED\)"' $J 2>/dev/null)
    if [ "${n:-0}" -ge "$N" ]; then
      date -u +%FT%TZ > $R/terminal_seen.txt; sleep $W
      probe_ok $R $R/probe_after.txt status
      if [ "$MODE" = e1b ]; then
        STALE=$(python3 - "$J" <<'PY'
import json, sys
url = ""
for l in open(sys.argv[1]):
    try: r = json.loads(l)
    except Exception: continue
    if r.get("kind") == "member_engines" and r.get("engine_urls"): url = sorted(r["engine_urls"])[0]
print(url)
PY
)
        echo "STALE=$STALE" > $R/probe_stale_url.txt
        [ -n "$STALE" ] && probe_ok $R $R/probe_stale.txt stale $STALE
        probe_ok $R $R/probe_oldepoch.txt oldepoch
      fi
      HOME=/home/michael timeout 120 ssh -o StrictHostKeyChecking=no $CL 'cat ~/yeto-rl/gpu_samples.jsonl' > $R/gpu_samples.after.jsonl 2>/dev/null
      date -u +%FT%TZ > $R/probes_done.txt
      exit 0
    fi
  fi
  sleep 10
done
echo "launcher ended before $N terminal phases" > $R/probes_done.txt; exit 1

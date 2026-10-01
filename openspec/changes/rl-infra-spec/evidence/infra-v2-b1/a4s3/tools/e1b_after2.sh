#!/bin/bash
# usage: e1b_after2.sh <run dir> <prefix> : after the first terminal transaction, run fork probes (status, stale ACK on a stopped cell URL, old epoch), pull samplers, stop
source /home/michael/work/gpu-b1-runs/probelib.sh
R=$1; P=$2; CL=$(cat $R/cluster.txt); B=/home/michael/work/gpu-b1-runs
until [ -f $R/rc.txt ]; do
  if [ -s $R/pulled/elastic-state.tgz.b64 ]; then
    rm -rf $R/.st; mkdir -p $R/.st; base64 -d $R/pulled/elastic-state.tgz.b64 2>/dev/null | tar xz -C $R/.st 2>/dev/null
    n=$(grep -c '"phase":"\(SUCCEEDED\|REBUILT_OLD\|CANCELLED\|RECOVERY_REQUIRED\)"' $R/.st/elastic-state/reconfig/journal.jsonl 2>/dev/null)
    if [ "${n:-0}" -ge 1 ]; then
      date -u +%FT%TZ > $R/terminal_seen.txt; sleep 20
      probe_ok $R $R/probe_after.txt status
      STALE=$(HOME=/home/michael timeout 60 ssh -o StrictHostKeyChecking=no $CL "python3 - <<'PY'
import json,os
seen=set(); last=set()
for l in open(os.path.expanduser('~/yeto-rl/router_samples.jsonl')):
    d=json.loads(l).get('data') or {}
    ks=set((d.get('inflight') or {}).keys()); seen|=ks; last=ks if ks else last
print(sorted(seen-last)[0] if seen-last else '')
PY" 2>/dev/null | tail -1)
      echo "STALE=$STALE" > $R/probe_stale.txt
      [ -n "$STALE" ] && probe_ok $R $R/probe_stale_out.txt stale $STALE; cat $R/probe_stale_out.txt >> $R/probe_stale.txt 2>/dev/null
      probe_ok $R $R/probe_oldepoch.txt oldepoch
      HOME=/home/michael timeout 120 ssh $CL 'cat ~/yeto-rl/gpu_samples.jsonl' > $R/gpu_samples.jsonl 2>/dev/null
      $B/nstop.sh $P > $R/after_stop.txt 2>&1
      exit 0
    fi
  fi
  sleep 10
done

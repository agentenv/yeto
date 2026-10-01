#!/bin/bash
# usage: nstop_item.sh <item prefix>   (chain mode; env RUN_ROOT, CHAIN_DIR, CLUSTER_PREFIX as set by chain8.sh)
# Pulls the item's final evidence from the (kept, still UP) island and RETURNS WITHOUT RELEASING anything -- the chain owns the cluster.
# If the item never started (<run>/startup_failed) it is a new infrastructure blocker: release everything (cleanup_run.sh <chain prefix>) and write <chain>/ABORT.
P=$1; B=${BDIR:-/home/michael/work/gpu-b1-runs}; R=${RUN_ROOT:-$B}/$P; CL=$(cat $R/cluster.txt 2>/dev/null); CHAIN=${CHAIN_DIR:?}; CPX=${CLUSTER_PREFIX:?}
export HOME=/home/michael
mkdir -p $R/pulled/diag
if [ -n "$CL" ]; then
  S="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 $CL"
  timeout 120 $S 'cd ~/yeto-rl && tar czf - --exclude=cuts elastic-state | base64 -w0' > $R/pulled/.esf 2>/dev/null; [ -s $R/pulled/.esf ] && mv $R/pulled/.esf $R/pulled/elastic-state-final.b64
  for f in yeto-output/rl-island-0.jsonl:rl-island-0.final.jsonl yeto-rl/router_samples.jsonl:router_samples.jsonl yeto-rl/inwatch.log:inwatch.final.log yeto-rl/dkill.log:dkill.log yeto-rl/dctl.log:dctl.log yeto-rl/gpu_samples.jsonl:gpu_samples.jsonl yeto-rl/sampler.out:sampler.out yeto-rl/probe_after.txt:probe_after.txt yeto-rl/probe_stale.txt:probe_stale.txt yeto-rl/probe_oldepoch.txt:probe_oldepoch.txt yeto-rl/term_probe.log:term_probe.log; do
    timeout 120 $S "cat ~/${f%%:*} 2>/dev/null" > $R/pulled/.t 2>/dev/null; [ -s $R/pulled/.t ] && mv $R/pulled/.t $R/pulled/${f##*:}
  done
  timeout 60 $S 'date -u +%FT%TZ; nvidia-smi; ps -eo pid,ppid,etime,stat,args --no-headers | cut -c1-200' > $R/pulled/diag/final_snapshot.txt 2>/dev/null
  if [ -s $R/pulled/samplers.tgz.b64 ]; then rm -rf $R/pulled/.sm; mkdir -p $R/pulled/.sm; base64 -d $R/pulled/samplers.tgz.b64 2>/dev/null | tar xz -C $R/pulled/.sm 2>/dev/null
    for f in router_samples.jsonl gpu_samples.jsonl sampler.out dkill.log dctl.log; do [ -s $R/pulled/$f ] || [ ! -s $R/pulled/.sm/$f ] || cp $R/pulled/.sm/$f $R/pulled/$f; done; fi
fi
[ -s $R/pulled/elastic-state-final.b64 ] || [ ! -s $R/pulled/elastic-state.tgz.b64 ] || cp $R/pulled/elastic-state.tgz.b64 $R/pulled/elastic-state-final.b64
[ -s $R/pulled/rl-island-0.final.jsonl ] || { f=$(ls $R/runs/*/events/*.jsonl 2>/dev/null | head -1); [ -n "$f" ] && cp $f $R/pulled/rl-island-0.final.jsonl; }
if [ -f $R/startup_failed ]; then
  echo "{\"item\":\"$P\",\"reason\":\"startup_failed\",\"ts\":\"$(date -u +%FT%TZ)\"}" > $CHAIN/ABORT
  $B/cleanup_run.sh $CPX > $CHAIN/abort_cleanup.out 2>&1; rc=$?; echo $rc > $CHAIN/abort_cleanup_rc.txt; exit $rc
fi
exit 0

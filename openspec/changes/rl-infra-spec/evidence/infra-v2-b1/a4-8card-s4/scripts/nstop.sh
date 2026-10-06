#!/bin/bash
# usage: nstop.sh <prefix> : pull the final state from the island (best effort, bounded), then run cleanup_run.sh (the only release path).
# exit code = cleanup_run.sh's (0 clean twice, 2 residue/unverified).
P=$1; B=${BDIR:-/home/michael/work/gpu-b1-runs}; R=$B/$P; CL=$(cat $R/cluster.txt 2>/dev/null)
export HOME=/home/michael
mkdir -p $R/pulled
if [ -n "$CL" ]; then
  S="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 $CL"
  timeout 120 $S 'cd ~/yeto-rl && tar czf - --exclude=cuts elastic-state | base64 -w0' > $R/pulled/elastic-state-final.b64 2>/dev/null
  for f in yeto-output/rl-island-0.jsonl:rl-island-0.final.jsonl yeto-rl/router_samples.jsonl:router_samples.jsonl yeto-rl/inwatch.log:inwatch.final.log yeto-rl/dkill.log:dkill.log yeto-rl/dctl.log:dctl.log yeto-rl/gpu_samples.jsonl:gpu_samples.jsonl yeto-rl/sampler.out:sampler.out yeto-rl/elastic-state/side_effects.jsonl:side_effects.jsonl; do
    timeout 120 $S "cat ~/${f%%:*} 2>/dev/null" > $R/pulled/${f##*:} 2>/dev/null
  done
  # the island may already be gone (launcher tears it down after the job): fall back to the periodically pulled small-file bundle for anything the direct pull left empty
  if [ -s $R/pulled/samplers.tgz.b64 ]; then rm -rf $R/pulled/.sm; mkdir -p $R/pulled/.sm; base64 -d $R/pulled/samplers.tgz.b64 2>/dev/null | tar xz -C $R/pulled/.sm 2>/dev/null
    for f in router_samples.jsonl gpu_samples.jsonl sampler.out dkill.log dctl.log; do [ -s $R/pulled/$f ] || [ ! -s $R/pulled/.sm/$f ] || cp $R/pulled/.sm/$f $R/pulled/$f; done
    [ -s $R/pulled/inwatch.final.log ] || [ ! -s $R/pulled/.sm/inwatch.log ] || cp $R/pulled/.sm/inwatch.log $R/pulled/inwatch.final.log; fi
  [ -s $R/pulled/rl-island-0.final.jsonl ] || { f=$(ls $R/runs/*/events/*.jsonl 2>/dev/null | head -1); [ -n "$f" ] && cp $f $R/pulled/rl-island-0.final.jsonl; }
  # tape/journal fallbacks: final tape = the launcher's complete event file; elastic-state-final = the last periodic pull
  [ -s $R/pulled/elastic-state-final.b64 ] || [ ! -s $R/pulled/elastic-state.tgz.b64 ] || cp $R/pulled/elastic-state.tgz.b64 $R/pulled/elastic-state-final.b64
fi
# NOTE: no `| tee $R/cleanup.out` here: tee's command line contains /<prefix>/ and cleanup_run.sh phase 1 kills every process matching the prefix,
# which killed tee, then SIGPIPE'd cleanup_run (exit 141, only the phase-1 line written) -- found by the 8xH100 smoke (a8sm-20261001-1). Plain shell redirection has no process to match.
$B/cleanup_run.sh $P > $R/cleanup.out 2>&1; rc=$?
cat $R/cleanup.out
exit $rc

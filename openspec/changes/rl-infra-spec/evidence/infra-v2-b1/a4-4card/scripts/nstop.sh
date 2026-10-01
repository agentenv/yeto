#!/bin/bash
# usage: nstop.sh <prefix> : pull the final state from the island (best effort, bounded), then run cleanup_run.sh (the only release path).
# exit code = cleanup_run.sh's (0 clean twice, 2 residue/unverified).
P=$1; B=/home/michael/work/gpu-b1-runs; R=$B/$P; CL=$(cat $R/cluster.txt 2>/dev/null)
export HOME=/home/michael
mkdir -p $R/pulled
if [ -n "$CL" ]; then
  S="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 $CL"
  timeout 120 $S 'cd ~/yeto-rl && tar czf - --exclude=cuts elastic-state | base64 -w0' > $R/pulled/elastic-state-final.b64 2>/dev/null
  for f in yeto-output/rl-island-0.jsonl:rl-island-0.final.jsonl yeto-rl/router_samples.jsonl:router_samples.jsonl yeto-rl/inwatch.log:inwatch.final.log yeto-rl/dkill.log:dkill.log yeto-rl/dctl.log:dctl.log yeto-rl/gpu_samples.jsonl:gpu_samples.jsonl yeto-rl/sampler.out:sampler.out; do
    timeout 120 $S "cat ~/${f%%:*} 2>/dev/null" > $R/pulled/${f##*:} 2>/dev/null
  done
fi
$B/cleanup_run.sh $P 2>&1 | tee $R/cleanup.out
exit ${PIPESTATUS[0]}

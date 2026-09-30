#!/bin/bash
# usage: nstop.sh <prefix> : pull final state, stop launcher, sky down by cluster name, verify via nebius API
P=$1; B=/home/michael/work/gpu-b1-runs; R=$B/$P; CL=$(cat $R/cluster.txt); SKY=/home/michael/work/gpu-head/venv/bin/sky
export HOME=/home/michael
timeout 120 ssh -o StrictHostKeyChecking=no $CL 'cd ~/yeto-rl && tar czf - --exclude=cuts elastic-state | base64 -w0' > $R/pulled/elastic-state-final.b64 2>/dev/null
timeout 120 ssh $CL 'cat ~/yeto-output/rl-island-0.jsonl' > $R/pulled/rl-island-0.final.jsonl 2>/dev/null
timeout 120 ssh $CL 'cat ~/yeto-rl/router_samples.jsonl 2>/dev/null' > $R/pulled/router_samples.jsonl 2>/dev/null
timeout 60 ssh $CL 'cat ~/yeto-rl/inwatch.log 2>/dev/null' > $R/pulled/inwatch.final.log 2>/dev/null
timeout 60 ssh $CL 'cat ~/yeto-rl/dkill.log 2>/dev/null' > $R/pulled/dkill.log 2>/dev/null
timeout 60 ssh $CL 'cat ~/yeto-rl/dctl.log 2>/dev/null' > $R/pulled/dctl.log 2>/dev/null
timeout 120 ssh $CL 'cat ~/yeto-rl/gpu_samples.jsonl 2>/dev/null' > $R/pulled/gpu_samples.jsonl 2>/dev/null
for pid in $(pgrep -f "[c]luster-prefix $P "); do kill -TERM $pid; done
pkill -f "yeto.cli _worker $P" 2>/dev/null
timeout 600 $SKY down -y $CL > $R/nstop.out 2>&1
w=$(cat $R/watchdog.pid); pkill -P $w; kill $w $(cat $R/puller.pid) 2>/dev/null
for i in $(seq 1 30); do
  n=$(timeout 90 ~/.nebius/bin/nebius --profile michael compute instance list --parent-id project-e00eqrj3pr00622zrgdeyc --format json 2>/dev/null | python3 -c "import json,sys;d=json.load(sys.stdin);print(len([i for i in d.get('items',[]) if '$P' in i['metadata']['name']]))")
  [ "$n" = "0" ] && break; sleep 10
done
echo "nebius instances with prefix: $n" | tee -a $R/nstop.out

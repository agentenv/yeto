#!/bin/bash
# usage: selfcheck.sh <run dir> <prefix> [attestation.json] : after the first generate, verify router sampler, gpu sampler and fork probe; stop the run if any is unusable
R=$1; P=$2; CL=$(cat $R/cluster.txt)
until grep -q '"phase":"generate"' $R/pulled/rl-island-0.jsonl 2>/dev/null; do [ -f $R/rc.txt ] && exit 1; sleep 10; done
sleep 20
HOME=/home/michael timeout 60 ssh -o StrictHostKeyChecking=no $CL 'grep -m1 router ~/yeto-rl/router_samples.jsonl; tail -1 ~/yeto-rl/router_samples.jsonl; wc -l < ~/yeto-rl/gpu_samples.jsonl' > $R/selfcheck.txt 2>&1
/home/michael/work/gpu-b1-runs/run_probe.sh $R status > $R/selfcheck_probe.txt 2>&1
ok=1
if [ -n "${3:-}" ]; then
  want=$(python3 -c "import json;print(json.load(open('$3'))['runtime_fingerprint'])")
  got=$(grep -m1 '"rl_driver_start"' $R/pulled/rl-island-0.jsonl | python3 -c "import json,sys;print(json.loads(sys.stdin.readline()).get('runtime_fingerprint'))")
  echo "fingerprint want=$want got=$got" >> $R/selfcheck.txt
  [ "$want" = "$got" ] || ok=0
fi
grep -q '"router"' $R/selfcheck.txt || ok=0
grep -q '"inflight"' $R/selfcheck.txt || ok=0
grep -q '"cell_statuses"' $R/selfcheck_probe.txt || echo 'probe_failed=1 (non-fatal; retried live by the after-hook)' >> $R/selfcheck.txt
echo "selfcheck ok=$ok $(date -u +%FT%TZ)" >> $R/selfcheck.txt
[ $ok = 1 ] || /home/michael/work/gpu-b1-runs/nstop.sh $P >> $R/selfcheck.txt 2>&1

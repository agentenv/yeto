#!/bin/bash
# usage: selfcheck.sh <run dir> <prefix> [attestation.json]
# After the first generate: verify router sampler, gpu sampler, fork probe (attested) and fingerprint.
# Marker files in <run dir> (JSON {reason,ts}), so the verdict can tell "no valid test" from a product result:
#   startup_failed  : the run never reached its first generate (launch ended / startup deadline), or a self-check tool is unusable / wrong Ray / fingerprint mismatch
#   injection_not_reached, recovery_failed : written by judge_inject.py (see judge_after.sh), never by this script
# On startup_failed the run is stopped via nstop (-> cleanup_run.sh).
R=$1; P=$2; ATT=${3:-}; B=${BDIR:-/home/michael/work/gpu-b1-runs}; NSTOP=${NSTOP:-$B/nstop.sh}; SSH=${SSH:-ssh}; PROBE=${PROBE:-$B/run_probe.sh}
CL=$(cat $R/cluster.txt); DL=${STARTUP_DEADLINE_S:-1800}; t0=$(date +%s)
mark() { echo "{\"reason\":\"$2\",\"ts\":\"$(date -u +%FT%TZ)\"}" > $R/$1; echo "$1: $2" >> $R/selfcheck.txt; }
until grep -q '"phase":"generate"' $R/pulled/rl-island-0.jsonl 2>/dev/null; do
  if [ -f $R/rc.txt ]; then mark startup_failed "launch_ended_before_first_generate ($(cat $R/rc.txt))"; $NSTOP $P >> $R/selfcheck.txt 2>&1; exit 1; fi
  if [ $(( $(date +%s) - t0 )) -ge $DL ]; then mark startup_failed "no_generate_within_${DL}s"; $NSTOP $P >> $R/selfcheck.txt 2>&1; exit 1; fi
  sleep ${SC_POLL_S:-10}
done
sleep ${SC_SETTLE_S:-20}
HOME=/home/michael timeout 60 $SSH -o StrictHostKeyChecking=no $CL 'grep -m1 router ~/yeto-rl/router_samples.jsonl; tail -1 ~/yeto-rl/router_samples.jsonl; wc -l < ~/yeto-rl/gpu_samples.jsonl' >> $R/selfcheck.txt 2>&1
$PROBE $R status > $R/selfcheck_probe.txt 2>&1
bad=""
if [ -n "${EXPECT_GPU_NAME:-}" ]; then   # GPU assertion (e.g. L40S x4): gpu.txt = nvidia-smi index,uuid,name,driver
  n=$(grep -c -i -- "$EXPECT_GPU_NAME" $R/pulled/gpu.txt 2>/dev/null); tot=$(grep -c . $R/pulled/gpu.txt 2>/dev/null)
  echo "gpu assert want=${EXPECT_GPU_N}x$EXPECT_GPU_NAME got=${n}/${tot}" >> $R/selfcheck.txt
  [ "${n:-0}" = "${EXPECT_GPU_N}" ] && [ "${tot:-0}" = "${EXPECT_GPU_N}" ] || bad="$bad gpu_assert"
fi
if [ -n "$ATT" ]; then
  want=$(python3 -c "import json;print(json.load(open('$ATT'))['runtime_fingerprint'])")
  got=$(grep -m1 '"rl_driver_start"' $R/pulled/rl-island-0.jsonl | python3 -c "import json,sys;print(json.loads(sys.stdin.readline()).get('runtime_fingerprint'))")
  echo "fingerprint want=$want got=$got" >> $R/selfcheck.txt
  [ "$want" = "$got" ] || bad="$bad fingerprint_mismatch"
fi
grep -q '"router"' $R/selfcheck.txt || bad="$bad router_sampler_unusable"
grep -q '"inflight"' $R/selfcheck.txt || bad="$bad router_no_inflight"
if grep -q '"probe_attested": false' $R/selfcheck_probe.txt; then bad="$bad probe_not_attested(wrong_ray)"
elif ! grep -q '"probe_attested": true' $R/selfcheck_probe.txt; then echo 'probe_failed=1 (non-fatal; retried live by the after-hook)' >> $R/selfcheck.txt; fi
if [ -n "$bad" ]; then mark startup_failed "selfcheck_tool:$bad"; $NSTOP $P >> $R/selfcheck.txt 2>&1; exit 1; fi
echo "selfcheck ok=1 $(date -u +%FT%TZ)" >> $R/selfcheck.txt

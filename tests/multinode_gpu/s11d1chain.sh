#!/bin/bash
# D1-2 + D1-3 (tasks 2.4 / 5.1) on ONE 2x2 L40S --keep cluster (nebius eu-north1, $9.14/h).
# order (expected-pass first, risky T1R3 at the tail): T2R1S1 s17 (cold) -> T2R2S0 s17 -> T2R1S1 s29 -> T2R2S0 s29 -> d1e1 (3 up/down pairs, KEEP_M3=1)
#   -> T1R3S0 s17 -> T1R3S0 s29 -> sky down <this cluster> + cleanup_run.sh (clean twice). s1reset.sh between runs; reset not clean -> stop.
# budget guard: no new run starts after BUDGET_S (default 3.5 h = $32) since chain start.
# usage: s11d1chain.sh <prefix>   (cluster = <prefix>-l0-eu-north1)
set -u
P=$1; D=$(cd "$(dirname "$0")" && pwd); B=${RUN_ROOT:-/home/michael/work/s1-runs}; CL=$P-l0-eu-north1; export HOME=/home/michael
SKY=/home/michael/work/gpu-head/venv/bin/sky; T0=$(date +%s); BUDGET_S=${BUDGET_S:-12600}
LOG=$B/$P.chain.log; log() { echo "[chain $(date -u +%FT%TZ)] $*" | tee -a $LOG; }
finish() {
  for f in $B/$P-*/watchdog.pid; do [ -f "$f" ] && kill $(cat $f) 2>/dev/null; done
  log "sky down $CL"; timeout 900 $SKY down -y $CL >> $LOG 2>&1
  log "cleanup_run.sh $P"; RUNS_BASE=$B /home/michael/work/gpu-b1-runs/cleanup_run.sh $P > $B/$P.cleanup1.out 2>&1; log "cleanup1 rc=$? $(grep -o 'RESULT:.*' $B/$P.cleanup1.out | tail -1)"
  RUNS_BASE=$B /home/michael/work/gpu-b1-runs/cleanup_run.sh $P > $B/$P.cleanup2.out 2>&1; log "cleanup2 rc=$? $(grep -o 'RESULT:.*' $B/$P.cleanup2.out | tail -1)"
  timeout 120 $SKY status 2>/dev/null | grep -q "$CL" && log "WARNING: $CL still listed" || log "sky status: $CL gone"
  log "elapsed $(( $(date +%s) - T0 )) s"
}
trap finish EXIT
df_free=$(df -BG --output=avail / | tail -1 | tr -dc 0-9); [ "$df_free" -ge 50 ] || { log "abort: root disk ${df_free}G < 50G"; exit 9; }
log "start $P (code $(git -C $D/../.. rev-parse --short HEAD))"
N=0
step() {  # step <name> <case> [env...]
  local name=$1 c=$2; shift 2
  [ $(( $(date +%s) - T0 )) -lt $BUDGET_S ] || { log "budget guard: skip $name and the rest"; exit 4; }
  if [ $N -gt 0 ]; then
    timeout 120 $SKY status $CL 2>/dev/null | grep -q ' UP ' || { log "cluster $CL not UP before $name -> stop"; exit 2; }
    $D/s1reset.sh $CL $B/$P-$name.reset.txt || { log "reset NOT CLEAN before $name -> stop"; exit 3; }
  fi
  N=$((N+1))
  env "$@" CLUSTER_PREFIX=$P RUN_ROOT=$B $D/s1run.sh $c $P-$name ${HARD:-3000} ${WD:-3600} > $B/$P-$name.out 2>&1
  log "$name $(cat $B/$P-$name/rc.txt 2>/dev/null) $(cat $B/$P-$name/start_utc.txt $B/$P-$name/end_utc.txt 2>/dev/null | tr '\n' ' ')"
  kill $(cat $B/$P-$name/watchdog.pid 2>/dev/null) 2>/dev/null   # its watchdog would sky-down the shared cluster later
  [ -f $B/$P-$name/provision_failed.txt ] && { log "provision failed at $name -> stop"; exit 5; }
}
sw() { step $1 d1sweep D1_CFG=$2 SEED=$3; python3 $D/s1cost.py sweep $B/$P-$1 > $B/$P-$1/sweep.json 2>&1; log "$1 valid=$(python3 -c "import json;d=json.load(open('$B/$P-$1/sweep.json'))[0];print(d['valid'],d['median_round_s'])" 2>/dev/null)"; }
sw a17 T2R1S1 17
grep -q '"valid": true' $B/$P-a17/sweep.json || log "WARNING: first run invalid (continuing only if cluster UP)"
sw b17 T2R2S0 17
sw a29 T2R1S1 29
sw b29 T2R2S0 29
step e1 d1e1 KEEP_M3=1
python3 $D/s1cost.py $B/$P-e1 --json $B/$P-e1/cost.json > $B/$P-e1/cost.tsv 2>&1; log "e1 cost rows $(($(wc -l < $B/$P-e1/cost.tsv)-1)); inwatch $(grep -c submitted $B/$P-e1/pulled/inwatch.log 2>/dev/null)"
sw c17 T1R3S0 17
sw c29 T1R3S0 29

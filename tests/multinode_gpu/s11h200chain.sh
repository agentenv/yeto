#!/bin/bash
# S11 combined chain (user 10-05: one launch, maximise value; stage budget $300, THIS chain capped at CAP_USD=150).
#   seg 1 D1-2 sweep (2.4): T2R1S1 s17 -> T2R2S0 s17 -> T2R1S1 s29 -> T2R2S0 s29
#   seg 2 D1-3 e1 (5.1): d1e1, 3 up/down pairs, 14 rounds
#   seg 2b lp: teacher-forced logprob of the base model on the island's GPUs / TP shapes (s11lp.sh)
#   seg 2c T1R3S0 s17, s29 (sweep config with the most layout risk -> after the must-have data)
#   seg 3 fnA (Flash-Next stage-A slice; FAILED-risk highest -> chain tail): FNA_CMD (env, from fn-align) or skipped.
# Path: PREFER_L40S=1 (default) tries the first run on nebius 2x2 L40S eu-north1 ($9.14/h). Provision failure (provision_failed.txt)
#   -> immediately falls back to nebius 1x8 H200 eu-north1 (gpu-h200-sxm_8gpu-128vcpu-1600gb, $36/h, island allocated GPUs 0-3) for
#   segs 1-2c. fnA always runs on H200 (separate cluster when segs 1-2c ran on L40S; the L40S cluster is downed first).
# Clusters: <P>-l-l0-eu-north1 (L40S), <P>-h-l0-eu-north1 (H200). Exit: watchdogs killed, both clusters downed, cleanup_run.sh <P> x2.
# Budget guard: spent = sum(cluster wall x rate); a step starts only if spent + its estimate <= CAP_USD.
# usage: s11h200chain.sh <prefix>     env: PREFER_L40S (1), CAP_USD (150), FNA_CMD, FNA_EST_USD (70), HARD (3000)
set -u
P=$1; D=$(cd "$(dirname "$0")" && pwd); REPO=$(cd $D/../.. && pwd); B=${RUN_ROOT:-/home/michael/work/s1-runs}; export HOME=/home/michael
SKY=/home/michael/work/gpu-head/venv/bin/sky; CAP=${CAP_USD:-150}; LOG=$B/$P.chain.log; mkdir -p $B
CLL=$P-l-l0-eu-north1; CLH=$P-h-l0-eu-north1; RATE_L=9.14; RATE_H=36.0
log() { echo "[chain $(date -u +%FT%TZ)] $*" | tee -a $LOG; }
UP_L=0; UP_H=0; ACC=0   # cluster-up epoch (0 = down) and accumulated $ of downed clusters
spent() { python3 -c "import time;n=time.time();print(round($ACC+(($RATE_L*(n-$UP_L)) if $UP_L else 0)/3600+(($RATE_H*(n-$UP_H)) if $UP_H else 0)/3600,2))"; }
down() {  # down <cluster> <L|H>
  timeout 900 $SKY down -y $1 >> $LOG 2>&1; log "sky down $1 rc=$?"
  if [ $2 = L ] && [ $UP_L != 0 ]; then ACC=$(python3 -c "import time;print($ACC+$RATE_L*(time.time()-$UP_L)/3600)"); UP_L=0; fi
  if [ $2 = H ] && [ $UP_H != 0 ]; then ACC=$(python3 -c "import time;print($ACC+$RATE_H*(time.time()-$UP_H)/3600)"); UP_H=0; fi
}
finish() {
  for f in $B/$P-*/watchdog.pid; do [ -f "$f" ] && kill $(cat $f) 2>/dev/null; done
  down $CLL L; down $CLH H
  RUNS_BASE=$B /home/michael/work/gpu-b1-runs/cleanup_run.sh $P > $B/$P.cleanup1.out 2>&1; log "cleanup1 rc=$? $(grep -o 'RESULT:.*' $B/$P.cleanup1.out | tail -1)"
  RUNS_BASE=$B /home/michael/work/gpu-b1-runs/cleanup_run.sh $P > $B/$P.cleanup2.out 2>&1; log "cleanup2 rc=$? $(grep -o 'RESULT:.*' $B/$P.cleanup2.out | tail -1)"
  timeout 120 $SKY status 2>/dev/null | grep -E "$CLL|$CLH" && log "WARNING: cluster still listed" || log "sky status: $CLL / $CLH gone"
  log "estimated spend \$$(spent) (cap \$$CAP)"
}
trap finish EXIT
df_free=$(df -BG --output=avail / | tail -1 | tr -dc 0-9); [ "$df_free" -ge 50 ] || { log "abort: root disk ${df_free}G < 50G"; exit 9; }
T=$(ps -u michael -L -o pid= | wc -l); [ "$T" -lt 2850 ] || { log "abort: $T threads"; exit 9; }
log "start $P code $(git -C $REPO rev-parse --short HEAD) prefer_l40s=${PREFER_L40S:-1} cap=\$$CAP"
H=0; CL=$CLL; CP=$P-l; [ "${PREFER_L40S:-1}" = 1 ] || { H=1; CL=$CLH; CP=$P-h; }
N=0
post() {  # per-run data capture (all local; cluster untouched)
  local R=$1
  python3 $D/s1cost.py sweep $R > $R/sweep.json 2>&1
  grep -E "Pull|pull|Download|download|Loading|loaded|ready|Ready|Provision|provision|Launching|Launched|sglang|engine.*start" $R/launch.ts.log > $R/coldstart.txt 2>/dev/null
  grep -hiE "tok/s|tokens/s|throughput|perf/" $R/pulled/run.log > $R/perf.txt 2>/dev/null
  (cd $REPO && PYTHONPATH=. timeout 300 /tmp/yeto-venv/bin/python -m yeto.cli dashboard export --tapes $R/pulled -o $R/dashboard.html > $R/dashboard.out 2>&1)
}
step() {  # step <name> <case> <est_usd> [env...]
  local name=$1 c=$2 est=$3; shift 3
  local s; s=$(spent); python3 -c "import sys;sys.exit(0 if $s+$est<=$CAP else 1)" || { log "budget guard: spent \$$s + est \$$est > \$$CAP -> skip $name and the rest"; exit 4; }
  if [ $N -gt 0 ]; then
    timeout 120 $SKY status $CL 2>/dev/null | grep -q ' UP ' || { log "cluster $CL not UP before $name -> stop"; exit 2; }
    $D/s1reset.sh $CL $B/$P-$name.reset.txt || { log "reset NOT CLEAN before $name -> stop"; exit 3; }
  fi
  local t0; t0=$(date +%s)
  env "$@" D1_H200=$H CLUSTER_PREFIX=$CP RUN_ROOT=$B $D/s1run.sh $c $P-$name ${HARD:-3000} ${WD:-3600} > $B/$P-$name.out 2>&1
  kill $(cat $B/$P-$name/watchdog.pid 2>/dev/null) 2>/dev/null   # would sky-down the shared cluster later
  if [ -f $B/$P-$name/provision_failed.txt ]; then log "$name: provision failed on $CL"; return 7; fi
  if [ $N = 0 ]; then
    if [ $H = 1 ]; then UP_H=$t0; else UP_L=$t0; fi
    timeout 120 $SKY status -v $CL > $B/$P-sky-status-$CL.txt 2>&1   # SKU/instance type evidence
  fi
  N=$((N+1)); post $B/$P-$name
  log "$name $(cat $B/$P-$name/rc.txt 2>/dev/null) $(python3 -c "import json;d=json.load(open('$B/$P-$name/sweep.json'))[0];print('valid',d['valid'],'median_round_s',d['median_round_s'])" 2>/dev/null) spent~\$$(spent)"
}
EST=$( [ $H = 1 ] && echo 8 || echo 3 )   # per warm run (8 min); first run adds cold start
sw() { step $1 d1sweep $2 D1_CFG=$3 SEED=$4; }
sw a17 $(( EST * 3 )) T2R1S1 17; rc=$?
if [ $rc = 7 ] && [ $H = 0 ]; then
  log "L40S provision failed -> fallback to H200 ($CLH)"; H=1; CL=$CLH; CP=$P-h; EST=8
  sw a17h 24 T2R1S1 17; rc=$?
fi
[ $rc = 7 ] && { log "H200 provision failed too -> stop (capacity; report to main agent)"; exit 8; }
sw b17 $EST T2R2S0 17; sw a29 $EST T2R1S1 29; sw b29 $EST T2R2S0 29
step e1 d1e1 $(( EST * 2 )) KEEP_M3=1
python3 $D/s1cost.py $B/$P-e1 --json $B/$P-e1/cost.json > $B/$P-e1/cost.tsv 2>&1; log "e1 cost rows $(($(wc -l < $B/$P-e1/cost.tsv)-1)) inwatch=$(grep -c submitted $B/$P-e1/pulled/inwatch.log 2>/dev/null)"
# seg 2b: lp on the same cluster, island stopped
if python3 -c "import sys;sys.exit(0 if $(spent)+$EST<=$CAP else 1)" && $D/s1reset.sh $CL $B/$P-lp.reset.txt; then
  timeout 2400 $D/s11lp.sh $CL $B/$P-lp $( [ $H = 1 ] && echo 4 || echo 2 ) > $B/$P-lp.out 2>&1; log "lp $(tail -1 $B/$P-lp.out | cut -c1-300)"
else log "lp skipped (budget or reset)"; fi
sw c17 $EST T1R3S0 17; sw c29 $EST T1R3S0 29
# seg 3: fnA (Flash-Next stage A) on H200 -- chain tail
if [ -z "${FNA_CMD:-}" ]; then log "fnA: FNA_CMD not set (fn-align not merged) -> skipped"; exit 0; fi
if [ $H = 0 ]; then down $CLL L; H=1; CL=$CLH; CP=$P-h; N=0; else $D/s1reset.sh $CL $B/$P-fna.reset.txt || { log "reset NOT CLEAN before fnA -> stop"; exit 3; }; fi
s=$(spent); python3 -c "import sys;sys.exit(0 if $s+${FNA_EST_USD:-70}<=$CAP else 1)" || { log "budget guard: fnA est \$${FNA_EST_USD:-70} + \$$s > \$$CAP -> skip"; exit 4; }
t0=$(date +%s)
# FNA_CMD contract: runs one fnA launch on cluster prefix $CLUSTER_PREFIX (H200), writes evidence to $RUN_DIR (pulled/ like s1run), exits.
env CLUSTER_PREFIX=$CP RUN_ROOT=$B RUN_DIR=$B/$P-fna CLUSTER=$CL bash -c "$FNA_CMD" > $B/$P-fna.out 2>&1; log "fnA rc=$?"
[ $UP_H = 0 ] && UP_H=$t0
[ -d $B/$P-fna/pulled ] && post $B/$P-fna

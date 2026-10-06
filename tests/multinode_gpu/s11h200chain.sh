#!/bin/bash
# S11 combined chain (user 10-05: one launch, maximise value; stage budget $300, THIS chain capped at CAP_USD=150).
#   seg 1 D1-2 sweep (2.4): T2R1S1 s17 -> T2R2S0 s17 -> T2R1S1 s29 -> T2R2S0 s29
#   seg 2 D1-3 e1 (5.1): d1e1, 3 up/down pairs, 14 rounds
#   seg 2b lp: teacher-forced logprob of the base model on the island's GPUs / TP shapes (s11lp.sh)
#   seg 2c T1R3S0 s17, s29 (sweep config with the most layout risk -> after the must-have data)
#   seg 3 fnboot -> fnprep -> fna -> fnconv (Flash-Next stage A, 4layer, own FS-attached H200 cluster <P>-f; highest FAILED risk -> chain tail)
# Path: PREFER_L40S=1 (default) tries the first run on nebius 2x2 L40S eu-north1 ($9.14/h). Provision failure (provision_failed.txt)
#   -> immediately falls back to nebius 1x8 D1_GPU (h200 default; h100 = gpu-h100-sxm_8gpu-128vcpu-1600gb, $30.8/h, 80GB; if h100 also
#   fails and FALLBACK_H200=1 (default) one more try on H200 after sky-downing the failed cluster; all failed -> exit 8) -> was: 1x8 H200 eu-north1 (gpu-h200-sxm_8gpu-128vcpu-1600gb, $36/h, island allocated GPUs 0-3) for
#   segs 1-2c. The fn segments always run on a separate FS-attached H200 cluster (sweep cluster downed first).
# Clusters: <P>-l-l0-eu-north1 (L40S), <P>-h-l0-eu-north1 (H200). Exit: watchdogs killed, both clusters downed, cleanup_run.sh <P> x2.
# Budget guard: spent = sum(cluster wall x rate); a step starts only if spent + its estimate <= CAP_USD.
# START_AT=e1: skip seg 1 (sweep already valid in an earlier chain); e1 is the first run and provisions the cluster
#   (set PREFER_L40S=0 D1_GPU=h100 to stay on the H100 path). Then lp, c17/c29, fn as usual.
# A failed run whose launcher tore the cluster down (job FAILED -> recovery teardown, even with --keep) no longer stops the chain:
#   the next step re-provisions (cost of the dead cluster folded into ACC); lp runs after the re-provisioning c17 in that case.
# usage: s11h200chain.sh <prefix>     env: START_AT (sweep|e1), PREFER_L40S (1), CAP_USD (150; stage total $300), FN_ENABLE (1), FNA_EST_USD (50), D1_GPU (h200|h100), FALLBACK_H200 (1), FN_GPU (h200|h100), FN_CONV (1; 0 when FN_GPU=h100), FNCONV_EST_USD (50), HARD (3000), HARD_FNBOOT, HARD_FNA
set -u
P=$1; D=$(cd "$(dirname "$0")" && pwd); REPO=$(cd $D/../.. && pwd); B=${RUN_ROOT:-/home/michael/work/s1-runs}; export HOME=/home/michael
SKY=/home/michael/work/gpu-head/venv/bin/sky; CAP=${CAP_USD:-150}; LOG=$B/$P.chain.log; mkdir -p $B
CLL=$P-l-l0-eu-north1; CLH=$P-h-l0-eu-north1; RATE_L=9.14
G=${D1_GPU:-h200}; FG=${FN_GPU:-h200}
for g in $G $FG; do case $g in h100|h200) ;; *) echo "abort: D1_GPU/FN_GPU must be h100|h200 (got $g)"; exit 64;; esac; done
rate() { [ $1 = h100 ] && echo 30.8 || echo 36.0; }               # nebius eu-north1 8-GPU on-demand $/h [sky catalog 2026-10-05]
sku() { echo gpu-$1-sxm_8gpu-128vcpu-1600gb; }
hest() { [ $1 = h100 ] && echo 7 || echo 8; }                       # $ per warm run (H200 8; scaled by 30.8/36 for h100)
RATE_H=$(rate $G)
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
  down $CLL L; down $CLH H; down $P-f-l0-eu-north1 H
  RUNS_BASE=$B /home/michael/work/gpu-b1-runs/cleanup_run.sh $P > $B/$P.cleanup1.out 2>&1; log "cleanup1 rc=$? $(grep -o 'RESULT:.*' $B/$P.cleanup1.out | tail -1)"
  RUNS_BASE=$B /home/michael/work/gpu-b1-runs/cleanup_run.sh $P > $B/$P.cleanup2.out 2>&1; log "cleanup2 rc=$? $(grep -o 'RESULT:.*' $B/$P.cleanup2.out | tail -1)"
  timeout 120 $SKY status 2>/dev/null | grep -E "$CLL|$CLH|$P-f-l0" && log "WARNING: cluster still listed" || log "sky status: $CLL / $CLH gone"
  log "estimated spend \$$(spent) (cap \$$CAP)"
}
trap finish EXIT
df_free=$(df -BG --output=avail / | tail -1 | tr -dc 0-9); [ "$df_free" -ge 50 ] || { log "abort: root disk ${df_free}G < 50G"; exit 9; }
T=$(ps -u michael -L -o pid= | wc -l); [ "$T" -lt 2850 ] || { log "abort: $T threads"; exit 9; }
log "start $P code $(git -C $REPO rev-parse --short HEAD) prefer_l40s=${PREFER_L40S:-1} cap=\$$CAP d1_gpu=$G ($(sku $G) \$$RATE_H/h) fn_gpu=$FG ($(sku $FG) \$$(rate $FG)/h)"
H=0; CL=$CLL; CP=$P-l; export NODES=2; [ "${PREFER_L40S:-1}" = 1 ] || { H=1; CL=$CLH; CP=$P-h; export NODES=1; }
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
    if ! timeout 120 $SKY status $CL 2>/dev/null | grep -q ' UP '; then
      log "cluster $CL not UP before $name (previous run's launcher tore it down) -> fold its cost, re-provision with $name"
      if [ $H = 1 ]; then down $CL H; else down $CL L; fi; N=0
    fi
  fi
  if [ $N -gt 0 ]; then
    $D/s1reset.sh $CL $B/$P-$name.reset.txt || { log "reset NOT CLEAN before $name -> stop"; exit 3; }
  fi
  local t0; t0=$(date +%s)
  env "$@" D1_H200=$H D1_GPU=$G CLUSTER_PREFIX=$CP RUN_ROOT=$B $D/s1run.sh $c $P-$name ${HARD:-3000} ${WD:-3600} > $B/$P-$name.out 2>&1
  kill $(cat $B/$P-$name/watchdog.pid 2>/dev/null) 2>/dev/null   # would sky-down the shared cluster later
  if [ -f $B/$P-$name/provision_failed.txt ]; then log "$name: provision failed on $CL"; return 7; fi
  if [ $N = 0 ]; then
    if [ $H = 1 ]; then UP_H=$t0; else UP_L=$t0; fi
    timeout 120 $SKY status -v $CL > $B/$P-sky-status-$CL.txt 2>&1   # SKU/instance type evidence
  fi
  N=$((N+1)); post $B/$P-$name
  log "$name $(cat $B/$P-$name/rc.txt 2>/dev/null) $(python3 -c "import json;d=json.load(open('$B/$P-$name/sweep.json'))[0];print('valid',d['valid'],'median_round_s',d['median_round_s'])" 2>/dev/null) spent~\$$(spent)"
}
EST=$( [ $H = 1 ] && hest $G || echo 3 )   # per warm run (8 min); first run adds cold start
sw() { step $1 d1sweep $2 D1_CFG=$3 SEED=$4; }
SA=${START_AT:-sweep}; case $SA in sweep|e1) ;; *) log "abort: START_AT must be sweep|e1 (got $SA)"; exit 64;; esac
if [ $SA = sweep ]; then
sw a17 $(( EST * 3 )) T2R1S1 17; rc=$?
if [ $rc = 7 ] && [ $H = 0 ]; then
  log "L40S provision failed -> fallback to $G $(sku $G) \$$RATE_H/h ($CLH)"; H=1; CL=$CLH; CP=$P-h; EST=$(hest $G); export NODES=1
  sw a17h $(( EST * 3 )) T2R1S1 17; rc=$?
fi
if [ $rc = 7 ] && [ $G = h100 ] && [ "${FALLBACK_H200:-1}" = 1 ]; then
  log "h100 provision failed -> sky down $CLH, fallback to h200 $(sku h200) \$36.0/h"; down $CLH H
  G=h200; RATE_H=$(rate h200); H=1; CL=$CLH; CP=$P-h; EST=$(hest h200); export NODES=1
  sw a17h2 $(( EST * 3 )) T2R1S1 17; rc=$?
fi
[ $rc = 7 ] && { log "$G provision failed too -> stop (capacity; report to main agent)"; exit 8; }
log "sweep SKU: $( [ $H = 1 ] && echo "$(sku $G) \$$RATE_H/h" || echo "L40S \$$RATE_L/h")"
sw b17 $EST T2R2S0 17; sw a29 $EST T2R1S1 29; sw b29 $EST T2R2S0 29
step e1 d1e1 $(( EST * 2 )) KEEP_M3=1
else
log "START_AT=e1: sweep skipped; e1 provisions $CL"
step e1 d1e1 $(( EST * 4 )) KEEP_M3=1; rc=$?
[ $rc = 7 ] && { log "e1 provision failed on $CL -> stop (capacity)"; exit 8; }
fi
python3 $D/s1cost.py $B/$P-e1 --json $B/$P-e1/cost.json > $B/$P-e1/cost.tsv 2>&1; log "e1 cost rows $(($(wc -l < $B/$P-e1/cost.tsv)-1)) inwatch=$(grep -c submitted $B/$P-e1/pulled/inwatch.log 2>/dev/null)"
# seg 2b: lp on the same cluster, island stopped (if e1's launcher tore the cluster down, lp runs after c17 re-provisions it)
lp() {
if python3 -c "import sys;sys.exit(0 if $(spent)+$EST<=$CAP else 1)" && $D/s1reset.sh $CL $B/$P-lp.reset.txt; then
  timeout 2400 $D/s11lp.sh $CL $B/$P-lp $( [ $H = 1 ] && echo 4 || echo 2 ) > $B/$P-lp.out 2>&1; log "lp $(tail -1 $B/$P-lp.out | cut -c1-300)"
else log "lp skipped (budget or reset)"; fi
}
if timeout 120 $SKY status $CL 2>/dev/null | grep -q ' UP '; then lp; sw c17 $EST T1R3S0 17
else log "cluster $CL not UP after e1 -> c17 re-provisions, lp after it"; sw c17 $(( EST * 3 )) T1R3S0 17; lp; fi
sw c29 $EST T1R3S0 29
# seg 3 (chain tail): Flash-Next stage A on its OWN H200 cluster <P>-f (the model store FS is attached only at provision time, so the
# sweep cluster -- launched without --model-store -- cannot be reused). The sweep cluster is downed first.
#   fnboot: fnrun fn8s launch (--keep) to provision the FS-attached node; with no torch_dist yet the learner refuses the ref-load
#           (fail-fast, expected) -- if torch_dist already exists it simply IS the fnA run.
#   fnprep: s11fnprep.sh (populate 4layer snapshot + convert torch_dist; df gates; never deletes) -> fnprep.json
#   fna   : fnrun fn8s again (warm), STEPS=6; judge_qwen3_8_next_lora_log.py on run.log
#   fnconv: s11fnconv.sh (B0-2 full -> torch_dist, TP2 PP4 nproc 8; df/marker/no-retry gates) -> fnconv.json
[ "${FN_ENABLE:-1}" = 1 ] || { log "fn segments disabled (FN_ENABLE=0)"; exit 0; }
if [ $H = 1 ]; then down $CLH H; else down $CLL L; fi
CLF=$P-f-l0-eu-north1; CL=$CLF; CP=$P-f; H=1; N=0; export NODES=1; RATE_H=$(rate $FG); export FN_GPU=$FG   # sweep cluster already folded into ACC
log "fn SKU: $(sku $FG) \$$RATE_H/h"
s=$(spent); python3 -c "import sys;sys.exit(0 if $s+${FNA_EST_USD:-50}<=$CAP else 1)" || { log "budget guard: fn est \$${FNA_EST_USD:-50} + \$$s > \$$CAP -> fn not executed"; exit 4; }
(cd $REPO && /tmp/yeto-venv/bin/python $D/fp_fn.py $REPO fn8s --seed 17 --total-steps 6 > $B/$P-fp_fn8s.json 2>$B/$P-fp_fn8s.err); log "fp_fn fn8s steps6: $(tail -1 $B/$P-fp_fn8s.json | python3 -c 'import json,sys;print(json.load(sys.stdin).get("fp"))' 2>/dev/null)"
fnrun() {  # fnrun <name> <hard>
  local t0; t0=$(date +%s)
  env STEPS=6 CLUSTER_PREFIX=$CP RUN_ROOT=$B $D/s1run.sh fn8s $P-$1 $2 $(( $2 + 600 )) > $B/$P-$1.out 2>&1
  kill $(cat $B/$P-$1/watchdog.pid 2>/dev/null) 2>/dev/null
  [ $UP_H = 0 ] && [ ! -f $B/$P-$1/provision_failed.txt ] && UP_H=$t0
  post $B/$P-$1
  grep -iE "lora|rank|expected_lora_keys|trainable" $B/$P-$1/pulled/run.log > $B/$P-$1/lora.txt 2>/dev/null
  grep -iE "ref.load|torch_dist|load.*checkpoint|loaded" $B/$P-$1/pulled/run.log > $B/$P-$1/torchdist-load.txt 2>/dev/null
  (cd $REPO && timeout 300 python3 scripts/judge_qwen3_8_next_lora_log.py $B/$P-$1/pulled/run.log > $B/$P-$1/judge-fn.out 2>&1; echo "judge rc=$?" >> $B/$P-$1/judge-fn.out)
  timeout 120 ssh -o StrictHostKeyChecking=no $CL 'cat ~/yeto-rl/terminal.json 2>/dev/null; echo; df -h /mnt/yeto-models | tail -1; nvidia-smi --query-gpu=index,memory.used --format=csv' > $B/$P-$1/terminal.txt 2>&1
  log "$1 $(cat $B/$P-$1/rc.txt 2>/dev/null) rounds=$(grep -c rl_round_trained $B/$P-$1/pulled/rl-island-0.jsonl 2>/dev/null) spent~\$$(spent)"
}
fnrun fnboot ${HARD_FNBOOT:-3000}
[ -f $B/$P-fnboot/provision_failed.txt ] && { log "fn $FG provision failed -> fn not executed (capacity)"; exit 8; }
if [ "$(grep -c rl_round_trained $B/$P-fnboot/pulled/rl-island-0.jsonl 2>/dev/null)" -ge 6 ]; then log "fnboot already trained 6 rounds (torch_dist existed) -> it is the fnA run"; exit 0; fi
timeout 120 $SKY status $CL 2>/dev/null | grep -q ' UP ' || { log "fn cluster not UP after fnboot -> stop"; exit 2; }
s=$(spent); python3 -c "import sys;sys.exit(0 if $s+30<=$CAP else 1)" || { log "budget guard before fnprep (\$$s) -> fnprep/fna not executed"; exit 4; }
$D/s1reset.sh $CL $B/$P-fnprep.reset.txt || { log "reset NOT CLEAN before fnprep -> stop"; exit 3; }
mkdir -p $B/$P-fnprep; git -C $REPO archive HEAD | (rm -rf $B/$P-fnprep/yeto && mkdir -p $B/$P-fnprep/yeto && tar x -C $B/$P-fnprep/yeto)
$D/s11fnprep.sh $CL $B/$P-fnprep $B/$P-fnprep/yeto > $B/$P-fnprep.out 2>&1; log "fnprep $(tail -1 $B/$P-fnprep.out) $(cat $B/$P-fnprep/fnprep.json 2>/dev/null | tr -d '\n ' | cut -c1-300)"
grep -q FNPREP_OK $B/$P-fnprep.out || { log "fnprep failed -> fnA not executed"; exit 6; }
s=$(spent); python3 -c "import sys;sys.exit(0 if $s+25<=$CAP else 1)" || { log "budget guard before fna (\$$s) -> fnA not executed"; exit 4; }
$D/s1reset.sh $CL $B/$P-fna.reset.txt || { log "reset NOT CLEAN before fna -> stop"; exit 3; }
fnrun fna ${HARD_FNA:-3600}
# seg 3d fnconv (B0-2, user-approved 10-06): full HF -> torch_dist on the same FS cluster, after fnA (chain tail; never re-run on failure:
# a partial output dir makes s11fnconv.sh FAIL instead of retrying). Est $22-50 [speculative; conversion time never measured] -> guard
# uses FNCONV_EST_USD (50) inside the same CAP. FN_CONV=0 skips. fnA failing does not block it (independent of the 4layer run).
[ "${FN_CONV:-$( [ $FG = h100 ] && echo 0 || echo 1 )}" = 1 ] || { log "fnconv disabled (FN_CONV=0; default 0 on FN_GPU=h100: TP2 PP4 ~45GB/rank + HF-load peak unverified on 80GB)"; exit 0; }
timeout 120 $SKY status $CL 2>/dev/null | grep -q ' UP ' || { log "fn cluster not UP before fnconv -> stop"; exit 2; }
s=$(spent); python3 -c "import sys;sys.exit(0 if $s+${FNCONV_EST_USD:-50}<=$CAP else 1)" || { log "budget guard before fnconv (\$$s + \$${FNCONV_EST_USD:-50} > \$$CAP) -> fnconv not executed"; exit 4; }
$D/s1reset.sh $CL $B/$P-fnconv.reset.txt || { log "reset NOT CLEAN before fnconv -> stop"; exit 3; }
mkdir -p $B/$P-fnconv; tc=$(date +%s)
$D/s11fnconv.sh $CL $B/$P-fnconv $B/$P-fnprep/yeto > $B/$P-fnconv.out 2>&1
log "fnconv $(tail -1 $B/$P-fnconv.out) wall=$(( $(date +%s) - tc ))s spent~\$$(spent) $(tr -d '\n ' < $B/$P-fnconv/fnconv.json 2>/dev/null | cut -c1-400)"

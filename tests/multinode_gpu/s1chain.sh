#!/bin/bash
# §3 chain on ONE kept 2x1xL40S cluster: g12 (G1+G2) -> reset -> g3 (G3, launcher teardown = G4) -> cleanup_run.sh (nebius/sky/procs clean twice) -> judgments.
# usage: s1chain.sh <prefix>      env: RUN_ROOT, HARD_G12 (3600 = 30 min case window + up to 30 min cold start: image pull, setup, model fetch; the 30 min case window is judged from "Job submitted"), HARD_G3 (1800, warm cluster)
set -u
P=$1; D=$(cd "$(dirname "$0")" && pwd); B=${RUN_ROOT:-/home/michael/work/s1-runs}; CL=$P-l0-eu-north1; export HOME=/home/michael
LOG=$B/$P.chain.log; log() { echo "[chain $(date -u +%FT%TZ)] $*" | tee -a $LOG; }
finish() { log "final cleanup_run.sh $P"; RUNS_BASE=$B /home/michael/work/gpu-b1-runs/cleanup_run.sh $P > $B/$P.cleanup.out 2>&1; crc=$?; log "cleanup rc=$crc"; cp $B/$P.cleanup.out $B/$P-g3/cleanup.out 2>/dev/null; }
trap finish EXIT
log "start $P"
CLUSTER_PREFIX=$P RUN_ROOT=$B $D/s1run.sh g12 $P-g12 ${HARD_G12:-3600} ${WD_G12:-4200} > $B/$P-g12.out 2>&1; log "g12 rc=$(cat $B/$P-g12/rc.txt)"
python3 $D/s1judge.py $B/$P-g12 g1 > $B/$P-g12/judge-g1.out 2>&1; log "G1 $(python3 -c "import json;print(json.load(open('$B/$P-g12/judgment-g1.json'))['verdict'])")"
python3 $D/s1judge.py $B/$P-g12 g2 > $B/$P-g12/judge-g2.out 2>&1; log "G2 $(python3 -c "import json;print(json.load(open('$B/$P-g12/judgment-g2.json'))['verdict'])")"
grep -q '"verdict": "PASS"' $B/$P-g12/judgment-g1.json || { log "G1 not PASS -> chain stops (no G3)"; exit 1; }
kill $(cat $B/$P-g12/watchdog.pid) 2>/dev/null
timeout 120 /home/michael/work/gpu-head/venv/bin/sky status $CL | grep -q ' UP ' || { log "cluster $CL not UP after g12 -> stop"; exit 2; }
$D/s1reset.sh $CL $B/$P.reset.txt || { log "reset NOT CLEAN -> stop"; exit 3; }
log "reset ok; starting g3"
# g3 reuses the cluster: the launcher's sky launch on the existing cluster name, no --keep -> its teardown is G4
CLUSTER_PREFIX=$P RUN_ROOT=$B $D/s1run.sh g3 $P-g3 ${HARD_G3:-1800} ${WD_G3:-2400} > $B/$P-g3.out 2>&1; log "g3 rc=$(cat $B/$P-g3/rc.txt)"

#!/bin/bash
# Multinode check chain on ONE kept 2x1xL40S cluster: case topology (checks g1+g2) -> reset -> case fault (fault check; the launcher teardown is the last check) -> cleanup_run.sh (nebius/sky/procs clean twice) -> judgments.
# usage: multinode_check_chain.sh <prefix>      env: RUN_ROOT, YETO_SKY (sky binary), YETO_CLEANUP_RUN (cleanup script), HARD_TOPOLOGY (5400 = 30 min case window + up to 60 min cold start (ghcr pull retries took 53 min once); a launcher killed by timeout tears the island down (SIGTERM path), so the window is generous and the per-case judgement uses Job submitted: image pull, setup, model fetch; the 30 min case window is judged from "Job submitted"), HARD_FAULT (1800, warm cluster)
set -u
P=$1; D=$(cd "$(dirname "$0")" && pwd); B=${RUN_ROOT:-${YETO_HOME:-$HOME}/yeto-gpu-runs}; CL=$P-l0-eu-north1; export HOME=${YETO_HOME:-$HOME}
LOG=$B/$P.chain.log; log() { echo "[chain $(date -u +%FT%TZ)] $*" | tee -a $LOG; }
finish() { log "final cleanup_run.sh $P"; RUNS_BASE=$B ${YETO_CLEANUP_RUN:-cleanup_run.sh} $P > $B/$P.cleanup.out 2>&1; crc=$?; log "cleanup rc=$crc"; cp $B/$P.cleanup.out $B/$P-fault/cleanup.out 2>/dev/null; }
trap finish EXIT
log "start $P"
CLUSTER_PREFIX=$P RUN_ROOT=$B $D/multinode_run.sh topology $P-topology ${HARD_TOPOLOGY:-5400} ${WD_TOPOLOGY:-6000} > $B/$P-topology.out 2>&1; log "topology rc=$(cat $B/$P-topology/rc.txt)"
python3 $D/multinode_judge.py $B/$P-topology g1 > $B/$P-topology/judge-g1.out 2>&1; log "check g1 $(python3 -c "import json;print(json.load(open('$B/$P-topology/judgment-g1.json'))['verdict'])")"
python3 $D/multinode_judge.py $B/$P-topology g2 > $B/$P-topology/judge-g2.out 2>&1; log "check g2 $(python3 -c "import json;print(json.load(open('$B/$P-topology/judgment-g2.json'))['verdict'])")"
grep -q '"verdict": "PASS"' $B/$P-topology/judgment-g1.json || { log "check g1 not PASS -> chain stops (no fault case)"; exit 1; }
kill $(cat $B/$P-topology/watchdog.pid) 2>/dev/null
timeout 120 ${YETO_SKY:-sky} status $CL | grep -q ' UP ' || { log "cluster $CL not UP after topology -> stop"; exit 2; }
$D/multinode_reset.sh $CL $B/$P.reset.txt || { log "reset NOT CLEAN -> stop"; exit 3; }
log "reset ok; starting fault case"
# the fault case reuses the cluster: the launcher's sky launch on the existing cluster name, no --keep -> its teardown is the last check
CLUSTER_PREFIX=$P RUN_ROOT=$B $D/multinode_run.sh fault $P-fault ${HARD_FAULT:-1800} ${WD_FAULT:-2400} > $B/$P-fault.out 2>&1; log "fault rc=$(cat $B/$P-fault/rc.txt)"

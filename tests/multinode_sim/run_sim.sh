#!/usr/bin/env bash
# rl-multinode-island task 2.1: two Ray "nodes" on one machine (CPU only), then the
# rehearsal cases of sim.py, each under a hard timeout. Logs -> $OUT.
set -u
PY=${PY:-/tmp/review-miles-venv/bin/python}
RAY=${RAY:-$(dirname "$PY")/ray}
OUT=${OUT:-/home/michael/work/infra-drafts/s1-sim}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" YETO_SIM_RAY="$RAY"
export YETO_SIM_HEAD_DIR=/tmp/yeto-s1-ray-h YETO_SIM_WORKER_DIR=/tmp/yeto-s1-ray-w YETO_SIM_ADDRESS=127.0.0.1:6379
mkdir -p "$OUT"
threads() { ps -eLf | wc -l; }
cleanup() {
  pkill -f "$YETO_SIM_WORKER_DIR/" >/dev/null 2>&1 || true
  pkill -f "$YETO_SIM_HEAD_DIR/" >/dev/null 2>&1 || true
  timeout 60 "$RAY" stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT
echo "threads before: $(threads)" | tee "$OUT/summary.txt"
cleanup
timeout 120 "$RAY" start --head --node-ip-address=127.0.0.1 --port=6379 --num-cpus=8 --num-gpus=4 --resources='{"yeto_node:0": 1}' \
  --temp-dir="$YETO_SIM_HEAD_DIR" --include-dashboard=false >"$OUT/ray-head.log" 2>&1 || { echo "head start failed" | tee -a "$OUT/summary.txt"; exit 1; }
timeout 120 "$RAY" start --address=127.0.0.1:6379 --num-cpus=4 --num-gpus=4 --resources='{"yeto_node:1": 1}' \
  --temp-dir="$YETO_SIM_WORKER_DIR" >"$OUT/ray-worker.log" 2>&1 || { echo "worker start failed" | tee -a "$OUT/summary.txt"; exit 1; }
echo "threads with 2 ray nodes: $(threads)" | tee -a "$OUT/summary.txt"
rc_all=0
# The driver's Ray registration hangs intermittently on this host (A27B-PROGRESS:
# process_failed_pending_registration); a hang is a fresh-process retry, not a case failure.
for case in ${CASES:-pg_blocks cells_bind node_loss teardown}; do
  rc=1
  for attempt in 1 2 3; do
    timeout 300 "$PY" "$REPO/tests/multinode_sim/sim.py" "$case" >"$OUT/$case.log" 2>&1
    rc=$?
    if grep -q "RegisterClient" "$OUT/$case.log" && [ $rc -ne 0 ]; then
      echo "$case attempt $attempt: ray registration hang (environmental), retrying" | tee -a "$OUT/summary.txt"
      cp "$OUT/$case.log" "$OUT/$case.hang$attempt.log"; continue
    fi
    break
  done
  echo "$case rc=$rc $(grep -o 'PASS [a-z_]*' "$OUT/$case.log" | tail -1)" | tee -a "$OUT/summary.txt"
  [ $rc -eq 0 ] || rc_all=1
done
cleanup
sleep 3
echo "threads after: $(threads)" | tee -a "$OUT/summary.txt"
exit $rc_all

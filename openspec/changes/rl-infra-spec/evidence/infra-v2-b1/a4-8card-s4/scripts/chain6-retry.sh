#!/bin/bash
# Retry chain 6 (A4 top-up) every 20 min while Nebius eu-north1 has no 8xH100 capacity. Stops at the first chain that gets past provisioning, or at the deadline.
B=/home/michael/work/gpu-b1-runs; DEADLINE=$(date -d "2026-10-02 07:30" +%s); N=${START_N:-1}
log(){ echo "[retry $(date -u +%FT%TZ)] $*"; }
while [ $(date +%s) -lt $DEADLINE ]; do
  CP=infra-v2-b1-a4s6-20261001-6r$N; log "attempt $N: $CP"
  (cd $B && SHA=3be4933 CAP_USD=${CAP:-170} THREAD_MAX=3600 ./chain8.sh $CP ${SPENT:-108.2} ${ITEMS:-d2:2100 e1b:1800 wd:1500 d4:1500} > $B/chain6-$CP.nohup 2>&1 < /dev/null)
  log "attempt $N ended: $(tail -1 $B/$CP/chain.log)"
  if grep -q 'startup_failed' $B/$CP/chain.log && grep -q 'Failed to acquire resources\|ResourcesUnavailable' $B/$CP/items/*-d2.n2run.out 2>/dev/null; then
    until grep -q 'final cleanup rc' $B/$CP/chain.log; do sleep 10; done
    N=$((N+1)); log "capacity unavailable; sleeping 20 min"; sleep 1200; continue
  fi
  log "chain ran (not a capacity failure); stop retrying"; exit 0
done
log "deadline reached; stop"

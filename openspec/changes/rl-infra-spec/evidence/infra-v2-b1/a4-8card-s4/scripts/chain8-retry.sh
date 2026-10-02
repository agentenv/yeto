#!/bin/bash
# Retry chain 8 (gated A4/E1-D chain on the new image) every 20 min while Nebius eu-north1 has no 8xH100 capacity. Stops at the first chain that gets past provisioning, or at the deadline.
B=/home/michael/work/gpu-b1-runs; DEADLINE=$(( $(date +%s) + ${MAX_S:-43200} )); N=${START_N:-1}
log(){ echo "[retry8 $(date -u +%FT%TZ)] $*"; }
while [ $(date +%s) -lt $DEADLINE ]; do
  CP=infra-v2-b1-a4s8-20261002-1r$N; log "attempt $N: $CP"
  (cd $B && SHA=${SHA:-2eb415f3} ATTEST=${ATTEST:-$B/cfg/attestation-8-6-71672312.json} A8GO=$B/a8go_strict.sh RESET=$B/reset_island_strict.sh CAP_USD=700 THREAD_MAX=3600 \
     ./chain8.sh $CP ${SPENT:-533} ${ITEMS:-d2:2400 a4bc:1500 r6:1800 r7:2400 r5:2400 d4:1500} > $B/chain8-$CP.nohup 2>&1 < /dev/null)
  log "attempt $N ended: $(tail -1 $B/$CP/chain.log)"
  until grep -q 'final cleanup rc' $B/$CP/chain.log; do sleep 10; done
  if grep -q "${FIRST:-d2}: \(ABORT\|done rc=6\)" $B/$CP/chain.log && grep -qa 'ResourcesUnavailable\|Failed to acquire resources' $B/$CP/items/$CP-${FIRST:-d2}/launch.log 2>/dev/null; then
    N=$((N+1)); log "capacity unavailable; sleeping 20 min"; sleep 1200; continue
  fi
  log "chain ran (not a capacity failure); stop retrying"; exit 0
done
log "deadline reached; stop"

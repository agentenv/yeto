#!/bin/bash
# usage: nstop_item_strict.sh <item prefix>   (a8go_strict.sh final guard; env RUN_ROOT, CHAIN_DIR, CLUSTER_PREFIX as set by chain8.sh; NSTOP_INNER = the real stop script)
# Strict-avg items have a host-side syncer: kill its tree (syncer_host_clean.sh, scoped to this run dir, idempotent), keep its log/checkpoint in <run>/home,
# then hand over to the unchanged nstop_item.sh (chain mode: pull evidence, never release) or nstop.sh (single run: pull + cleanup_run.sh).
# The existing nstop_item.sh / nstop.sh are NOT modified (chain 6 uses them live).
P=$1; B=${BDIR:-/home/michael/work/gpu-b1-runs}; R=${RUN_ROOT:-$B}/$P; INNER=${NSTOP_INNER:-$B/nstop_item.sh}
$B/syncer_host_clean.sh $R >> $R/syncer_clean.txt 2>&1; src=$?
echo "syncer_host_clean rc=$src" >> $R/syncer_clean.txt
cp $R/home/yeto-syncer.log $R/pulled/yeto-syncer.log 2>/dev/null
cp $R/home/yeto-output/yeto-tape.jsonl $R/pulled/syncer-tape.jsonl 2>/dev/null
$INNER $P; rc=$?
[ $src = 0 ] || { echo "WARNING: host syncer of $P not clean (see $R/syncer_clean.txt)"; [ $rc = 0 ] && rc=3; }
exit $rc

#!/bin/bash
# usage: chain8.sh <chain prefix CP> <spent_before_usd> <case:hard_s[:sha]> [<case:hard_s[:sha]> ...]      (gpu-plan-v2 9.23: several A4 items on ONE kept 8xH100 sky cluster)
# env: CAP_USD (100)  PRICE_PER_MIN (0.5133)  UP_DEADLINE_S (240; wd)  + test hooks: A8GO RESET SKY CLEANUP NSTOP_ITEM THREAD_CMD POLL_S CHAIN_WD_S
# Flow: first item cold-starts the cluster (a8go.sh with CLUSTER_PREFIX=CP KEEP=1 SHARED=1); every later item: gate -> cancel leftover job -> reset_island.sh (must verify a fresh island) -> a8go.sh.
# Gate before each item:  spent_before + cost of the chain so far (wall since the chain started x price) + this item's worst case (hard_s x price)  <=  CAP_USD, else the item and the rest are "not run (budget)".
# Any startup failure, unclean reset, cluster no longer UP, or chain-level watchdog => stop and release (cleanup_run.sh CP must exit 0).  The chain ALWAYS ends with cleanup_run.sh CP (trap), so the cluster is never left idle.
CP=$1; SPENT=$2; shift 2; ITEMS=("$@")
B=${BDIR:-/home/michael/work/gpu-b1-runs}; CHAIN=$B/$CP; CL=$CP-l0-eu-north1; CAP=${CAP_USD:-100}; PPM=${PRICE_PER_MIN:-0.5133}
A8GO=${A8GO:-$B/a8go.sh}; RESET=${RESET:-$B/reset_island.sh}; SKY=${SKY:-/home/michael/work/gpu-head/venv/bin/sky}; CLEANUP=${CLEANUP:-$B/cleanup_run.sh}; NSTOP_ITEM=${NSTOP_ITEM:-$B/nstop_item.sh}
THREAD_CMD=${THREAD_CMD:-"ps -u michael -L -o pid= | wc -l"}; POLL_S=${POLL_S:-15}; export HOME=/home/michael
mkdir -p $CHAIN/items; echo $CL > $CHAIN/cluster.txt; LOG=$CHAIN/chain.log; log() { echo "[chain $(date -u +%FT%TZ)] $*" | tee -a $LOG; }
T0=$(date +%s); WDPID=""; CLEANED=0; FINAL_RC=0
finish() {
  [ $CLEANED = 1 ] && return; CLEANED=1
  log "final cleanup_run.sh $CP"; $CLEANUP $CP > $CHAIN/final_cleanup.out 2>&1; crc=$?; echo $crc > $CHAIN/final_cleanup_rc.txt; log "final cleanup rc=$crc"
  [ -n "$WDPID" ] && kill $WDPID 2>/dev/null; [ $crc = 0 ] || FINAL_RC=2
  echo "$FINAL_RC" > $CHAIN/chain_rc.txt
}
trap 'log "signal"; finish; exit 130' INT TERM; trap 'finish' EXIT
sumhard=0; for it in "${ITEMS[@]}"; do h=${it#*:}; h=${h%%:*}; sumhard=$((sumhard + h)); done
WD_S=${CHAIN_WD_S:-$((sumhard + 1500))}
setsid nohup bash -c "sleep $WD_S; HOME=/home/michael $SKY down -y $CL > $CHAIN/chain_watchdog.out 2>&1; touch $CHAIN/CHAIN_WD_FIRED" > /dev/null 2>&1 & WDPID=$!
log "chain $CP start spent_before=\$$SPENT cap=\$$CAP items=${ITEMS[*]} chain watchdog ${WD_S}s"
cost_now() { python3 -c "import sys;print(round($SPENT + ($(date +%s) - $T0)/60*$PPM, 2))"; }
first=1
for it in "${ITEMS[@]}"; do
  case=${it%%:*}; rest=${it#*:}; hard=${rest%%:*}; isha=""; [ "$rest" != "$hard" ] && isha=${rest#*:}   # optional 3rd field = code SHA for this item (same image digest only: each item syncs its own `git archive <sha>` workdir)
  P=$CP-$case; R=$CHAIN/items/$P
  [ -s $CHAIN/cap.txt ] && CAP=$(cat $CHAIN/cap.txt)   # the user may raise/lower the cap while the chain runs (echo 120 > <chain>/cap.txt)
  worst=$(python3 -c "print(round($hard/60*$PPM,2))"); now=$(cost_now)
  if ! python3 -c "import sys;sys.exit(0 if $now + $worst <= $CAP else 1)"; then log "$case: NOT RUN (budget): spent_so_far=\$$now worst=\$$worst cap=\$$CAP"; echo "{\"item\":\"$case\",\"status\":\"not_run_budget\",\"spent_so_far\":$now,\"worst\":$worst}" >> $CHAIN/items.jsonl; continue; fi
  if [ $first = 1 ]; then
    n=$(eval "$THREAD_CMD"); if [ "$n" -ge ${THREAD_MAX:-3000} ]; then log "$case: user threads $n >= ${THREAD_MAX:-3000} before the cold start: nothing provisioned, chain stops"; FINAL_RC=3; break; fi
  else
    if ! $SKY status $CL 2>/dev/null | grep -q ' UP '; then log "$case: cluster $CL is not UP any more -> chain stops (no re-provisioning inside a chain)"; FINAL_RC=4; break; fi
    $SKY cancel -a -y $CL > $CHAIN/cancel-$case.out 2>&1
    if ! $RESET $CL $CHAIN/reset-$case.txt; then log "$case: island reset NOT CLEAN (see reset-$case.txt) -> chain stops and releases the cluster"; FINAL_RC=5; break; fi
    log "$case: island reset ok (fresh: no compute apps, GPU memory < 1.5 GiB each, state dirs removed)"
    n=$(eval "$THREAD_CMD"); if [ "$n" -ge ${THREAD_MAX:-3000} ]; then log "$case: user threads $n >= ${THREAD_MAX:-3000}: chain stops and releases the cluster (no idle waiting)"; FINAL_RC=3; break; fi
  fi
  log "$case: start (hard ${hard}s, worst \$$worst, spent_so_far \$$now)"
  rm -f $R/item_done
  env ${isha:+SHA=$isha} RUN_ROOT=$CHAIN/items CLUSTER_PREFIX=$CP KEEP=1 SHARED=1 NSTOP=$NSTOP_ITEM CHAIN_DIR=$CHAIN UP_DEADLINE_S=${UP_DEADLINE_S:-240} $A8GO $case $P $hard $((hard + 120)) > $CHAIN/start-$case.out 2>&1
  src=$?; if [ $src != 0 ]; then log "$case: a8go failed to start (rc=$src): $(tail -2 $CHAIN/start-$case.out | tr '\n' ' ')"; FINAL_RC=8; break; fi
  tw=$(date +%s)
  until [ -f $R/item_done ] || [ -f $CHAIN/ABORT ] || [ -f $CHAIN/CHAIN_WD_FIRED ] || grep -qs '^abort:' $R.n2run.out || [ $(( $(date +%s) - tw )) -gt $((hard + ${ITEM_GRACE_S:-900})) ]; do sleep $POLL_S; done
  if grep -qs '^abort:' $R.n2run.out; then log "$case: n2run aborted before launch: $(head -1 $R.n2run.out) -> chain stops and releases the cluster"; FINAL_RC=10; break; fi
  if [ ! -f $R/item_done ] && [ ! -f $CHAIN/ABORT ] && [ ! -f $CHAIN/CHAIN_WD_FIRED ]; then log "$case: no item_done within hard+grace -> chain stops"; FINAL_RC=9; break; fi
  if [ -f $CHAIN/ABORT ]; then log "$case: ABORT: $(cat $CHAIN/ABORT)"; FINAL_RC=6; break; fi
  if [ -f $CHAIN/CHAIN_WD_FIRED ]; then log "$case: chain watchdog fired"; FINAL_RC=7; break; fi
  python3 $B/scan_run.py $R > $R/scan.out 2>&1
  # new-image first-use rule (gpu-plan 9.23): a LoRA admission refused as lora_unverifiable (read-back carries no adapter keys) => stop at once, do not run later items
  lu=""; for f in $R/pulled/rl-island-0.final.jsonl $R/launch.log; do grep -qs 'lora_unverifiable' $f && lu="$lu $f"; done
  rm -rf $R/.jx; mkdir -p $R/.jx; b=$R/pulled/elastic-state-final.b64; [ -s $b ] && base64 -d $b 2>/dev/null | tar xz -C $R/.jx 2>/dev/null; grep -qs 'lora_unverifiable' $R/.jx/elastic-state/reconfig/journal.jsonl && lu="$lu journal"
  if [ -n "$lu" ]; then log "$case: LORA_UNVERIFIABLE seen in:$lu -> chain stops and releases the cluster (report; do not continue)"; echo "{\"item\":\"$case\",\"status\":\"lora_unverifiable\"}" >> $CHAIN/items.jsonl; FINAL_RC=11; break; fi
  echo "{\"item\":\"$case\",\"status\":\"ran\",\"rc\":\"$(cat $R/rc.txt 2>/dev/null)\",\"spent_so_far\":$(cost_now)}" >> $CHAIN/items.jsonl
  log "$case: done $(cat $R/rc.txt 2>/dev/null) spent_so_far=\$$(cost_now)"
  first=0
done
log "chain loop ended (FINAL_RC=$FINAL_RC); spent_estimate=\$$(cost_now)"
finish; exit $FINAL_RC

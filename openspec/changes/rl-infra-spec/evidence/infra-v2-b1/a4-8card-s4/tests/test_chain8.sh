#!/bin/bash
# CPU-only test of chain8.sh with stub a8go/reset/sky/cleanup/nstop: ordering, budget gate, reset failure, ABORT, thread guard, cluster-not-UP, always-final-cleanup.
set -u
B=$(cd "$(dirname "$0")/.." && pwd); [ -f $B/chain8.sh ] || B=$B/scripts; T=$(mktemp -d); trap 'rm -rf $T; pkill -f "sleep 7[0-9][0-9][0-9]" 2>/dev/null' EXIT
fail=0; ok() { echo "PASS $1"; }; bad() { echo "FAIL $1"; fail=1; }
mkdir -p $T/bin $T/b; cp $B/chain8.sh $T/b/; cp $B/scan_run.py $T/b/ 2>/dev/null
cat > $T/bin/a8go <<'S'
#!/bin/bash
# stub a8go: args case prefix hard wd; env RUN_ROOT; records, then (async) finishes the item
echo "$1 $2 $3 $4 KEEP=$KEEP SHARED=$SHARED CP=$CLUSTER_PREFIX RUN_ROOT=$RUN_ROOT" >> $STUBLOG/a8go.log
R=$RUN_ROOT/$2; mkdir -p $R
[ -f $STUBDIR/n2abort_$1 ] && { echo "abort: 3008 user threads" > $R.n2run.out; exit 0; }
[ -f $STUBDIR/a8go_fail_$1 ] && { echo boom; exit 5; }
( sleep 1; [ -f $STUBDIR/abort_$1 ] && echo '{"reason":"startup_failed"}' > $CHAIN_DIR/ABORT || { echo rc=0 > $R/rc.txt; touch $R/item_done; } ) &
exit 0
S
cat > $T/bin/reset <<'S'
#!/bin/bash
echo "$1" >> $STUBLOG/reset.log; [ -f $STUBDIR/reset_fail ] && exit 1; exit 0
S
cat > $T/bin/sky <<'S'
#!/bin/bash
echo "sky $*" >> $STUBLOG/sky.log
case "$1" in status) [ -f $STUBDIR/notup ] && echo "$2 INIT 1m" || echo "$2 UP 1m ago";; esac; exit 0
S
cat > $T/bin/cleanup <<'S'
#!/bin/bash
echo "$1" >> $STUBLOG/cleanup.log; exit ${CLEANUP_RC:-0}
S
chmod +x $T/bin/*
run() { # name expected_rc spent items...   (env from caller)
  local name=$1 want=$2 spent=$3; shift 3; local CP=infra-v2-test-chain-$name
  rm -rf $T/log $T/st; mkdir -p $T/log $T/st; for f in ${SETUP:-}; do touch $T/st/$f; done
  STUBLOG=$T/log STUBDIR=$T/st BDIR=$T/b A8GO=$T/bin/a8go RESET=$T/bin/reset SKY=$T/bin/sky CLEANUP=$T/bin/cleanup NSTOP_ITEM=/bin/true POLL_S=1 CHAIN_WD_S=7777 THREAD_CMD="echo ${THREADS:-2000}" CAP_USD=100 PRICE_PER_MIN=0.5133 \
    bash $T/b/chain8.sh $CP $spent "$@" > $T/out.$name 2>&1; rc=$?
  [ $rc = $want ] && ok "$name rc=$rc" || { bad "$name rc=$rc want $want"; tail -5 $T/out.$name; }
  [ "$(grep -c . $T/log/cleanup.log 2>/dev/null)" = 1 ] && ok "$name: final cleanup exactly once" || bad "$name: cleanup calls=$(grep -c . $T/log/cleanup.log 2>/dev/null)"
}
SETUP="" THREADS=2000 run normal 0 20 wd:60 a4b:60 d4:60
[ "$(cut -d' ' -f1 $T/log/a8go.log | tr '\n' ' ')" = "wd a4b d4 " ] && ok "normal: item order" || bad "normal order: $(cat $T/log/a8go.log)"
[ "$(grep -c . $T/log/reset.log)" = 2 ] && ok "normal: reset before every reused item (2), not before the cold start" || bad "normal: reset calls $(grep -c . $T/log/reset.log)"
grep -q 'KEEP=1 SHARED=1 CP=infra-v2-test-chain-normal' $T/log/a8go.log && ok "normal: chain env passed" || bad "normal: env"
grep -q 'infra-v2-test-chain-normal-wd ' $T/log/a8go.log && ok "normal: item prefix = <CP>-<case>" || bad "item prefix"
[ -f $T/b/infra-v2-test-chain-normal/chain_rc.txt ] && ok "normal: chain_rc written" || bad "chain_rc"
SETUP="" run budget 0 99.9 wd:600 a4b:600
[ ! -s $T/log/a8go.log ] && ok "budget: nothing started when spent+worst > cap" || bad "budget started"
grep -c not_run_budget $T/b/infra-v2-test-chain-budget/items.jsonl | grep -q 2 && ok "budget: both items marked not_run_budget" || bad "budget marks"
SETUP="reset_fail" run resetfail 5 0 wd:60 a4b:60 d4:60
[ "$(cut -d' ' -f1 $T/log/a8go.log | tr '\n' ' ')" = "wd " ] && ok "resetfail: stopped after the first item" || bad "resetfail order"
SETUP="abort_wd" run abort 6 0 wd:60 a4b:60
SETUP="notup" run notup 4 0 wd:60 a4b:60
THREADS=3100 SETUP="" run threads_cold 3 0 wd:60
[ ! -s $T/log/a8go.log ] && ok "threads: no cold start at >=3000" || bad "threads cold start"
SETUP="a8go_fail_a4b" run a8gofail 8 0 wd:60 a4b:60
SETUP="n2abort_a4b" run n2abort 10 0 wd:60 a4b:60
CLEANUP_RC=2 SETUP="" run cleanup_rc2 2 0 wd:60
[ "$fail" = 0 ] && echo "ALL PASS" || { echo "SOME FAILED"; exit 1; }

#!/bin/bash
# CPU-only tests of the strict-avg (option A) wrappers: a8go_strict.sh DRY output (no no-sync, EXR/kill switches, arms, tprobe), n2run_strict.sh preflight
# (busy syncer port -> "abort:" line, rc 3; the launch args never carry --rl-single-island-no-sync / --controller local), syncer_host_clean.sh (no-op without the
# marker; kills only command lines under <run dir>/home/yeto-syncer; idempotent), nstop_item_strict.sh (cleans, then hands over to NSTOP_INNER),
# reset_island_strict.sh (port free -> inner reset rc propagated; port held -> rc 1, inner never called).  Never touches a real syncer, cloud or GPU.
set -u
B=$(cd "$(dirname "$0")/.." && pwd); [ -f $B/a8go_strict.sh ] || B=$B/scripts; T=$(mktemp -d); trap 'rm -rf $T' EXIT; fail=0
ok() { echo "PASS $1"; }; bad() { echo "FAIL $1"; fail=1; }
for f in a8go_strict.sh n2run_strict.sh nstop_item_strict.sh reset_island_strict.sh syncer_host_clean.sh; do bash -n $B/$f && ok "syntax $f" || bad "syntax $f"; done
# --- a8go_strict DRY
for c in s0 r6 r7 r5 r5c; do SHA=deadbee DRY=1 bash $B/a8go_strict.sh $c infra-v2-test-$c 2400 2520 > $T/dry.$c 2>&1 || bad "dry $c rc"; done
grep -q "n2run_strict.sh infra-v2-test-s0 8 2400 2520 --total-steps 6 " $T/dry.s0 && ! grep -q "rl-elastic-restart-attempts\|kill-learner" $T/dry.s0 && ok "s0: strict runner, no kill/restart switches" || bad "s0 dry"
for c in s0 r6 r7 r5 r5c; do grep -q "rl-single-island-no-sync\|--controller local" $T/dry.$c && bad "$c carries no-sync/controller"; grep -q triggers-json-ok $T/dry.$c || bad "$c triggers json"; done; ok "no case carries --rl-single-island-no-sync / --controller local; triggers are JSON"
grep -q -- "--rl-elastic-restart-attempts 2 --rl-elastic-max-recovery-attempts 3 --rl-test-kill-learner-at QUIESCING" $T/dry.r6 && grep -q '"up1"' $T/dry.r6 && grep -q "judge: r6" $T/dry.r6 && ok "r6 switches/judge" || bad "r6"
grep -q -- "--rl-elastic-restart-attempts 2 --rl-elastic-max-recovery-attempts 3\$" $T/dry.r7 && grep -q "dctl.py|kill_after_tx|up1|dn1|" $T/dry.r7 && grep -q "tprobe: 1 rec" $T/dry.r7 && ok "r7: EXR only, kill_after_tx arm, term_probe rec" || bad "r7"
grep -q -- "--rl-test-kill-learner-at COMMITTED" $T/dry.r5 && grep -q "tprobe: 1 rec" $T/dry.r5 && grep -q "arms: none" $T/dry.r5 && ok "r5: kill at COMMITTED, probe rec, no arm" || bad "r5"
grep -q -- "--rl-test-kill-learner-at COMMITTED" $T/dry.r5c && grep -q "dctl.py|marker|up1" $T/dry.r5c && grep -q '"dn1"' $T/dry.r5c && ok "r5c: marker arm + dn1 trigger" || bad "r5c"
for c in s0 r6 r7 r5 r5c; do grep -q -- "--rl-elastic-quorum-timeout-s 1800 " $T/dry.$c || bad "$c lacks --rl-elastic-quorum-timeout-s 1800 (s0 lesson: 450 s default budget rejects 600 s deadlines)"; done; ok "every strict case raises the syncer quorum timeout (pause budget 900 s >= 600 s deadlines)"
grep -q "attestation-8-6.json" $T/dry.r5 && ok "attestation-8-6 (fingerprint unchanged without no-sync)" || bad "attestation"
SHA=x DRY=1 bash $B/a8go_strict.sh d2 p 1 1 >/dev/null 2>&1; [ $? = 64 ] && ok "existing a8go cases are not accepted here (rc 64)" || bad "unknown case rc"
# --- n2run_strict preflight: busy port
mkdir -p $T/bin; cat > $T/bin/ss <<'S'
#!/bin/bash
echo "State Recv-Q Send-Q Local Address:Port Peer Address:Port"; [ -f $STUBDIR/busy ] && echo "LISTEN 0 128 0.0.0.0:29400 0.0.0.0:*"; echo "LISTEN 0 128 127.0.0.1:22 0.0.0.0:*"
S
chmod +x $T/bin/ss; mkdir -p $T/st; touch $T/st/busy
out=$(STUBDIR=$T/st PATH=$T/bin:$PATH RUN_ROOT=$T/rr THREAD_MAX=999999 bash $B/n2run_strict.sh infra-v2-test-busy 8 10 20 --total-steps 6 2>&1); rc=$?
[ $rc = 3 ] && [[ "$out" == abort:*"port 29400 busy"* ]] && [ ! -d $T/rr/infra-v2-test-busy ] && ok "n2run_strict: busy syncer port -> 'abort:' (chain8 stops), nothing created" || bad "busy port: rc=$rc out=$out"
rm $T/st/busy
out=$(STUBDIR=$T/st PATH=$T/bin:$PATH RUN_ROOT=$T/rr THREAD_MAX=999999 SYNCER_BIN=$T/nope bash $B/n2run_strict.sh infra-v2-test-nobin 8 10 20 --total-steps 6 2>&1); rc=$?
[ $rc = 3 ] && [[ "$out" == abort:*missing* ]] && ok "n2run_strict: missing syncer binary -> abort" || bad "nobin: rc=$rc out=$out"
# the launch line (same template as n2run.sh minus no-sync/controller): check the args.txt text the script would write
grep -o 'echo "launch [^>]*' $B/n2run_strict.sh > $T/launch.tmpl
grep -q -- "--rl-single-island-no-sync\|--controller" $T/launch.tmpl && bad "launch template carries no-sync/controller" || ok "launch template: no --rl-single-island-no-sync, no --controller"
grep -q 'cluster-prefix $CP ${KEEP:+--keep} --no-island-relaunch --modal-retries 0 --model Qwen/Qwen3-0.6B' $T/launch.tmpl && ok "launch template keeps the chain/cluster flags and the model pins" || bad "launch template pins"
grep -q 'SYNCER_PUBLIC_IP=\$SYNCER_PUBLIC_IP' $B/n2run_strict.sh && grep -q 'run_local_head.py \$R/args.txt' $B/n2run_strict.sh && ok "head mode: run_local_head.py with SYNCER_PUBLIC_IP exported" || bad "head mode wiring"
# --- syncer_host_clean: no marker -> no-op; marker -> kills only this run's pattern
R=$T/run1; mkdir -p $R/home; cat > $T/bin/pkill <<'S'
#!/bin/bash
echo "pkill $*" >> $STUBLOG; exit 0
S
cat > $T/bin/pgrep <<'S'
#!/bin/bash
echo "pgrep $*" >> $STUBLOG; [ -f $STUBDIR/alive ] && { echo "4242 /bin/sh -c mkdir -p ~/yeto-output && $RUNDIR/home/yeto-syncer --port 29400"; exit 0; }; exit 1
S
chmod +x $T/bin/pkill $T/bin/pgrep
: > $T/log; STUBLOG=$T/log STUBDIR=$T/st RUNDIR=$R PKILL=$T/bin/pkill PGREP=$T/bin/pgrep bash $B/syncer_host_clean.sh $R > $T/c0 2>&1; rc=$?
[ $rc = 0 ] && grep -q "no host syncer" $T/c0 && [ ! -s $T/log ] && ok "syncer_host_clean: no marker -> no-op (no pkill/pgrep)" || bad "clean no marker rc=$rc $(cat $T/c0)"
echo 185.189.44.160:29400 > $R/syncer_host.txt; touch $T/st/alive; : > $T/log
STUBLOG=$T/log STUBDIR=$T/st RUNDIR=$R PKILL=$T/bin/pkill PGREP=$T/bin/pgrep PATH=$T/bin:$PATH bash $B/syncer_host_clean.sh $R > $T/c1 2>&1; rc=$?
grep -q -- "pkill -TERM -f -- $R/home/yeto-syncer" $T/log && grep -q -- "pkill -KILL -f -- $R/home/yeto-syncer" $T/log && ok "syncer_host_clean: TERM then KILL, pattern = <run dir>/home/yeto-syncer only" || bad "clean pattern: $(cat $T/log)"
[ $rc = 1 ] && ok "syncer_host_clean: survivor -> rc 1" || bad "survivor rc=$rc"
rm $T/st/alive; STUBLOG=$T/log STUBDIR=$T/st RUNDIR=$R PKILL=$T/bin/pkill PGREP=$T/bin/pgrep PATH=$T/bin:$PATH bash $B/syncer_host_clean.sh $R > $T/c2 2>&1; rc=$?
[ $rc = 0 ] && grep -q "left=\[\] port_29400_held_by=\[none\]" $T/c2 && ok "syncer_host_clean: gone + port free -> rc 0 (idempotent)" || bad "clean idempotent rc=$rc $(cat $T/c2)"
# --- nstop_item_strict: cleans then calls NSTOP_INNER with the prefix
printf '#!/bin/bash\necho "inner $*" >> $STUBLOG; exit ${INNER_RC:-0}\n' > $T/bin/inner; chmod +x $T/bin/inner
mkdir -p $T/rr2/infra-v2-test-x/home $T/rr2/infra-v2-test-x/pulled; echo 1.2.3.4:29400 > $T/rr2/infra-v2-test-x/syncer_host.txt; echo syncerlog > $T/rr2/infra-v2-test-x/home/yeto-syncer.log
: > $T/log; STUBLOG=$T/log STUBDIR=$T/st RUNDIR=$T/rr2/infra-v2-test-x PKILL=$T/bin/pkill PGREP=$T/bin/pgrep PATH=$T/bin:$PATH RUN_ROOT=$T/rr2 BDIR=$B NSTOP_INNER=$T/bin/inner bash $B/nstop_item_strict.sh infra-v2-test-x > $T/n1 2>&1; rc=$?
[ $rc = 0 ] && grep -q "inner infra-v2-test-x" $T/log && [ -s $T/rr2/infra-v2-test-x/syncer_clean.txt ] && [ -s $T/rr2/infra-v2-test-x/pulled/yeto-syncer.log ] && ok "nstop_item_strict: clean -> inner stop(prefix) -> rc 0; syncer log kept under pulled/" || bad "nstop_item_strict rc=$rc $(cat $T/n1 $T/log)"
: > $T/log; INNER_RC=2 STUBLOG=$T/log STUBDIR=$T/st RUNDIR=$T/rr2/infra-v2-test-x PKILL=$T/bin/pkill PGREP=$T/bin/pgrep PATH=$T/bin:$PATH RUN_ROOT=$T/rr2 BDIR=$B NSTOP_INNER=$T/bin/inner bash $B/nstop_item_strict.sh infra-v2-test-x > /dev/null 2>&1; rc=$?
[ $rc = 2 ] && ok "nstop_item_strict: inner rc propagated" || bad "inner rc propagate rc=$rc"
# --- reset_island_strict: port free -> inner reset; port held -> rc 1, inner not called
printf '#!/bin/bash\necho "reset $1" >> $STUBLOG; echo RESET_OK > $2; exit ${RESET_RC:-0}\n' > $T/bin/rinner; chmod +x $T/bin/rinner
: > $T/log; STUBLOG=$T/log STUBDIR=$T/st PKILL=$T/bin/pkill PGREP=$T/bin/pgrep PATH=$T/bin:$PATH BDIR=$B RESET_INNER=$T/bin/rinner bash $B/reset_island_strict.sh clu $T/r1.log; rc=$?
[ $rc = 0 ] && grep -q "reset clu" $T/log && grep -q SYNCER_PORT_FREE $T/r1.log && grep -q RESET_OK $T/r1.log && ok "reset_island_strict: port free -> inner reset ran, logs merged" || bad "reset free rc=$rc $(cat $T/r1.log)"
: > $T/log; RESET_RC=1 STUBLOG=$T/log STUBDIR=$T/st PKILL=$T/bin/pkill PGREP=$T/bin/pgrep PATH=$T/bin:$PATH BDIR=$B RESET_INNER=$T/bin/rinner bash $B/reset_island_strict.sh clu $T/r2.log; rc=$?
[ $rc = 1 ] && ok "reset_island_strict: inner NOT CLEAN propagated" || bad "reset inner rc=$rc"
touch $T/st/busy; : > $T/log; STUBLOG=$T/log STUBDIR=$T/st PKILL=$T/bin/pkill PGREP=$T/bin/pgrep PATH=$T/bin:$PATH BDIR=$B RESET_INNER=$T/bin/rinner timeout 120 bash $B/reset_island_strict.sh clu $T/r3.log; rc=$?
[ $rc = 1 ] && ! grep -q "reset clu" $T/log && grep -q SYNCER_PORT_HELD $T/r3.log && ok "reset_island_strict: port held -> rc 1, island reset not attempted" || bad "reset held rc=$rc $(cat $T/r3.log)"
[ $fail = 0 ] && echo "ALL PASS" || { echo "SOME FAILED"; exit 1; }

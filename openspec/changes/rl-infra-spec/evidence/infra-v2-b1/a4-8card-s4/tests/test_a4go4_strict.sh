#!/bin/bash
# CPU-only tests of the 4xL40S strict toolchain (chain 8 IV, 2026-10-03): a4go4_strict.sh DRY output (4 cards, T1R1S2<->T1R3S0, resources-4, cells c0,c1,c2,
# strict runner without no-sync, judge topology args --trainer-world 1 --committed-members 3 + --rc-file, r7 arm target T1R1S2, dispatch of non-strict cases to
# a4go4.sh), a4go4.sh chain protocol (RUN_ROOT, NSTOP, item_done, ATTEST, <run>.n2run.out, d4 judge + hook), chain8-retry.sh env overrides, mkatt4.sh FP/REPO,
# the 5cf4d902 4-card attestation files, judge_after.sh pass-through.  Never touches a cloud, GPU or syncer.  run: bash tests/test_a4go4_strict.sh
set -u
B=$(cd "$(dirname "$0")/.." && pwd); [ -f $B/a4go4_strict.sh ] || B=$B/scripts; T=$(mktemp -d); trap 'rm -rf $T' EXIT; fail=0
ok() { echo "PASS $1"; }; bad() { echo "FAIL $1"; fail=1; }
for f in a4go4_strict.sh a4go4.sh chain8-retry.sh mkatt4.sh fp_local4_strict.py; do [ -f $B/$f ] || { bad "missing $f"; continue; }; case $f in *.sh) bash -n $B/$f && ok "syntax $f" || bad "syntax $f";; *.py) python3 -c "import ast;ast.parse(open('$B/$f').read())" && ok "parses $f" || bad "parse $f";; esac; done
CFG=$B/cfg; [ -d $CFG ] || CFG=$B/../cfg
# --- a4go4_strict DRY (ATTEST given: the default attestation-4-6.json is the b19b781 one)
touch $T/att.json
for c in r6 r7 r5 r5c s0; do SHA=deadbee DRY=1 ATTEST=$T/att.json RUN_ROOT=$T/rr bash $B/a4go4_strict.sh $c infra-v2-test4-$c 3000 3120 > $T/dry.$c 2>&1 || bad "dry $c rc"; grep -q triggers-json-ok $T/dry.$c || bad "$c triggers json"; done
for c in r6 r7 r5 r5c s0; do
  grep -q "GPU_SPEC=nebius:4xl40s@eu-north1 .*n2run_strict.sh infra-v2-test4-$c 4 3000 3120 --total-steps 6 --rl-placement fixed-partition --rl-rollout-gpus 1 --rl-standby-gpus 2 --rl-elastic --rl-elastic-declare-cells --rl-elastic-cells c0,c1,c2 --rl-elastic-resources /home/michael/work/gpu-b1-runs/cfg/resources-4.json --rl-elastic-initial-config T1R1S2 --rl-observe-timeline --rl-elastic-attestation $T/att.json --rl-elastic-quorum-timeout-s 1800" $T/dry.$c || bad "$c 4-card strict launch line: $(head -1 $T/dry.$c)"
  grep -q "rl-single-island-no-sync\|--controller local\|8xh100\|resources-8\|c0,c1,c2,c3\|T4R2S2\|T4R4S0" $T/dry.$c && bad "$c carries 8-card/no-sync tokens"
done; ok "r6 r7 r5 r5c s0: 4xL40S strict runner, resources-4, cells c0,c1,c2, no 8-card / no-sync tokens"
grep -q 'triggers=\[\["train",1,"up1",{"target":"T1R3S0","expected_config_epoch":0,"deadline_s":600}\]\]' $T/dry.r6 && grep -q -- "--rl-test-kill-learner-at QUIESCING" $T/dry.r6 && grep -q "judge: r6 --rc-file $T/rr/infra-v2-test4-r6/rc.txt" $T/dry.r6 && ! grep -q "trainer-world" $T/dry.r6 && ok "r6: up1@train rid1 -> T1R3S0, kill at QUIESCING, judge r6 with --rc-file only (criteria topology-free)" || bad "r6: $(cat $T/dry.r6)"
grep -q 'arms: dctl.py|kill_after_tx|up1|dn1|{"target":"T1R1S2","expected_config_epoch":1,"deadline_s":600}' $T/dry.r7 && grep -q "judge: r7 --trainer-world 1 --committed-members 3 --rc-file $T/rr/infra-v2-test4-r7/rc.txt" $T/dry.r7 && grep -q "tprobe: 1 rec" $T/dry.r7 && ok "r7: kill_after_tx dn1 -> T1R1S2, judge world 1 / members 3 / rc-file, term probe rec" || bad "r7: $(cat $T/dry.r7)"
grep -q -- "--rl-test-kill-learner-at COMMITTED" $T/dry.r5 && grep -q "judge: r5 --trainer-world 1 --committed-members 3 --rc-file" $T/dry.r5 && grep -q "arms: none" $T/dry.r5 && grep -q "^gate: 1" $T/dry.r5 && ok "r5: kill at COMMITTED, judge world 1 / members 3 / rc-file, gated" || bad "r5: $(cat $T/dry.r5)"
grep -q "judge: s0 --epochs" $T/dry.s0 && ok "s0: own judge args kept" || bad "s0 judge"
SHA=x DRY=1 bash $B/a4go4_strict.sh r5 pfx 1 1 > $T/dry.def 2>&1; grep -q "attestation-4-6.json" $T/dry.def && ok "default attestation = cfg/attestation-4-6.json (ATTEST overrides it)" || bad "default attestation: $(head -1 $T/dry.def)"
SHA=x DRY=1 ATTEST=$T/none.json bash $B/a4go4_strict.sh r5 pfx 1 1 >/dev/null 2>&1; [ $? = 6 ] && ok "ATTEST missing -> rc 6" || bad "ATTEST missing rc"
# --- dispatch: non-strict cases exec to a4go4.sh (A8GO_INNER stub), with the same argv
printf '#!/bin/bash\necho "inner $*" >> $STUBLOG; exit 0\n' > $T/inner; chmod +x $T/inner; : > $T/log
STUBLOG=$T/log A8GO_INNER=$T/inner SHA=x DRY=1 bash $B/a4go4_strict.sh d4 pfx 1800 1920 >/dev/null 2>&1; rc=$?
[ $rc = 0 ] && grep -q "inner d4 pfx 1800 1920" $T/log && ok "dispatch: d4 -> a4go4.sh (same argv)" || bad "dispatch d4 rc=$rc $(cat $T/log)"
grep -q 'A8GO_INNER:-/home/michael/work/gpu-b1-runs/a4go4.sh' $B/a4go4_strict.sh && ok "dispatch default = a4go4.sh (not a8go.sh)" || bad "dispatch default"
printf '#!/bin/bash\necho "inner-att ${ATTEST:-unset}" >> $STUBLOG; exit 0\n' > $T/inner2; chmod +x $T/inner2; : > $T/log
STUBLOG=$T/log A8GO_INNER=$T/inner2 SHA=x DRY=1 ATTEST=$T/strict.json ATTEST_NOSYNC=$T/nosync.json bash $B/a4go4_strict.sh d4 pfx 1 1 >/dev/null 2>&1
grep -q "inner-att $T/nosync.json" $T/log && ok "dispatch: a4go4.sh gets ATTEST=ATTEST_NOSYNC (d4 4-round fingerprint), not the strict one" || bad "dispatch attest: $(cat $T/log)"
: > $T/log; STUBLOG=$T/log A8GO_INNER=$T/inner2 SHA=x DRY=1 ATTEST=$T/strict.json bash $B/a4go4_strict.sh d4 pfx 1 1 >/dev/null 2>&1
grep -q "inner-att unset" $T/log && ok "dispatch: without ATTEST_NOSYNC the strict ATTEST does not leak (a4go4.sh default applies)" || bad "dispatch attest leak: $(cat $T/log)"
# --- a4go4.sh: DRY d4 + chain protocol
SHA=deadbee DRY=1 ATTEST=$T/att.json RUN_ROOT=$T/rr bash $B/a4go4.sh d4 infra-v2-test4-d4 1800 1920 > $T/dry.d4 2>&1 || bad "a4go4 d4 dry rc"
grep -q "GPU_SPEC=nebius:4xl40s@eu-north1 n2run.sh infra-v2-test4-d4 4 1800 1920 --total-steps 4 .*--rl-elastic-cells c0,c1,c2 .*--rl-elastic-attestation $T/att.json --rl-test-inject-stop-failures 100000 --rl-elastic-recovery-timeout-s 120\$" $T/dry.d4 && grep -q '"up1",{"target":"T1R3S0","expected_config_epoch":0,"deadline_s":600}\],\["train",1,"dn1",{"target":"T1R1S2","expected_config_epoch":1,"deadline_s":600}' $T/dry.d4 && grep -q "^judge: d4$" $T/dry.d4 && grep -q "^hook: 2 30$" $T/dry.d4 && ok "a4go4 d4: no-sync 4-card launch, stop-failure injection + recovery timeout 120, up1@rid0/dn1@rid1, judge d4, hook 2 30" || bad "a4go4 d4: $(cat $T/dry.d4)"
SHA= DRY=1 bash $B/a4go4.sh d4 p 1 1 >/dev/null 2>&1; [ $? != 0 ] && ok "a4go4: SHA required" || bad "a4go4 SHA default"
grep -q 'R=${RUN_ROOT:-$B}/$P' $B/a4go4.sh && grep -q '> $R.n2run.out 2>&1 &' $B/a4go4.sh && grep -q '${NSTOP:-$B/nstop.sh} $P > $R/final_stop.txt' $B/a4go4.sh && grep -q 'touch $R/item_done' $B/a4go4.sh && grep -q 'ATT=${ATTEST:-$B/cfg/attestation-4-$ATTN.json}' $B/a4go4.sh && grep -q 'diag_pull.sh $R' $B/a4go4.sh && grep -q 'probe_after_term.sh $R $P $HOOK' $B/a4go4.sh && ok "a4go4 chain protocol: RUN_ROOT, <run>.n2run.out, NSTOP, item_done after judge, ATTEST, diag_pull, after-term hook" || bad "a4go4 protocol"
# the final-guard line, executed with stubs: writes cleanup_rc, runs judge_after (stub), touches item_done
R=$T/rr/g; mkdir -p $R; echo rc=0 > $R/rc.txt; printf '#!/bin/bash\necho "stop $*"; exit 0\n' > $T/nstop; printf '#!/bin/bash\necho "judge $*" > $STUBLOG; exit 0\n' > $T/judge; chmod +x $T/nstop $T/judge
line=$(grep -F 'touch $R/item_done' $B/a4go4.sh | sed -e 's/^setsid nohup //' -e 's/ > \/dev\/null 2>&1 &$//' -e 's/sleep 20;/sleep 0;/' -e "s#\$B/judge_after.sh#$T/judge#")
(export R B=$B NSTOP=$T/nstop JUDGE=d4 P=g STUBLOG=$T/jlog; eval "$line")
[ "$(cat $R/cleanup_rc.txt 2>/dev/null)" = 0 ] && [ -f $R/item_done ] && grep -q "judge $R d4" $T/jlog && grep -q "stop g" $R/final_stop.txt && ok "a4go4 final guard: NSTOP(prefix) -> cleanup_rc -> judge_after(d4) -> item_done" || bad "a4go4 final guard: $(ls $R) $(cat $T/jlog 2>&1)"
# --- chain8-retry overrides
for v in 'A8GO=${A8GO:-$B/a8go_strict.sh}' 'RESET=${RESET:-$B/reset_island_strict.sh}' 'CAP_USD=${CAP_USD:-700}' 'PRICE_PER_MIN=${PRICE_PER_MIN:-0.5133}' 'THREAD_MAX=${THREAD_MAX:-3600}'; do grep -qF -- "$v" $B/chain8-retry.sh || bad "chain8-retry lacks $v"; done; ok "chain8-retry.sh: A8GO RESET CAP_USD PRICE_PER_MIN THREAD_MAX overridable by env"
grep -q 'A8GO=$B/a8go_strict.sh RESET\|CAP_USD=700 THREAD' $B/chain8-retry.sh && bad "chain8-retry still hard-codes A8GO/CAP" || ok "chain8-retry.sh: no hard-coded A8GO/CAP_USD left"
# --- mkatt4 / fingerprints at 5cf4d902 (CPU recomputation; the GPU selfcheck compares the attestation with the island's rl_driver_start)
grep -q '${FP:-fp_local4.py}' $B/mkatt4.sh && grep -q '${REPO:-' $B/mkatt4.sh && ok "mkatt4.sh: FP= (strict variant) and REPO= (git archive tree) selectable" || bad "mkatt4 env"
[ "$(diff $B/fp_local4.py $B/fp_local4_strict.py | grep -c '^[<>]')" = 4 ] && ! sed -e 1d -e "s/#.*//" $B/fp_local4_strict.py | grep -q 'rl-single-island-no-sync\|"--controller", "local"' && ok "fp_local4_strict.py = fp_local4.py minus --rl-single-island-no-sync/--controller local" || bad "fp_local4_strict diff"
for n in 6-0e7a78bb:0e7a78bb37cc884056ac94eda903f41ed8edbc745669b5cc21a841aa10d5714a 4-6e13c79d:6e13c79dc4d1f671badb89f7eaf04794def3138b9f7100ff101d4004e47b159b; do f=$CFG/attestation-4-${n%%:*}.json
  python3 -c "
import json,sys; d=json.load(open('$f')); e=d['certified_edges']
assert d['runtime_fingerprint']=='sha256:${n#*:}', d['runtime_fingerprint']
assert d['execution_modes']==['partitioned-serial'] and {(x['source'],x['target']) for x in e}=={('T1R1S2','T1R3S0'),('T1R3S0','T1R1S2')} and all(x['kind']=='rollout-only' for x in e)" 2>/dev/null && ok "cfg/attestation-4-${n%%:*}.json: fingerprint + T1R1S2<->T1R3S0 rollout-only edges" || bad "attestation $f"; done
grep -qF -- '--out $R/judgment.json --marker-dir $R "$@"' $B/judge_after.sh && ok "judge_after.sh passes the case's extra judge args (topology / rc-file) through" || bad "judge_after pass-through"
[ $fail = 0 ] && echo "ALL PASS" || { echo "SOME FAILED"; exit 1; }

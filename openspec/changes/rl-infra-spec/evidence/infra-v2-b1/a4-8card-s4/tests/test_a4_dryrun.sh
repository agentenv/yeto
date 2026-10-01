#!/bin/bash
# CPU-only DRY=1 checks of the a8go.sh A4b family (a4b / a4bu / a4bc): the launch switches, triggers, judge and hook of each case;
# a4bc = a4b + --rl-test-tool-side-effect-log (3.3 X5 (b)); nstop/nstop_item pull the side-effect journal; judge_after passes --side-effects.
# Never starts anything (a8go.sh DRY=1 exits before n2run).  run: bash tests/test_a4_dryrun.sh
set -u
B=$(cd "$(dirname "$0")/.." && pwd); [ -f $B/a8go.sh ] || B=$B/scripts; T=$(mktemp -d); trap 'rm -rf $T' EXIT; fail=0
ok() { echo "PASS $1"; }; bad() { echo "FAIL $1"; fail=1; }
for f in a8go.sh nstop.sh nstop_item.sh judge_after.sh; do bash -n $B/$f && ok "syntax $f" || bad "syntax $f"; done
for c in a4b a4bu a4bc; do SHA=deadbee DRY=1 bash $B/a8go.sh $c infra-v2-test-$c 1500 1600 > $T/dry.$c 2>&1 || bad "dry $c rc"; grep -q triggers-json-ok $T/dry.$c || bad "$c triggers json"; done
BASE="--rl-elastic-tool-wait-board --rl-elastic-drain-timeout-s 5 --rl-test-inject-tool-wait-s 30"
grep -q -- "--total-steps 6 .* $BASE\$" $T/dry.a4b && ok "a4b: tool-wait board + drain timeout 5 + inject 30 s, nothing else" || bad "a4b switches: $(head -1 $T/dry.a4b)"
grep -q -- " $BASE --rl-test-inject-undrain-fail 1\$" $T/dry.a4bu && ok "a4bu: a4b + undrain fail" || bad "a4bu switches"
grep -q -- " $BASE --rl-test-tool-side-effect-log\$" $T/dry.a4bc && ok "a4bc: a4b + --rl-test-tool-side-effect-log (no undrain fail)" || bad "a4bc switches: $(head -1 $T/dry.a4bc)"
for c in a4b a4bu a4bc; do diff <(grep ^triggers= $T/dry.a4b) <(grep ^triggers= $T/dry.$c) > /dev/null || bad "$c triggers differ from a4b"; grep -q '"up1"' $T/dry.$c && grep -q '"dn1".*"T4R2S2".*"expected_config_epoch":1' $T/dry.$c || bad "$c triggers"; done; ok "a4b/a4bu/a4bc share the triggers: up1 @train rid0, dn1 @generate rid2 (epoch 1)"
grep -q "attestation-8-6.json" $T/dry.a4bc && ok "a4bc: attestation-8-6 (6 rounds)" || bad "a4bc attestation"
# case table: judge + hook (the DRY output does not print them; read the script)
for c in a4b a4bu a4bc; do grep -A1 "^  $c)" $B/a8go.sh | grep -q "JUDGE=\"$c\"; HOOK=\"2 0\";;" && ok "$c: judge $c, term probe hook 2 0" || bad "$c judge/hook line"; done
# pull list + judge wiring
for f in nstop.sh nstop_item.sh; do grep -q 'yeto-rl/elastic-state/side_effects.jsonl:side_effects.jsonl' $B/$f && ok "$f pulls elastic-state/side_effects.jsonl" || bad "$f pull list"; done
grep -q 'SE=\$R/.j/elastic-state/side_effects.jsonl; \[ -s \$SE \] || SE=\$R/pulled/side_effects.jsonl' $B/judge_after.sh && grep -q -- '--side-effects \$SE' $B/judge_after.sh && ok "judge_after: --side-effects from the elastic-state tarball, else the direct pull" || bad "judge_after side-effects wiring"
grep -q 'elif a.case == "a4bc": res = judge_a4bc' $B/judge_inject.py && ok "judge_inject: a4bc case" || bad "judge_inject a4bc"
SHA=x DRY=1 bash $B/a8go.sh a4bx p 1 1 >/dev/null 2>&1; [ $? = 64 ] && ok "unknown case rc 64" || bad "unknown case rc"
[ $fail = 0 ] && echo ALL-PASS || { echo SOME-FAIL; exit 1; }

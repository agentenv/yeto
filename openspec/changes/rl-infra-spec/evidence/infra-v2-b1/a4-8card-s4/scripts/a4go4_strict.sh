#!/bin/bash
# usage: a4go4_strict.sh <case> <prefix> <hard_s> <wd_s>    4x L40S on Nebius eu-north1, T1R1S2 <-> T1R3S0 (trainer G0, rollout c0 on G1, standby G2/G3 = c1/c2), STRICT-AVG
# single island with the head syncer on THIS host.  = a8go_strict.sh parameterised for 4 cards (2026-10-03, chain 8 IV; PARALLEL-PLAN-S7 / T1-S7-PROGRESS 5.1):
# NG=4, GPU_SPEC nebius:4xl40s@eu-north1, EXPECT L40S x4, cfg/resources-4.json, cells c0,c1,c2, --rl-rollout-gpus 1 --rl-standby-gpus 2, up = T1R1S2->T1R3S0 (starts c1,c2), GPU_SPEC / EXPECT_GPU_NAME overridable (cloud fallback, e.g. GPU_SPEC=aws:4xl4@us-east-1 EXPECT_GPU_NAME=L4; fingerprint/attestation/resources-4 are GPU-model independent).
# down = T1R3S0->T1R1S2, r7 kill_after_tx dn1 target T1R1S2, attestation-4-<N>[-<fp>] (fp_local4_strict.py / mkatt4.sh FP=fp_local4_strict.py), non-strict cases -> a4go4.sh.
# Criteria conversion (gpu-plan-v2 9.22 rule: only card numbers and member counts change, the decision logic does not): judge r5/r7 get --trainer-world 1
# --committed-members 3 (trainer is one rank; the committed set after up1 is c0,c1,c2; the recovery restarts c1,c2); every judged case gets --rc-file (rc=124 -> INVALID_TEST(hard_timeout)).
# env: ATTEST = strict attestation (6 rounds; chain 8 IV cfg/attestation-4-6-0e7a78bb.json), ATTEST_NOSYNC = the a4go4.sh cases' attestation (d4: 4 rounds, cfg/attestation-4-4-6e13c79d.json).
# ---- original a8go_strict.sh header follows ----
# cases: chk | s0 | r6 | r7 | r5 | r5c      (DRY=1: print the launch args + triggers + arms, start nothing)
# GATE (user ruling 2026-10-02: judged cases stop the chain on a non-PASS): strict cases s0/r6/r7/r5/r5c write <chain>/ABORT (reason gate_<case>_<verdict>)
# in their final guard BEFORE item_done; for the a8go.sh cases (d2, a4bc; d4 = chain tail, no gate) the same rule is applied by reset_island_strict.sh, which
# chain8 runs after item_done and before the next item: it reads the previous item's judgment.json (GATED_CASES) and refuses the reset (chain stops + releases).
# CHAIN DISPATCH (CHAIN8-PLAN.md, one A8GO per chain): every case NOT listed here (d2 a4bc d4 e1b wd ... = a8go.sh's) is exec'ed to a8go.sh unchanged, so a
# single chain8 run mixes no-sync and strict items: `A8GO=$B/a8go_strict.sh RESET=$B/reset_island_strict.sh chain8.sh ... chk:900 s0:2400 d2:2100 ...`.
# chk = chain-head self-check (chk_launch.py): provisions the SAME island cluster (make_miles_island_task resources/setup) and runs the image checks instead of
# the learner (fork pin + workers_lost, Megatron hc_head_contraction, LoRA module import); fail -> <chain>/ABORT (chain8 stops + releases); ok -> rc 0, cluster kept.
# E1-D ⑤⑥⑦ with the 3.7 restart recovery (recovery-design.md §10, E1D-RECOVERY-PROGRESS "方案 A"): the learner is killed and restarted IN PLACE by the island's
# restart loop; the restart point is the syncer's current policy version (StrictAvgSync.start -> ledger.rebase(v)), so there is no --rl-single-island-no-sync here
# (n2run_strict.sh; a no-sync island fails at rebase(0)).  Same COMMON as a8go.sh (fingerprint 172652ea... unchanged -> cfg/attestation-8-6.json, re-checked with
# fp_local8_strict.py on infra-e1-recovery 2de3ac1 and gpu-b1 3673da9, also with --rl-elastic-restart-attempts/--rl-elastic-max-recovery-attempts/--rl-test-kill-learner-at).
# Existing a8go.sh cases are untouched: chain8.sh selects this script with A8GO=.../a8go_strict.sh (and RESET=.../reset_island_strict.sh).
# Helpers as in a8go.sh: n2inwatch (triggers), selfcheck (8xH100 + fingerprint), diag_pull, arms (dctl.py), in-container term_probe (TPROBE), final guard
# (nstop_item_strict.sh -> kills the host syncer tree, then NSTOP/nstop_item.sh; judge_after.sh <case>).
# s0 (a4s7-20261001-1) lesson: strict-avg audits every pause against min(pause_margin x syncer --quorum-timeout-s, idle-flow) = 0.5 x 900 = 450 s by default;
# a 600 s deadline is "pause not allowed: expected pause 600s exceeds budget 450s" (inbox status, no journal phase) -- no-sync has no outer budget, so a8go.sh
# never hit it.  STRICT_EX below raises the syncer quorum timeout to 1800 s (budget 900 s >= the 600 s deadlines; also covers a ~300 s in-place learner restart
# without the syncer giving up on the learner).  Not a Miles argv -> fingerprint unchanged (fp_local8_strict.py re-checked).
# >=1 PUSH before every kill (restart point v>=1): r6/r5 up1@train rid1 (the up runs before generate 2: versions 0,1 already pushed); r7 kill after up1 SUCCEEDED
# and the next generate; r5c dn1@train rid2 (killed at its COMMITTED).  Round ~15 s (8 cards, 0.6B), strict PUSH at every round boundary.
C=$1; P=$2; HARD=$3; WD=$4
case $C in chk|s0|r6|r7|r5|r5c) ;; *) exec env ATTEST="${ATTEST_NOSYNC:-}" ${A8GO_INNER:-/home/michael/work/gpu-b1-runs/a4go4.sh} "$@";; esac   # non-strict cases (d4 ...): a4go4.sh (chain protocol: RUN_ROOT/NSTOP/item_done); its attestation = ATTEST_NOSYNC (4-step no-sync fingerprint differs from the 6-step strict one), never the strict ATTEST
SHA=${SHA:?set SHA to the frozen code commit of this batch (must contain the 3.7 restart recovery, infra-e1-recovery >= 0e68962)}; B=/home/michael/work/gpu-b1-runs; R=${RUN_ROOT:-$B}/$P
UP=${UP_DEADLINE_S:-600}; EX=""; STEPS=6; ATTN=6; JUDGE=""; JARGS=""; GATE=0; ARMS=(); TPROBE=""
EXR="--rl-elastic-restart-attempts 2 --rl-elastic-max-recovery-attempts 3"
STRICT_EX="--rl-elastic-quorum-timeout-s ${QUORUM_TIMEOUT_S:-1800}"   # pause budget = 0.5 x this must be >= every request deadline_s (600)
req() { printf '["%s",%s,"%s",{"target":"%s","expected_config_epoch":%s,"deadline_s":%s}]' "$1" "$2" "$3" "$4" "$5" "$6"; }   # phase rid id target epoch deadline
UPB() { req train $1 $2 T1R3S0 0 ${3:-$UP}; }; DNB() { req train $1 $2 T1R1S2 1 ${3:-600}; }
NG=4; GPU_SPEC=${GPU_SPEC:-nebius:4xl40s@eu-north1}; EXPECT_GPU_NAME=${EXPECT_GPU_NAME:-L40S}; TOPO_JARGS="--trainer-world 1 --committed-members 3"; RCJ="--rc-file $R/rc.txt"
if [ "$C" = chk ]; then
  [ -f $B/chk_launch.py ] || { echo "missing $B/chk_launch.py"; exit 6; }
  if [ "${DRY:-0}" = 1 ]; then echo "SHA=$SHA GPU_SPEC=$GPU_SPEC chk_launch.sh $P $NG $HARD (no training, no attestation; checks: fork pin+workers_lost, hc_head_contraction, lora import; fail -> ABORT)"; echo "triggers=[]"; echo triggers-json-ok; echo "arms: none"; echo "tprobe: none"; echo "judge: chk_launch job status"; exit 0; fi
  GPU_SPEC=$GPU_SPEC setsid nohup $B/chk_launch.sh $P $NG $HARD > $R.n2run.out 2>&1 &
  echo started $P chk; exit 0
fi
case $C in
  s0)   TRIG="[$(UPB 1 up1 600),$(DNB 3 dn1 600)]"; JUDGE="s0"; GATE=1   # smoke: strict island + --rl-elastic, no kill; judge_s0 (criteria fixed in E1D-RECOVERY-PROGRESS "s0 结果"); GATE: non-PASS -> <chain>/ABORT (chain8 does not stop on a judge FAIL by itself)
        JARGS="--epochs $R/.j/elastic-state/reconfig/epochs.json --inbox-dir $R/.j/elastic-state/inbox --syncer-log $R/pulled/yeto-syncer.log --rc-file $R/rc.txt --syncer-clean $R/syncer_clean.txt";;
  r6)   EX="$EXR --rl-test-kill-learner-at QUIESCING"; TRIG="[$(UPB 1 up1 600)]"; JUDGE="r6"; JARGS="$RCJ"; GATE=1;;                       # ⑥ up1 killed at QUIESCING -> restart -> CANCELLED, no recovery record, rounds go on
  r7)   EX="$EXR"; TRIG="[$(UPB 0 up1 600)]"; JUDGE="r7"; JARGS="$TOPO_JARGS $RCJ"; TPROBE="1 rec"; GATE=1                                               # ⑦ up1 SUCCEEDED, then kill in steady state (fork epoch 0 vs journal 1) -> restart -> recovery verified -> dn1 SUCCEEDED
        ARMS+=("dctl.py|kill_after_tx|up1|dn1|{\"target\":\"T1R1S2\",\"expected_config_epoch\":1,\"deadline_s\":600}");;
  r5)   EX="$EXR --rl-test-kill-learner-at COMMITTED"; TRIG="[$(UPB 1 up1 600)]"; JUDGE="r5"; JARGS="$TOPO_JARGS $RCJ"; TPROBE="1 rec"; GATE=1;;        # ⑤ up1 killed at COMMITTED -> restart -> recovery (c2,c3 restarted) verified -> up1 SUCCEEDED(recovered_after_restart)
  r5c)  EX="$EXR --rl-test-kill-learner-at COMMITTED"; TRIG="[$(UPB 0 up1 600),$(DNB 2 dn1 600)]"; JUDGE="r5c"; JARGS="$TOPO_JARGS $RCJ"; GATE=1       # ⑤c ruling (c) regression: up1 ok (kill suppressed by the marker), dn1 killed at COMMITTED -> startup shape, no recovery record
        ARMS+=("dctl.py|marker|up1");;
  *) echo "unknown case $C (a4go4_strict.sh: s0 r6 r7 r5 r5c)"; exit 64;;
esac
ATT=${ATTEST:-$B/cfg/attestation-4-$ATTN.json}; [ -f $ATT ] || { echo "missing $ATT (mkatt4.sh)"; exit 6; }   # ATTEST: attestation file for another code SHA (chain 8 IV: cfg/attestation-4-6-0e7a78bb.json at 5cf4d902)
COMMON="--total-steps $STEPS --rl-placement fixed-partition --rl-rollout-gpus 1 --rl-standby-gpus 2 --rl-elastic --rl-elastic-declare-cells --rl-elastic-cells c0,c1,c2 --rl-elastic-resources $B/cfg/resources-4.json --rl-elastic-initial-config T1R1S2 --rl-observe-timeline --rl-elastic-attestation $ATT"
if [ "${DRY:-0}" = 1 ]; then echo "SHA=$SHA GPU_SPEC=$GPU_SPEC SYNCER_PUBLIC_IP=${SYNCER_PUBLIC_IP:-185.189.44.160}:${SYNCER_PORT:-29400} n2run_strict.sh $P $NG $HARD $WD $COMMON $STRICT_EX $EX"; echo "triggers=$TRIG"; python3 -c "import json,sys;json.loads(sys.argv[1])" "$TRIG" && echo triggers-json-ok; printf 'arms: %s\n' "${ARMS[@]:-none}"; echo "tprobe: ${TPROBE:-none}"; echo "judge: ${JUDGE:-manual}${JARGS:+ $JARGS}"; echo "gate: $GATE"; exit 0; fi
SHA=$SHA GPU_SPEC=$GPU_SPEC setsid nohup $B/n2run_strict.sh $P $NG $HARD $WD $COMMON $STRICT_EX $EX > $R.n2run.out 2>&1 &
sleep 10
setsid nohup $B/n2inwatch.sh $R "$TRIG" > /dev/null 2>&1 &
EXPECT_GPU_NAME=$EXPECT_GPU_NAME EXPECT_GPU_N=4 NSTOP=$B/nstop_item_strict.sh NSTOP_INNER=${NSTOP:-$B/nstop.sh} setsid nohup $B/selfcheck.sh $R $P $ATT > /dev/null 2>&1 &
setsid nohup $B/diag_pull.sh $R > /dev/null 2>&1 &
for a in "${ARMS[@]}"; do IFS='|' read -ra parts <<< "$a"; setsid nohup $B/n2arm.sh $R "${parts[@]}" > /dev/null 2>&1 & done   # "script|arg1|arg2" (args must not contain |)
# in-container terminal probe (mode rec: fires on the journal's `recovery verified`, ~2 s + ~20 s Ray client; the run then still lives >= 3 rounds ~45 s; main evidence = files)
if [ -n "$TPROBE" ]; then set -- $TPROBE; setsid nohup bash -c "until [ -s $R/pulled/gpu.txt ]; do [ -f $R/rc.txt ] && exit 1; sleep 10; done; CL=\$(cat $R/cluster.txt); for f in fork_probe.py probe_remote.sh; do b=\$(base64 -w0 $B/\$f); HOME=/home/michael timeout 60 ssh -o StrictHostKeyChecking=no \$CL \"mkdir -p ~/yeto-rl && echo \$b | base64 -d > ~/yeto-rl/\$f\"; done; $B/n2arm.sh $R term_probe.py $1 ${2:-}" > $R.termprobe.out 2>&1 & fi
# final guard: when the launcher ended (or STOP), kill the host syncer + pull (chain: nstop_item.sh, never releases; single run: nstop.sh -> cleanup_run.sh), then judge
setsid nohup bash -c "until [ -f $R/rc.txt ] || [ -f $R/startup_failed ]; do sleep 15; done; sleep 20; NSTOP_INNER=${NSTOP:-$B/nstop.sh} $B/nstop_item_strict.sh $P > $R/final_stop.txt 2>&1; echo \$? > $R/cleanup_rc.txt; [ -n '$JUDGE' ] && $B/judge_after.sh $R $JUDGE $JARGS > $R/judge.out 2>&1; if [ '$GATE' = 1 ] && [ -n '${CHAIN_DIR:-}' ]; then v=\$(python3 -c \"import json;print(json.load(open('$R/judgment.json')).get('verdict'))\" 2>/dev/null || echo NO_JUDGMENT); [ \"\$v\" = PASS ] || echo \"{\\\"item\\\":\\\"$P\\\",\\\"reason\\\":\\\"gate_${C}_\$v\\\",\\\"ts\\\":\\\"\$(date -u +%FT%TZ)\\\"}\" > ${CHAIN_DIR:-/dev/null}/ABORT; fi; touch $R/item_done" > /dev/null 2>&1 &
echo started $P $C strict-4card

#!/bin/bash
# usage: a8go_strict.sh <case> <prefix> <hard_s> <wd_s>     8x H100 on Nebius eu-north1, T4R2S2 <-> T4R4S0, STRICT-AVG single island with the head syncer on THIS host
# cases: s0 | r6 | r7 | r5 | r5c      (DRY=1: print the launch args + triggers + arms, start nothing)
# E1-D ⑤⑥⑦ with the 3.7 restart recovery (recovery-design.md §10, E1D-RECOVERY-PROGRESS "方案 A"): the learner is killed and restarted IN PLACE by the island's
# restart loop; the restart point is the syncer's current policy version (StrictAvgSync.start -> ledger.rebase(v)), so there is no --rl-single-island-no-sync here
# (n2run_strict.sh; a no-sync island fails at rebase(0)).  Same COMMON as a8go.sh (fingerprint 172652ea... unchanged -> cfg/attestation-8-6.json, re-checked with
# fp_local8_strict.py on infra-e1-recovery 2de3ac1 and gpu-b1 3673da9, also with --rl-elastic-restart-attempts/--rl-elastic-max-recovery-attempts/--rl-test-kill-learner-at).
# Existing a8go.sh cases are untouched: chain8.sh selects this script with A8GO=.../a8go_strict.sh (and RESET=.../reset_island_strict.sh).
# Helpers as in a8go.sh: n2inwatch (triggers), selfcheck (8xH100 + fingerprint), diag_pull, arms (dctl.py), in-container term_probe (TPROBE), final guard
# (nstop_item_strict.sh -> kills the host syncer tree, then NSTOP/nstop_item.sh; judge_after.sh <case>).
# >=1 PUSH before every kill (restart point v>=1): r6/r5 up1@train rid1 (the up runs before generate 2: versions 0,1 already pushed); r7 kill after up1 SUCCEEDED
# and the next generate; r5c dn1@train rid2 (killed at its COMMITTED).  Round ~15 s (8 cards, 0.6B), strict PUSH at every round boundary.
C=$1; P=$2; HARD=$3; WD=$4; SHA=${SHA:?set SHA to the frozen code commit of this batch (must contain the 3.7 restart recovery, infra-e1-recovery >= 0e68962)}; B=/home/michael/work/gpu-b1-runs; R=${RUN_ROOT:-$B}/$P
UP=${UP_DEADLINE_S:-600}; EX=""; STEPS=6; ATTN=6; JUDGE=""; ARMS=(); TPROBE=""
EXR="--rl-elastic-restart-attempts 2 --rl-elastic-max-recovery-attempts 3"
req() { printf '["%s",%s,"%s",{"target":"%s","expected_config_epoch":%s,"deadline_s":%s}]' "$1" "$2" "$3" "$4" "$5" "$6"; }   # phase rid id target epoch deadline
UPB() { req train $1 $2 T4R4S0 0 ${3:-$UP}; }; DNB() { req train $1 $2 T4R2S2 1 ${3:-600}; }
case $C in
  s0)   TRIG="[$(UPB 1 up1 600),$(DNB 3 dn1 600)]";;   # smoke: strict island + --rl-elastic, no kill; judged by hand (progress file item 5: syncer line in launch.log, 6 rounds, up1/dn1 SUCCEEDED, publication versions 0..6, one outer_recorded per rollout, no lora_unverifiable)
  r6)   EX="$EXR --rl-test-kill-learner-at QUIESCING"; TRIG="[$(UPB 1 up1 600)]"; JUDGE="r6";;                       # ⑥ up1 killed at QUIESCING -> restart -> CANCELLED, no recovery record, rounds go on
  r7)   EX="$EXR"; TRIG="[$(UPB 0 up1 600)]"; JUDGE="r7"; TPROBE="1 rec"                                               # ⑦ up1 SUCCEEDED, then kill in steady state (fork epoch 0 vs journal 1) -> restart -> recovery verified -> dn1 SUCCEEDED
        ARMS+=("dctl.py|kill_after_tx|up1|dn1|{\"target\":\"T4R2S2\",\"expected_config_epoch\":1,\"deadline_s\":600}");;
  r5)   EX="$EXR --rl-test-kill-learner-at COMMITTED"; TRIG="[$(UPB 1 up1 600)]"; JUDGE="r5"; TPROBE="1 rec";;        # ⑤ up1 killed at COMMITTED -> restart -> recovery (c2,c3 restarted) verified -> up1 SUCCEEDED(recovered_after_restart)
  r5c)  EX="$EXR --rl-test-kill-learner-at COMMITTED"; TRIG="[$(UPB 0 up1 600),$(DNB 2 dn1 600)]"; JUDGE="r5c"       # ⑤c ruling (c) regression: up1 ok (kill suppressed by the marker), dn1 killed at COMMITTED -> startup shape, no recovery record
        ARMS+=("dctl.py|marker|up1");;
  *) echo "unknown case $C (a8go_strict.sh: s0 r6 r7 r5 r5c)"; exit 64;;
esac
ATT=$B/cfg/attestation-8-$ATTN.json; [ -f $ATT ] || { echo "missing $ATT (mkatt8.sh)"; exit 6; }
COMMON="--total-steps $STEPS --rl-placement fixed-partition --rl-rollout-gpus 2 --rl-standby-gpus 2 --rl-elastic --rl-elastic-declare-cells --rl-elastic-cells c0,c1,c2,c3 --rl-elastic-resources $B/cfg/resources-8.json --rl-elastic-initial-config T4R2S2 --rl-observe-timeline --rl-elastic-attestation $ATT"
if [ "${DRY:-0}" = 1 ]; then echo "SHA=$SHA GPU_SPEC=nebius:8xh100@eu-north1 SYNCER_PUBLIC_IP=${SYNCER_PUBLIC_IP:-185.189.44.160}:${SYNCER_PORT:-29400} n2run_strict.sh $P 8 $HARD $WD $COMMON $EX"; echo "triggers=$TRIG"; python3 -c "import json,sys;json.loads(sys.argv[1])" "$TRIG" && echo triggers-json-ok; printf 'arms: %s\n' "${ARMS[@]:-none}"; echo "tprobe: ${TPROBE:-none}"; echo "judge: ${JUDGE:-manual}"; exit 0; fi
SHA=$SHA GPU_SPEC=nebius:8xh100@eu-north1 setsid nohup $B/n2run_strict.sh $P 8 $HARD $WD $COMMON $EX > $R.n2run.out 2>&1 &
sleep 10
setsid nohup $B/n2inwatch.sh $R "$TRIG" > /dev/null 2>&1 &
EXPECT_GPU_NAME=H100 EXPECT_GPU_N=8 NSTOP=$B/nstop_item_strict.sh NSTOP_INNER=${NSTOP:-$B/nstop.sh} setsid nohup $B/selfcheck.sh $R $P $ATT > /dev/null 2>&1 &
setsid nohup $B/diag_pull.sh $R > /dev/null 2>&1 &
for a in "${ARMS[@]}"; do IFS='|' read -ra parts <<< "$a"; setsid nohup $B/n2arm.sh $R "${parts[@]}" > /dev/null 2>&1 & done   # "script|arg1|arg2" (args must not contain |)
# in-container terminal probe (mode rec: fires on the journal's `recovery verified`, ~2 s + ~20 s Ray client; the run then still lives >= 3 rounds ~45 s; main evidence = files)
if [ -n "$TPROBE" ]; then set -- $TPROBE; setsid nohup bash -c "until [ -s $R/pulled/gpu.txt ]; do [ -f $R/rc.txt ] && exit 1; sleep 10; done; CL=\$(cat $R/cluster.txt); for f in fork_probe.py probe_remote.sh; do b=\$(base64 -w0 $B/\$f); HOME=/home/michael timeout 60 ssh -o StrictHostKeyChecking=no \$CL \"mkdir -p ~/yeto-rl && echo \$b | base64 -d > ~/yeto-rl/\$f\"; done; $B/n2arm.sh $R term_probe.py $1 ${2:-}" > $R.termprobe.out 2>&1 & fi
# final guard: when the launcher ended (or STOP), kill the host syncer + pull (chain: nstop_item.sh, never releases; single run: nstop.sh -> cleanup_run.sh), then judge
setsid nohup bash -c "until [ -f $R/rc.txt ] || [ -f $R/startup_failed ]; do sleep 15; done; sleep 20; NSTOP_INNER=${NSTOP:-$B/nstop.sh} $B/nstop_item_strict.sh $P > $R/final_stop.txt 2>&1; echo \$? > $R/cleanup_rc.txt; [ -n '$JUDGE' ] && $B/judge_after.sh $R $JUDGE > $R/judge.out 2>&1; touch $R/item_done" > /dev/null 2>&1 &
echo started $P $C strict

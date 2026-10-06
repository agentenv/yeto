#!/bin/bash
# usage: a4go4.sh <case> <prefix> <hard_s> <wd_s>        4x L40S on Nebius eu-north1, T1R1S2 <-> T1R3S0 (trainer G0, rollout c0 on G1, standby G2/G3 = c1/c2)
# cases: smoke | base | e1a | e1b | wd | a4b | d123 | d4 | d5 | d6 | d7      (DRY=1: print the launch args + triggers, start nothing)
# Starts n2run (launch, --no-island-relaunch --modal-retries 0) + n2inwatch (triggers + router sampler) + selfcheck (+GPU assert 4xL40S, markers) + case helpers + final guard (nstop -> cleanup_run.sh, judge).
# CHAIN PROTOCOL (2026-10-03, chain 8 IV on 4xL40S; same as a8go.sh): RUN_ROOT (item dir root), CLUSTER_PREFIX/KEEP/SHARED (n2run.sh cluster reuse), NSTOP (nstop_item.sh: pull,
# never release), CHAIN_DIR, ATTEST (attestation for another code SHA), <run>.n2run.out next to the run dir, item_done touched after the judge.  Single-run defaults unchanged.
# d4 = E1-D (4) stop half-failure -> one bounded REBUILD_OLD -> RECOVERY_REQUIRED (judge_d4 (i) revision, hook 2 30 + in-container term probe as in a8go.sh).
# Required env for some cases:  UP_DEADLINE_S (wd: measured, see gpu-plan 9.22 step 2) | (e1b/a4b pass --rl-test-hold-before-check-s ${HOLD_S:-10} / --rl-test-inject-tool-wait-s 30 directly)
# Request time: a request submitted at "train" of rollout k executes before generate k+1 (= before round k+2).
C=$1; P=$2; HARD=$3; WD=$4; SHA=${SHA:?set SHA to the frozen code commit of this batch}; B=/home/michael/work/gpu-b1-runs; R=${RUN_ROOT:-$B}/$P
UP=${UP_DEADLINE_S:-600}; EX=""; STEPS=4; ATTN=4; JUDGE=""; ARMS=(); HOOK=""
req() { printf '["%s",%s,"%s",{"target":"%s","expected_config_epoch":%s,"deadline_s":%s}]' "$1" "$2" "$3" "$4" "$5" "$6"; }   # phase rid id target epoch deadline
UPB() { req train $1 $2 T1R3S0 0 ${3:-$UP}; }; DNB() { req train $1 $2 T1R1S2 1 ${3:-600}; }
case $C in
  smoke) STEPS=3; ATTN=3; TRIG="[$(UPB 0 up1 900),$(DNB 1 dn1 900)]";;
  base)  STEPS=6; ATTN=6; TRIG="[]";;
  e1a)   STEPS=6; ATTN=6; TRIG="[$(UPB 1 up1 600),$(UPB 2 up1 600),$(DNB 3 dn1 600)]"; JUDGE="e1a_c --expect-members 1,1,3,3,1,1";;
  e1b)   EX="--rl-test-inject-lora-perturb 0.01 --rl-test-hold-before-check-s ${HOLD_S:-10}"; TRIG="[$(UPB 1 up1 600)]"; JUDGE="e1b";;
  wd)    [ -n "${UP_DEADLINE_S:-}" ] || { echo "wd needs UP_DEADLINE_S (>=1.5x measured start_cells + margin, written in gpu-plan 9.22 before the run)"; exit 5; }
         EX="--rl-test-inject-update-weights-block-s 600"; TRIG="[$(UPB 1 up1 $UP)]"; JUDGE="wd";;
  a4b)   EX="--rl-elastic-tool-wait-board --rl-elastic-drain-timeout-s 5 --rl-test-inject-tool-wait-s 30"; ATTN=a4b
         TRIG="[$(UPB 0 up1 600),[\"generate\",2,\"dn1\",{\"target\":\"T1R1S2\",\"expected_config_epoch\":1,\"deadline_s\":600}]]"; JUDGE="a4b";;
  d123)  STEPS=5; ATTN=e1d5; EX="--rl-test-inject-stop-failures 1"
         # up1 ep0->1, dn1 ep1->2, up2 at ep2 (killed -> REBUILT_OLD, stays 2), up3 at ep2
         TRIG="[$(UPB 0 up1 600),$(DNB 1 dn1 600),$(req train 2 up2 T1R3S0 2 600),$(req train 3 up3 T1R3S0 2 600)]"
         ARMS+=("dkill.py|[{\"name\":\"d1\",\"tx\":\"up2\",\"when\":{\"kind\":\"fork_op\",\"op\":\"start\",\"status\":\"issued\"},\"gpu\":2,\"mode\":\"when_proc_appears\",\"min_age_s\":10},{\"name\":\"d2\",\"tx\":\"up3\",\"when\":{\"kind\":\"phase\",\"phase\":\"VERIFYING\"},\"gpu\":2,\"mode\":\"immediate\"}]");;
  d4)    EX="--rl-test-inject-stop-failures 100000 --rl-elastic-recovery-timeout-s 120"; TRIG="[$(UPB 0 up1 600),$(DNB 1 dn1 600)]"; JUDGE="d4"; HOOK="2 30";;   # standby c1/c2 on G2/G3; deadline 600 + recovery 120
  d5)    EX="--rl-elastic-restart-attempts 1 --rl-test-kill-learner-at COMMITTED"; TRIG="[$(UPB 0 up1 600),$(DNB 1 dn1 600)]"; ARMS+=("dctl.py|marker|up1");;
  d6)    STEPS=3; ATTN=3; EX="--rl-elastic-restart-attempts 1 --rl-test-kill-learner-at QUIESCING"; TRIG="[$(UPB 0 up1 600)]";;
  d7)    STEPS=5; ATTN=e1d5; EX="--rl-elastic-restart-attempts 1"; TRIG="[]"; ARMS+=("dctl.py|kill_then_up|up1|{\"target\":\"T1R3S0\",\"expected_config_epoch\":0,\"deadline_s\":600}");;
  *) echo "unknown case $C"; exit 64;;
esac
ATT=${ATTEST:-$B/cfg/attestation-4-$ATTN.json}; [ -f $ATT ] || { echo "missing $ATT (mkatt4.sh)"; exit 6; }   # ATTEST: attestation file for another code SHA (chain 8 IV: cfg/attestation-4-4-6e13c79d.json at 5cf4d902)
COMMON="--total-steps $STEPS --rl-placement fixed-partition --rl-rollout-gpus 1 --rl-standby-gpus 2 --rl-elastic --rl-elastic-declare-cells --rl-elastic-cells c0,c1,c2 --rl-elastic-resources $B/cfg/resources-4.json --rl-elastic-initial-config T1R1S2 --rl-observe-timeline --rl-elastic-attestation $ATT"
if [ "${DRY:-0}" = 1 ]; then echo "SHA=$SHA GPU_SPEC=nebius:4xl40s@eu-north1 n2run.sh $P 4 $HARD $WD $COMMON $EX"; echo "triggers=$TRIG"; python3 -c "import json,sys;json.loads(sys.argv[1])" "$TRIG" && echo triggers-json-ok; printf 'arms: %s\n' "${ARMS[@]:-none}"; echo "judge: ${JUDGE:-manual}"; echo "hook: ${HOOK:-none}"; exit 0; fi
SHA=$SHA GPU_SPEC=nebius:4xl40s@eu-north1 setsid nohup $B/n2run.sh $P 4 $HARD $WD $COMMON $EX > $R.n2run.out 2>&1 &
sleep 10
setsid nohup $B/n2inwatch.sh $R "$TRIG" > /dev/null 2>&1 &
EXPECT_GPU_NAME=L40S EXPECT_GPU_N=4 setsid nohup $B/selfcheck.sh $R $P $ATT > /dev/null 2>&1 &
setsid nohup $B/diag_pull.sh $R > /dev/null 2>&1 &
for a in "${ARMS[@]}"; do IFS='|' read -ra parts <<< "$a"; setsid nohup $B/n2arm.sh $R "${parts[@]}" > /dev/null 2>&1 & done   # "script|arg1|arg2" (args must not contain |)
[ -n "$HOOK" ] && setsid nohup $B/probe_after_term.sh $R $P $HOOK > $R.hook.out 2>&1 &
# in-container terminal-state probe (seconds after the terminal phase; the host-side hook above is only the fallback):
if [ -n "$HOOK" ]; then set -- $HOOK; setsid nohup bash -c "until [ -s $R/pulled/gpu.txt ]; do [ -f $R/rc.txt ] && exit 1; sleep 10; done; CL=\$(cat $R/cluster.txt); for f in fork_probe.py probe_remote.sh; do b=\$(base64 -w0 $B/\$f); HOME=/home/michael timeout 60 ssh -o StrictHostKeyChecking=no \$CL \"mkdir -p ~/yeto-rl && echo \$b | base64 -d > ~/yeto-rl/\$f\"; done; $B/n2arm.sh $R term_probe.py $1 ${3:-}" > $R.termprobe.out 2>&1 & fi
# final guard: when the launcher ended (or STOP), pull + cleanup (idempotent; chain: NSTOP=nstop_item.sh never releases), then judge from the pulled journal/tape, then item_done
setsid nohup bash -c "until [ -f $R/rc.txt ] || [ -f $R/startup_failed ]; do sleep 15; done; sleep 20; ${NSTOP:-$B/nstop.sh} $P > $R/final_stop.txt 2>&1; echo \$? > $R/cleanup_rc.txt; [ -n '$JUDGE' ] && $B/judge_after.sh $R $JUDGE > $R/judge.out 2>&1; touch $R/item_done" > /dev/null 2>&1 &
echo started $P $C

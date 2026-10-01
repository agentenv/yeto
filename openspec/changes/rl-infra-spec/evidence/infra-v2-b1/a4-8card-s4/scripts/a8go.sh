#!/bin/bash
# usage: a8go.sh <case> <prefix> <hard_s> <wd_s>        8x H100 on Nebius eu-north1, T4R2S2 <-> T4R4S0 (trainer G0-3, rollout c0,c1 on G4,G5, standby G6/G7 = c2/c3)
# cases: smoke | base | e1a | e1b | wd | a4b | d123 | d4 | d5 | d6 | d7      (DRY=1: print the launch args + triggers, start nothing)
# Starts n2run (launch, --no-island-relaunch --modal-retries 0) + n2inwatch (triggers + router sampler) + selfcheck (+GPU assert 4xL40S, markers) + case helpers + final guard (nstop -> cleanup_run.sh, judge).
# Required env for some cases:  UP_DEADLINE_S (wd: measured, see gpu-plan 9.22 step 2) | (e1b/a4b pass --rl-test-hold-before-check-s ${HOLD_S:-10} / --rl-test-inject-tool-wait-s 30 directly)
# Request time: a request submitted at "train" of rollout k executes before generate k+1 (= before round k+2).
C=$1; P=$2; HARD=$3; WD=$4; SHA=${SHA:-b19b781}; B=/home/michael/work/gpu-b1-runs; R=$B/$P
UP=${UP_DEADLINE_S:-600}; EX=""; STEPS=4; ATTN=4; JUDGE=""; ARMS=()
req() { printf '["%s",%s,"%s",{"target":"%s","expected_config_epoch":%s,"deadline_s":%s}]' "$1" "$2" "$3" "$4" "$5" "$6"; }   # phase rid id target epoch deadline
UPB() { req train $1 $2 T4R4S0 0 ${3:-$UP}; }; DNB() { req train $1 $2 T4R2S2 1 ${3:-600}; }
case $C in
  smoke) STEPS=3; ATTN=3; TRIG="[$(UPB 0 up1 900),$(DNB 1 dn1 900)]";;
  base)  STEPS=12; ATTN=12; TRIG="[]";;
  e1a)   STEPS=12; ATTN=12; TRIG="[$(UPB 1 up1 600),$(UPB 4 up1 600),$(DNB 6 dn1 600)]"; JUDGE="e1a_c --expect-members 2,2,4,4,4,4,4,2,2,2,2,2";;   # old 8-card scheme: up@train rid1 (round 3), same-id repeat @rid4, down@rid6 (round 8)
  e1b)   EX="--rl-test-inject-lora-perturb 0.01 --rl-test-hold-before-check-s ${HOLD_S:-10}"; TRIG="[$(UPB 1 up1 600)]"; JUDGE="e1b";;
  wd)    [ -n "${UP_DEADLINE_S:-}" ] || { echo "wd needs UP_DEADLINE_S (>=1.5x measured start_cells + margin, written in gpu-plan 9.22 before the run)"; exit 5; }
         EX="--rl-test-inject-update-weights-block-s 600"; TRIG="[$(UPB 1 up1 $UP)]"; JUDGE="wd";;
  a4b)   EX="--rl-elastic-tool-wait-board --rl-elastic-drain-timeout-s 5 --rl-test-inject-tool-wait-s 30"; ATTN=4
         TRIG="[$(UPB 0 up1 600),[\"generate\",2,\"dn1\",{\"target\":\"T4R2S2\",\"expected_config_epoch\":1,\"deadline_s\":600}]]"; JUDGE="a4b";;
  d123)  STEPS=5; ATTN=5; EX="--rl-test-inject-stop-failures 1"
         # standby cells on G6/G7 (c2/c3): kill target GPU 6. up1 ep0->1, dn1 ep1->2, up2 at ep2 (killed -> REBUILT_OLD, stays 2), up3 at ep2
         TRIG="[$(UPB 0 up1 600),$(DNB 1 dn1 600),$(req train 2 up2 T4R4S0 2 600),$(req train 3 up3 T4R4S0 2 600)]"
         ARMS+=("dkill.py|[{\"name\":\"d1\",\"tx\":\"up2\",\"when\":{\"kind\":\"fork_op\",\"op\":\"start\",\"status\":\"issued\"},\"gpu\":6,\"mode\":\"when_proc_appears\",\"min_age_s\":10},{\"name\":\"d2\",\"tx\":\"up3\",\"when\":{\"kind\":\"phase\",\"phase\":\"VERIFYING\"},\"gpu\":6,\"mode\":\"immediate\"}]");;
  d4)    EX="--rl-test-inject-stop-failures 100000 --rl-elastic-recovery-timeout-s 120"; TRIG="[$(UPB 0 up1 600),$(DNB 1 dn1 600)]";;
  d5)    EX="--rl-elastic-restart-attempts 1 --rl-test-kill-learner-at COMMITTED"; TRIG="[$(UPB 0 up1 600),$(DNB 1 dn1 600)]"; ARMS+=("dctl.py|marker|up1");;
  d6)    STEPS=3; ATTN=3; EX="--rl-elastic-restart-attempts 1 --rl-test-kill-learner-at QUIESCING"; TRIG="[$(UPB 0 up1 600)]";;
  d7)    STEPS=5; ATTN=5; EX="--rl-elastic-restart-attempts 1"; TRIG="[]"; ARMS+=("dctl.py|kill_then_up|up1|{\"target\":\"T4R4S0\",\"expected_config_epoch\":0,\"deadline_s\":600}");;
  *) echo "unknown case $C"; exit 64;;
esac
ATT=$B/cfg/attestation-8-$ATTN.json; [ -f $ATT ] || { echo "missing $ATT (mkatt8.sh)"; exit 6; }
COMMON="--total-steps $STEPS --rl-placement fixed-partition --rl-rollout-gpus 2 --rl-standby-gpus 2 --rl-elastic --rl-elastic-declare-cells --rl-elastic-cells c0,c1,c2,c3 --rl-elastic-resources $B/cfg/resources-8.json --rl-elastic-initial-config T4R2S2 --rl-observe-timeline --rl-elastic-attestation $ATT"
if [ "${DRY:-0}" = 1 ]; then echo "SHA=$SHA GPU_SPEC=nebius:8xh100@eu-north1 n2run.sh $P 8 $HARD $WD $COMMON $EX"; echo "triggers=$TRIG"; python3 -c "import json,sys;json.loads(sys.argv[1])" "$TRIG" && echo triggers-json-ok; printf 'arms: %s\n' "${ARMS[@]:-none}"; exit 0; fi
SHA=$SHA GPU_SPEC=nebius:8xh100@eu-north1 setsid nohup $B/n2run.sh $P 8 $HARD $WD $COMMON $EX > $B/$P.n2run.out 2>&1 &
sleep 10
setsid nohup $B/n2inwatch.sh $R "$TRIG" > /dev/null 2>&1 &
EXPECT_GPU_NAME=H100 EXPECT_GPU_N=8 setsid nohup $B/selfcheck.sh $R $P $ATT > /dev/null 2>&1 &
setsid nohup $B/diag_pull.sh $R > /dev/null 2>&1 &
for a in "${ARMS[@]}"; do IFS='|' read -ra parts <<< "$a"; setsid nohup $B/n2arm.sh $R "${parts[@]}" > /dev/null 2>&1 & done   # "script|arg1|arg2" (args must not contain |)
# final guard: when the launcher ended (or STOP), pull + cleanup (idempotent), then judge from the pulled journal/tape
setsid nohup bash -c "until [ -f $R/rc.txt ] || [ -f $R/startup_failed ]; do sleep 15; done; sleep 20; $B/nstop.sh $P > $R/final_stop.txt 2>&1; echo \$? > $R/cleanup_rc.txt; [ -n '$JUDGE' ] && $B/judge_after.sh $R $JUDGE > $R/judge.out 2>&1" > /dev/null 2>&1 &
echo started $P $C

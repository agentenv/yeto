#!/bin/bash
# usage: a4go.sh <prefix> <steps> <hard_s> <wd_s> <attestation.json|-> <triggers-json> <after: cmd template with {R} {P} | -> [extra learner args...]
# Starts n2run (launch) + n2inwatch (triggers + router sampler) + selfcheck + after-hook, all detached. Code SHA fixed to 9a06c4b (= origin/integ-decl).
P=$1; STEPS=$2; HARD=$3; WD=$4; ATT=$5; TRIG=$6; AFTER=$7; shift 7
B=/home/michael/work/gpu-b1-runs; R=$B/$P
COMMON="--total-steps $STEPS --rl-placement fixed-partition --rl-rollout-gpus 2 --rl-standby-gpus 2 --rl-elastic --rl-elastic-declare-cells --rl-elastic-cells c0,c1,c2,c3 --rl-elastic-resources $B/cfg/resources-8.json --rl-elastic-initial-config T4R2S2 --rl-observe-timeline"
[ "$ATT" != "-" ] && COMMON="$COMMON --rl-elastic-attestation $ATT"
SHA=9a06c4b setsid nohup $B/n2run.sh $P 8 $HARD $WD $COMMON "$@" > $B/$P.n2run.out 2>&1 &
sleep 10
setsid nohup $B/n2inwatch.sh $R "$TRIG" > /dev/null 2>&1 &
SC=""; [ "$ATT" != "-" ] && SC="$ATT"
setsid nohup $B/selfcheck.sh $R $P $SC > /dev/null 2>&1 &
if [ "$AFTER" != "-" ]; then
  cmd=${AFTER//\{R\}/$R}; cmd=${cmd//\{P\}/$P}
  setsid nohup bash -c "$cmd" > $R.after.out 2>&1 &
fi
echo started $P

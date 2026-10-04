#!/bin/bash
# rl-multinode-island tasks §3 (GPU): one launch of the yeto launcher on Nebius, evidence pulled to RUN_ROOT/<prefix>.
# usage: s1run.sh <case: g0|g12|g3|m1|m3|m2> <prefix> <hard_s> [watchdog_s]
#   g0  : 1 node x 1 L40S, --total-steps 1, no elastic (image/sm_89 probe: 1 train + 1 generate), launcher tears down.
#   g12 : 2 nodes x 1 L40S, --total-steps 2 (cold start ~15 min of the 30 min hard timeout), elastic cfg resources-2x1.json (trainer n0:0, rollout cell n1:0), --keep (G1 topology + G2 cross-node cell).
#   g3  : same cluster, --total-steps 8, --rl-elastic-restart-attempts 1, NO --keep: s1kill.sh kills the worker's raylet after round 1 train
#         -> node_lost/RECOVERY_REQUIRED (G3), the in-place restart is refused by the topology precheck, the launcher's teardown = G4 (per-node confirm lines).
#   m1  : 2 nodes x 2 L40S (nebius:2x2xl40s), Qwen3-0.6B LoRA, --tensor-parallel 1 --pipeline-parallel 2 (PP2 across nodes), cfg resources-2x2.json T2R1S1
#         (trainer n0:0+n1:0, rollout cell n0:1 = mixed node, standby n1:1), --total-steps 2, --keep (M1; HARD 3600 = 30 min case + cold start).
#   m3  : same cluster (CLUSTER_PREFIX=<m1 prefix>), --rl-elastic-declare-cells c0,c1, --total-steps 6, E1 rollout-only edges driven in-container by
#         s1inwatch.py (chain 8 inbox mechanism): up1 at train rid 1 (T2R1S1 -> T2R2S0: standby n1:1 starts as cell c1), dn1 at train rid 3
#         (back to T2R1S1); no --keep (KEEP_M3=1 keeps the cluster for m2).
#   m2  : 2 nodes x 2 L40S, MoE fzyzcjy/Qwen3-30B-A3B-5layer, --tuning lora --lora-targets attention (all-linear is refused with EP>1),
#         --tensor-parallel 1 --pipeline-parallel 1 --expert-parallel 2 (EP2 across nodes: the 2 trainer ranks are n0:0 and n1:0), T2R1S1, --total-steps 2.
# env: CLUSTER_PREFIX (cluster name prefix when several runs share one cluster; default = prefix), SHA (git rev of infra-multinode to archive, default HEAD), RUN_ROOT (/home/michael/work/s1-runs), IMAGE (digest-pinned --rl-image), DRY=1 prints args only.
set -u
C=$1; P=$2; HARD=$3; WD=${4:-$(( $3 + 300 ))}; CP=${CLUSTER_PREFIX:-$P}
D=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$D/../.." && pwd); B=${RUN_ROOT:-/home/michael/work/s1-runs}; R=$B/$P
SKY=/home/michael/work/gpu-head/venv/bin/sky; PY=/home/michael/work/gpu-head/venv/bin/python
# default --rl-image = the snapshot's own pin (yeto.rl.MILES_NEXT_IMAGE); a mismatch makes the sky setup take the slow clone+pip path (~8 min, s1-mn-20261004d)
IMAGE=${IMAGE:-$(cd "$REPO" && PYTHONPATH=. /usr/bin/python3 -c "import yeto.rl as r; print(r.MILES_NEXT_IMAGE if r.MILES_NEXT_IMAGE.startswith('docker:') else 'docker:'+r.MILES_NEXT_IMAGE)" 2>/dev/null)}
[ -n "$IMAGE" ] || { echo "abort: could not resolve MILES_NEXT_IMAGE from $REPO"; exit 65; }
T=$(ps -u michael -L -o pid= | wc -l); if [ "$T" -ge ${THREAD_MAX:-2900} ]; then echo "abort: $T user threads (max ${THREAD_MAX:-2900})"; exit 3; fi
ELASTIC="--rl-placement fixed-partition --rl-rollout-gpus 1 --rl-elastic --rl-elastic-resources $D/resources-2x1.json --rl-elastic-initial-config T1R1S0 --rl-observe-timeline"
# 2x2 (M1/M2/M3): island totals rollout 1 + standby 1; the cfg placement T2R1S1 drives the trainer shape (--actor-num-nodes 2 --actor-num-gpus-per-node 1) and the bundle map
ELASTIC22="--rl-placement fixed-partition --rl-rollout-gpus 1 --rl-standby-gpus 1 --rl-elastic --rl-elastic-resources $D/resources-2x2.json --rl-elastic-initial-config T2R1S1 --rl-observe-timeline"
MODEL="--model Qwen/Qwen3-0.6B --model-revision c1899de289a04d12100db370d81485cdf75e47ca"; LORA="--tuning lora --lora-r 16 --lora-targets all-linear"; PAR=""; TRIG=""
# m3 E1 triggers (s1inwatch.py: [phase, rollout_id, request_id, body]): up1 when train of rollout 1 starts (epoch 0 -> T2R2S0), dn1 at train of rollout 3 (epoch 1 -> T2R1S1)
req() { printf '["%s",%s,"%s",{"target":"%s","expected_config_epoch":%s,"deadline_s":%s}]' "$1" "$2" "$3" "$4" "$5" "$6"; }
case $C in
  g0)  GPU=nebius:1xl40s@eu-north1; STEPS=1; EX=""; KEEP=""; NODES=1;;
  g12) GPU=nebius:2x1xl40s@eu-north1; STEPS=2; EX="$ELASTIC"; KEEP="--keep"; NODES=2;;
  g3)  GPU=nebius:2x1xl40s@eu-north1; STEPS=8; EX="$ELASTIC --rl-elastic-restart-attempts 1"; KEEP=""; NODES=2;;
  m1)  GPU=nebius:2x2xl40s@eu-north1; STEPS=2; PAR="--tensor-parallel 1 --pipeline-parallel 2"; EX="$ELASTIC22"; KEEP="--keep"; NODES=2;;
  m3)  GPU=nebius:2x2xl40s@eu-north1; STEPS=6; PAR="--tensor-parallel 1 --pipeline-parallel 2"; EX="$ELASTIC22 --rl-elastic-declare-cells --rl-elastic-cells c0,c1"; KEEP="${KEEP_M3:+--keep}"; NODES=2
       TRIG="[$(req train 1 up1 T2R2S0 0 ${UP_DEADLINE_S:-600}),$(req train 3 dn1 T2R1S1 1 ${DN_DEADLINE_S:-600})]";;
  m2)  GPU=nebius:2x2xl40s@eu-north1; STEPS=2; MODEL="--model fzyzcjy/Qwen3-30B-A3B-5layer --model-revision 9c2ee37f22b7ef150675311b3d5e1c671838ffe1"; LORA="--tuning lora --lora-r 16 --lora-targets attention"
       PAR="--tensor-parallel 1 --pipeline-parallel 1 --expert-parallel 2"; EX="$ELASTIC22"; KEEP="${KEEP_M2:+--keep}"; NODES=2;;
  *) echo "unknown case $C"; exit 64;;
esac
ARGS="launch --controller local --training-mode rl --rl-single-island-no-sync --on-demand --gpu $GPU --cluster-prefix $CP $KEEP --no-island-relaunch --modal-retries 0 --rl-image $IMAGE $MODEL --data zhuzilin/gsm8k --data-revision 0cbd9f31d91ac21a7613dcbc7fef992adac459ae --reward-function gsm8k_reward:score $LORA $PAR --fragments 1 --pipeline 1 --rollout-batch-size 4 --n-samples-per-prompt 8 --rollout-max-response-len 384 --seq-len 1024 --inner-lr 1e-5 --seed 17 --apply-chat-template-kwargs '{\"enable_thinking\": false}' --trust-remote-code --total-steps $STEPS $EX"
CL=$CP-l0-eu-north1
if [ "${DRY:-0}" = 1 ]; then echo "cluster=$CL nodes=$NODES case=$C"; echo "$ARGS"; [ -n "$TRIG" ] && { echo "triggers=$TRIG"; /usr/bin/python3 -c "import json,sys;json.loads(sys.argv[1])" "$TRIG" && echo triggers-json-ok; }; exit 0; fi
mkdir -p $R/home $R/runs $R/pulled $R/yeto
for d in .sky .nebius .ssh; do ln -sfn /home/michael/$d $R/home/$d; done
git -C $REPO archive ${SHA:-HEAD} | tar x -C $R/yeto
cp /home/michael/work/gpu-default-modal/yeto/gsm8k_reward.py $R/yeto/; touch $R/yeto/yeto-rl-echo-events
# m3: the controller refuses every E1 request without a capability attestation ("no capability attestation: no transition is certified",
# s8-m1m3-20261004a M3 FAIL: requests consumed but never journaled). Fingerprint = local reconstruction of the island Miles argv from the
# snapshot (fp_local22.py; reproduced the real s8-m1m3-20261004a-m3 rl_driver_start value sha256:0d17e24b...), edges = resources-2x2.json.
if [ $C = m3 ]; then
  fp=$(/tmp/yeto-venv/bin/python $D/fp_local22.py $R/yeto 2>$R/fp_local22.err | python3 -c "import json,sys;print(json.load(sys.stdin)['fp'])" 2>/dev/null)
  [ -n "$fp" ] || { echo "abort: m3 attestation fingerprint failed (see $R/fp_local22.err)"; exit 67; }
  printf '{"runtime_fingerprint":"%s","execution_modes":["partitioned-serial"],"certified_edges":[{"source":"T2R1S1","target":"T2R2S0","kind":"rollout-only"},{"source":"T2R2S0","target":"T2R1S1","kind":"rollout-only"}]}' "$fp" > $R/attestation-m3.json
  ARGS="$ARGS --rl-elastic-attestation $R/attestation-m3.json"
fi
git -C $REPO rev-parse ${SHA:-HEAD} > $R/yeto_sha.txt; echo "$ARGS" > $R/args.txt; echo $CL > $R/cluster.txt; echo $NODES > $R/nodes.txt; echo $C > $R/case.txt; [ -n "$TRIG" ] && echo "$TRIG" > $R/triggers.json
# per-run watchdog: sky down by THIS cluster name only
setsid nohup bash -c "sleep $WD; HOME=/home/michael $SKY down -y $CL > $R/watchdog.out 2>&1; touch $R/WATCHDOG_FIRED" >/dev/null 2>&1 &
echo $! > $R/watchdog.pid
# puller (every 10 s while the launcher runs): events, journal, in-container probe log, per-node GPU/process snapshots
setsid nohup bash -c "
export HOME=/home/michael; armed=0; inarmed=0
while [ ! -f $R/rc.txt ]; do
  if timeout 60 $SKY status $CL 2>/dev/null | grep -q ' UP '; then
    S='ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20'
    if [ \$armed = 0 ] && [ $NODES = 2 ]; then b=\$(base64 -w0 $D/s1probe.sh); timeout 60 \$S $CL \"mkdir -p ~/yeto-rl && echo \$b | base64 -d > ~/yeto-rl/s1probe.sh && (setsid nohup bash ~/yeto-rl/s1probe.sh > ~/yeto-rl/s1probe.out 2>&1 &) ; sleep 1; pgrep -f s1probe.sh | head -1\" > $R/probe-arm.txt 2>&1 && [ -s $R/probe-arm.txt ] && armed=1; fi
    if [ \$inarmed = 0 ] && [ -s $R/triggers.json ]; then b=\$(base64 -w0 $D/s1inwatch.py); t=\$(base64 -w0 $R/triggers.json); timeout 60 \$S $CL \"mkdir -p ~/yeto-rl && echo \$b | base64 -d > ~/yeto-rl/s1inwatch.py && (setsid nohup python3 ~/yeto-rl/s1inwatch.py \\\"\\\$(echo \$t | base64 -d)\\\" > ~/yeto-rl/inwatch.out 2>&1 &) ; sleep 1; pgrep -f s1inwatch.py | head -1\" > $R/inwatch-arm.txt 2>&1 && [ -s $R/inwatch-arm.txt ] && inarmed=1; fi
    [ -s $R/triggers.json ] && timeout 60 \$S $CL 'cat ~/yeto-rl/inwatch.log 2>/dev/null' > $R/pulled/.iw 2>/dev/null && [ -s $R/pulled/.iw ] && mv $R/pulled/.iw $R/pulled/inwatch.log
    timeout 60 \$S $CL 'cat ~/yeto-output/rl-island-0.jsonl 2>/dev/null' > $R/pulled/.tmp 2>/dev/null && [ -s $R/pulled/.tmp ] && mv $R/pulled/.tmp $R/pulled/rl-island-0.jsonl
    timeout 60 \$S $CL 'cat ~/yeto-rl/elastic-state/reconfig/journal.jsonl 2>/dev/null' > $R/pulled/.j 2>/dev/null && [ -s $R/pulled/.j ] && mv $R/pulled/.j $R/pulled/journal.jsonl
    [ -s $R/triggers.json ] && timeout 60 \$S $CL 'tail -n +1 ~/yeto-rl/elastic-state/inbox/*.status.json 2>/dev/null' > $R/pulled/.st 2>/dev/null && [ -s $R/pulled/.st ] && mv $R/pulled/.st $R/pulled/inbox-status.txt
    timeout 60 \$S $CL 'cat ~/yeto-rl/s1probe.log 2>/dev/null' > $R/pulled/.p 2>/dev/null && [ -s $R/pulled/.p ] && mv $R/pulled/.p $R/pulled/s1probe.log
    timeout 90 \$S $CL 'tail -c 4000000 ~/sky_logs/*/run.log 2>/dev/null' > $R/pulled/.r 2>/dev/null && [ -s $R/pulled/.r ] && mv $R/pulled/.r $R/pulled/run.log   # the launcher streams only the setup; the job log stays on the head
    for n in $CL \$( [ $NODES = 2 ] && echo $CL-worker1 ); do
      [ -s $R/pulled/gpu-\$n.txt ] || timeout 60 \$S \$n 'hostname; nvidia-smi --query-gpu=index,uuid,name,driver_version --format=csv,noheader' > $R/pulled/gpu-\$n.txt 2>/dev/null
      timeout 60 \$S \$n 'date -u +%FT%TZ; nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader; ps -eo pid,args --no-headers | grep -E \"ray::|sglang|yeto.rl.learner|raylet\" | grep -v grep | cut -c1-140' >> $R/pulled/apps-\$n.txt 2>/dev/null
    done
  fi
  sleep 10
done" > $R/puller.log 2>&1 &
echo $! > $R/puller.pid
[ $C = g3 ] && { setsid nohup $D/s1kill.sh $R > $R/s1kill.out 2>&1 & echo $! > $R/s1kill.pid; }
(
export HOME=$R/home YETO_RUNS_DIR=$R/runs PYTHONPATH=$R/yeto
eval "$(/usr/bin/python3 - <<'PY'
import base64, json, shlex
a = json.load(open("/home/michael/.docker/config.json"))["auths"]["ghcr.io"]["auth"]
u, t = base64.b64decode(a).decode().split(":", 1)
print(f"export SKYPILOT_DOCKER_USERNAME={shlex.quote(u)} SKYPILOT_DOCKER_PASSWORD={shlex.quote(t)} SKYPILOT_DOCKER_SERVER=ghcr.io")
PY
)"
cd $R/yeto; date -u +%FT%TZ > $R/start_utc.txt
eval "timeout $HARD $PY -m yeto.cli $(cat $R/args.txt)" 2>&1 | tee $R/launch.log | awk '{ print strftime("%FT%TZ", systime(), 1) " " $0; fflush() }' > $R/launch.ts.log
echo "rc=${PIPESTATUS[0]}" > $R/rc.txt.tmp; date -u +%FT%TZ > $R/end_utc.txt
sleep 20; mv $R/rc.txt.tmp $R/rc.txt
)
echo "done $P $(cat $R/rc.txt)"

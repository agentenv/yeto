#!/bin/bash
# rl-multinode-island tasks §3 (GPU): one launch of the yeto launcher on Nebius, evidence pulled to RUN_ROOT/<prefix>.
# usage: s1run.sh <case: g0|g12|g3|m1|m3|m2|m4a|m4b|m4a1|m4b1|m5> <prefix> <hard_s> [watchdog_s]
#   g0  : 1 node x 1 L40S, --total-steps 1, no elastic (image/sm_89 probe: 1 train + 1 generate), launcher tears down.
#   g12 : 2 nodes x 1 L40S, --total-steps 2 (cold start ~15 min of the 30 min hard timeout), elastic cfg resources-2x1.json (trainer n0:0, rollout cell n1:0), --keep (G1 topology + G2 cross-node cell).
#   g3  : same cluster, --total-steps 8, --rl-elastic-restart-attempts 1, NO --keep: s1kill.sh kills the worker's raylet after round 1 train
#         -> node_lost/RECOVERY_REQUIRED (G3), the in-place restart is refused by the topology precheck, the launcher's teardown = G4 (per-node confirm lines).
#   m1  : 2 nodes x 2 L40S (nebius:2x2xl40s), Qwen3-0.6B LoRA, --tensor-parallel 1 --pipeline-parallel 2 (PP2 across nodes), cfg resources-2x2.json T2R1S1
#         (trainer n0:0+n1:0, rollout cell n0:1 = mixed node, standby n1:1), --total-steps 2, --keep (M1; HARD 3600 = 30 min case + cold start).
#   m3  : same cluster (CLUSTER_PREFIX=<m1 prefix>), --rl-elastic-declare-cells c0,c1, --total-steps 6, E1 rollout-only edges driven in-container by
#         s1inwatch.py (chain 8 inbox mechanism): up1 at train rid 1 (T2R1S1 -> T2R2S0: standby n1:1 starts as cell c1), dn1 at train rid 3
#         (back to T2R1S1); no --keep (KEEP_M3=1 keeps the cluster for m2).
#   m4a : M4 island A: m1's 2x2 cfg + --rl-checkpoint-store "$STORE" (env, required) --total-steps 8 (no --rl-elastic-restart-attempts: launcher rejects 0; absent = no restart loop), no --keep;
#         s1kill.sh (KILL_RID=2) kills the worker's island raylet once train of rollout 2 starts (rounds 0,1 trained, round cut r000002 in the store)
#         -> node_lost/RECOVERY_REQUIRED, launcher teardown. The chain then confirms A's instances are gone (cleanup_run.sh).
#   m4b : M4 island B: NEW prefix/cluster, same cfg + same STORE + --rl-elastic-accept-rebind, --total-steps M4B_STEPS (default 8 => >= 1 round after
#         the resume at rollout 2): store restore, gpu_pool rebind, round-cut weight restore, rollout ids continue. Judge: s1judge.py m4 <A run> <B run>.
#   m4a1/m4b1: M4 2x1 variant = g3 topology (fixed-partition, resources-2x1.json T1R1S0: trainer n0:0, rollout n1:0), TP1 PP1,
#         2 nodes x 1 GPU (M4X1_GPU, default nebius:2x1xl40s@eu-north1; H100: nebius:2x1xh100@eu-north1) + --rl-checkpoint-store,
#         steps 8 both; same chain as m4a/m4b (s1kill KILL_RID=2 kills the worker = rollout node; B: new cluster, restore, rebind,
#         continue). A/B argv identical except B's --rl-elastic-accept-rebind and the instance type: ITYPE -> A (and B unless
#         M4B_ITYPE), M4B_ITYPE -> B only. Judge: s1judge.py m4 <A> <B> (2 new uuids).
#   m2  : 2 nodes x 2 L40S, MoE fzyzcjy/Qwen3-30B-A3B-5layer, --tuning lora --lora-targets attention (all-linear is refused with EP>1),
#         --tensor-parallel 1 --pipeline-parallel 1 --expert-parallel 2 (EP2 across nodes: the 2 trainer ranks are n0:0 and n1:0), T2R1S1, --total-steps 2.
#   m5  : 2 nodes x 4 L40S (nebius:2x4xl40s = gpu-l40s-d_4gpu-128vcpu-768gb), Qwen3-0.6B LoRA (16 q / 8 kv heads: TP8 = 1 kv head per rank),
#         COLOCATED (default placement): trainer DP8 (--actor-num-nodes 2 x 4, tp1 pp1) and ONE sglang TP8 rollout engine share all 8 GPUs;
#         --rollout-num-gpus-per-engine 8 --rl-allow-cross-node-engine-tp (sglang nnodes=2, node_rank 0 = TP0-3 on n0, 1 = TP4-7 on n1),
#         --total-steps 2, --keep; the puller snapshots sglang::scheduler_TP<r> -> GPU uuid per node (s1m5ranks.py -> pulled/m5ranks.jsonl);
#         after the launcher returns, s1m5post.sh runs the cross-node NCCL all-reduce probe + generation-only TP8 (2 nodes) vs TP4 (head)
#         greedy reference and tears the cluster down (KEEP_M5=1 keeps it). Judge: s1judge.py <run> m5 (correctness verdict; perf numbers only).
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
  m2)  GPU=nebius:2x2xl40s@eu-north1; STEPS=2; MODEL="--model fzyzcjy/Qwen3-30B-A3B-5layer --model-revision 9c2ee37f22b7ef150675311b3d5e1c671838ffe1"; LORA="--tuning lora --lora-r 16 --lora-targets attention"; REWARD=length_reward:score
       PAR="--tensor-parallel 1 --pipeline-parallel 1 --expert-parallel 2"; EX="$ELASTIC22"; KEEP="${KEEP_M2:+--keep}"; NODES=2;;
  m4a|m4b) [ -n "${STORE:-}" ] || { echo "abort: $C needs STORE=s3://<bucket>/<prefix> (the --rl-checkpoint-store bucket; same value for m4a and m4b)"; exit 66; }
       GPU=nebius:2x2xl40s@eu-north1; PAR="--tensor-parallel 1 --pipeline-parallel 2"; KEEP=""; NODES=2
       if [ $C = m4a ]; then STEPS=8; EX="$ELASTIC22 --rl-checkpoint-store $STORE"
       else STEPS=${M4B_STEPS:-8}; EX="$ELASTIC22 --rl-checkpoint-store $STORE --rl-elastic-accept-rebind${M4B_ITYPE:+ --learner-instance-type $M4B_ITYPE}"; fi;;
  m4a1|m4b1) [ -n "${STORE:-}" ] || { echo "abort: $C needs STORE=s3://<bucket>/<prefix> (same value for m4a1 and m4b1)"; exit 66; }
       GPU=${M4X1_GPU:-nebius:2x1xl40s@eu-north1}; PAR="--tensor-parallel 1 --pipeline-parallel 1"; KEEP=""; NODES=2; STEPS=8
       EX="$ELASTIC --rl-checkpoint-store $STORE"
       if [ $C = m4b1 ]; then EX="$EX --rl-elastic-accept-rebind"; [ -n "${M4B_ITYPE:-}" ] && ITYPE=$M4B_ITYPE; fi;;
  m5)  GPU=${M5_GPU:-nebius:2x4xl40s@eu-north1}; STEPS=${M5_STEPS:-2}; PAR="--tensor-parallel 1 --pipeline-parallel 1"; KEEP="--keep"; NODES=2
       # M5_USE_GPUS=M: physical machines bigger than the test (e.g. M5_GPU=nebius:2x8xh100@eu-north1 M5_USE_GPUS=4): the island gets
       # GPUs 0..M-1 per node; network tier none (no IB GPU cluster / fixed fabric: NCCL runs over TCP like the L40S runs)
       case $GPU in *:2x*x*) ;; *) echo "abort: m5 needs a 2-node --gpu (got $GPU)"; exit 68;; esac
       EX="--rollout-num-gpus-per-engine 8 --rl-allow-cross-node-engine-tp --rl-observe-timeline --rl-island-network-tier ${M5_NET_TIER:-none}${M5_USE_GPUS:+ --rl-island-use-gpus-per-node $M5_USE_GPUS}"
       PHYS=$(echo "$GPU" | sed -E 's/^[a-z]+:2x([0-9]+)x.*/\1/'); USE=${M5_USE_GPUS:-$PHYS}
       [ $(( 2 * USE )) = 8 ] || { echo "abort: m5 needs 4 island GPUs per node (TP8 = 2 x 4), got physical $PHYS use $USE"; exit 68; }
       # watchdog must outlive the launcher AND the post step (it ssh-es into the kept cluster for up to M5_POST_HARD)
       WD=${4:-$(( HARD + ${M5_POST_HARD:-1500} + 300 ))};;
  d1sweep|d1e1) # D1-2 (2.4) fixed config D1_CFG (T2R1S1|T2R2S0|T1R3S0) seed SEED, no triggers, --keep; D1-3 (5.1) d1e1 = m3 with 3 up/down
       # pairs (up at train rid 1,5,9; dn at 3,7,11), 14 steps (dn3's safe point rid 12 needs a later round; fp_local22 gets the same steps/seed).
       # D1_H200=1 (S11): nebius 1 node x 8 H200 (gpu-h200-sxm_8gpu-128vcpu-1600gb, $36/h; D1_GPU=h100 -> 8xh100, gpu-h100-sxm_8gpu-128vcpu-1600gb, $30.8/h, 80GB), island allocated GPUs 0-3, resources-1x4-h200.json
       if [ "${D1_H200:-0}" = 1 ]; then case ${D1_GPU:-h200} in h100|h200) ;; *) echo "abort: D1_GPU must be h100|h200"; exit 69;; esac
         GPU=nebius:8x${D1_GPU:-h200}@eu-north1; NODES=1; RES=$D/resources-1x4-h200.json; ALLOC=" --rl-island-use-gpus-per-node 4 --rl-island-network-tier none"; USE=4; PHYS=8
       else GPU=nebius:2x2xl40s@eu-north1; NODES=2; RES=$D/resources-2x2.json; ALLOC=""; fi
       OBS="--rl-observe-timeline$ALLOC"; PAR="--tensor-parallel 1 --pipeline-parallel 2"
       if [ $C = d1e1 ]; then
         STEPS=${D1E1_STEPS:-14}; KEEP="${KEEP_M3:+--keep}"
         EX="--rl-placement fixed-partition --rl-rollout-gpus 1 --rl-standby-gpus 1 --rl-elastic --rl-elastic-resources $RES --rl-elastic-initial-config T2R1S1 $OBS --rl-elastic-declare-cells --rl-elastic-cells c0,c1"
         TRIG="[$(req train 1 up1 T2R2S0 0 600),$(req train 3 dn1 T2R1S1 1 600),$(req train 5 up2 T2R2S0 2 600),$(req train 7 dn2 T2R1S1 3 600),$(req train 9 up3 T2R2S0 4 600),$(req train 11 dn3 T2R1S1 5 600)]"
       else
         STEPS=${D1_STEPS:-6}; KEEP="--keep"
         case ${D1_CFG:-} in
           T2R1S1) EX="--rl-placement fixed-partition --rl-rollout-gpus 1 --rl-standby-gpus 1 --rl-elastic --rl-elastic-resources $RES --rl-elastic-initial-config T2R1S1 $OBS";;
           T2R2S0) EX="--rl-placement fixed-partition --rl-rollout-gpus 2 --rl-elastic --rl-elastic-resources $RES --rl-elastic-initial-config T2R2S0 $OBS";;
           T1R3S0) PAR="--tensor-parallel 1 --pipeline-parallel 1"; [ -n "$ALLOC" ] || RES=$D/resources-2x2-t1r3.json
                   EX="--rl-placement fixed-partition --rl-rollout-gpus 3 --rl-elastic --rl-elastic-resources $RES --rl-elastic-initial-config T1R3S0 $OBS";;
           *) echo "abort: d1sweep needs D1_CFG=T2R1S1|T2R2S0|T1R3S0"; exit 69;;
         esac
       fi;;
  fn8s|fn8r) # fn8r = fn8s + dense length reward (fnrun.sh). S11 seg 3 (Flash-Next stage A, FN-A-PRELAUNCH-REVIEW.md): argv rendered by fnrun.sh fn8s (1x8 FN_GPU h200 default | h100, 4layer, model store FS,
       # torch_dist ref-load, colocated + offload, observe-timeline), STEPS (default 6), --keep (the chain downs the cluster).
       GPU=nebius:1x8x${FN_GPU:-h200}@eu-north1; NODES=1; STEPS=${STEPS:-6}; KEEP="--keep"; EX="";;
  *) echo "unknown case $C"; exit 64;;
esac
ARGS="launch --controller local --training-mode rl --rl-single-island-no-sync --on-demand --gpu $GPU --cluster-prefix $CP $KEEP --no-island-relaunch --modal-retries 0 --rl-image $IMAGE $MODEL --data zhuzilin/gsm8k --data-revision 0cbd9f31d91ac21a7613dcbc7fef992adac459ae --reward-function ${REWARD:-gsm8k_reward:score} $LORA $PAR --fragments 1 --pipeline 1 --rollout-batch-size 4 --n-samples-per-prompt 8 --rollout-max-response-len 384 --seq-len 1024 --inner-lr 1e-5 --seed ${SEED:-17} --apply-chat-template-kwargs '{\"enable_thinking\": false}' --trust-remote-code --total-steps $STEPS $EX${ITYPE:+ --learner-instance-type $ITYPE}"
case $C in fn8s|fn8r) ARGS="$(PREFIX=$CP STEPS=$STEPS IMAGE=$IMAGE BOOT_ONLY=${BOOT_ONLY:-0} bash $D/fnrun.sh $C) --keep --modal-retries 0";; esac   # BOOT_ONLY=1: S11 fnboot (--rl-boot-only)
CL=$CP-l0-eu-north1
if [ "${DRY:-0}" = 1 ]; then echo "cluster=$CL nodes=$NODES case=$C hard=$HARD wd=$WD${USE:+ gpus_per_node physical=$PHYS use=$USE}"; echo "$ARGS"; [ -n "$TRIG" ] && { echo "triggers=$TRIG"; /usr/bin/python3 -c "import json,sys;json.loads(sys.argv[1])" "$TRIG" && echo triggers-json-ok; }; exit 0; fi
mkdir -p $R/home $R/runs $R/pulled $R/yeto
for d in .sky .nebius .ssh; do ln -sfn /home/michael/$d $R/home/$d; done
# m4: the S3 checkpoint store is mounted on Nebius nodes with the static AWS keys sky uploads (~/.aws/credentials, SHARED_CREDENTIALS_FILE identity)
case $C in m4*) ln -sfn /home/michael/.aws $R/home/.aws;; esac
git -C $REPO archive ${SHA:-HEAD} | tar x -C $R/yeto
cp /home/michael/work/gpu-default-modal/yeto/gsm8k_reward.py $R/yeto/tests/multinode_gpu/length_reward.py $R/yeto/; touch $R/yeto/yeto-rl-echo-events
# m3: the controller refuses every E1 request without a capability attestation ("no capability attestation: no transition is certified",
# s8-m1m3-20261004a M3 FAIL: requests consumed but never journaled). Fingerprint = local reconstruction of the island Miles argv from the
# snapshot (fp_local22.py; reproduced the real s8-m1m3-20261004a-m3 rl_driver_start value sha256:0d17e24b...), edges = resources-2x2.json.
if [ $C = m3 ] || [ $C = d1e1 ]; then
  fp=$(/tmp/yeto-venv/bin/python $D/fp_local22.py $R/yeto --total-steps $STEPS --seed ${SEED:-17} --gpu $GPU --rl-elastic-resources ${RES:-$D/resources-2x2.json}${ALLOC:-} 2>$R/fp_local22.err | tail -1 | python3 -c "import json,sys;print(json.load(sys.stdin)['fp'])" 2>/dev/null)
  [ -n "$fp" ] || { echo "abort: m3 attestation fingerprint failed (see $R/fp_local22.err)"; exit 67; }
  printf '{"runtime_fingerprint":"%s","execution_modes":["partitioned-serial"],"certified_edges":[{"source":"T2R1S1","target":"T2R2S0","kind":"rollout-only"},{"source":"T2R2S0","target":"T2R1S1","kind":"rollout-only"}]}' "$fp" > $R/attestation-m3.json
  ARGS="$ARGS --rl-elastic-attestation $R/attestation-m3.json"
fi
git -C $REPO rev-parse ${SHA:-HEAD} > $R/yeto_sha.txt; echo "$ARGS" > $R/args.txt; echo $CL > $R/cluster.txt; echo $NODES > $R/nodes.txt; echo $C > $R/case.txt; [ -n "${USE:-}" ] && printf '%s %s\n' "$PHYS" "$USE" > $R/gpus_per_node.txt; [ -n "$TRIG" ] && echo "$TRIG" > $R/triggers.json
# per-run watchdog: sky down by THIS cluster name only
setsid nohup bash -c "sleep $WD; HOME=/home/michael $SKY down -y $CL > $R/watchdog.out 2>&1; touch $R/WATCHDOG_FIRED" >/dev/null 2>&1 &
echo $! > $R/watchdog.pid
B5=$(base64 -w0 $D/s1m5ranks.py)   # m5 rank probe, piped into python3 on each node by the puller
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
    case $C in m4*) timeout 60 \$S $CL 'for f in STORE-MANIFEST.json round-cut.json; do echo \"== \$f\"; cat ~/yeto-checkpoint-store/\$f 2>/dev/null; echo; done; echo \"== round-cuts\"; ls ~/yeto-checkpoint-store/round-cuts 2>/dev/null' > $R/pulled/.s 2>/dev/null && [ -s $R/pulled/.s ] && mv $R/pulled/.s $R/pulled/store.txt;; esac
    timeout 90 \$S $CL 'tail -c 4000000 ~/sky_logs/*/run.log 2>/dev/null' > $R/pulled/.r 2>/dev/null && [ -s $R/pulled/.r ] && mv $R/pulled/.r $R/pulled/run.log   # the launcher streams only the setup; the job log stays on the head
    [ $C = m5 ] && { k=0; for n in $CL $CL-worker1; do timeout 60 \$S \$n \"echo $B5 | base64 -d | python3 - \$k\" >> $R/pulled/m5ranks.jsonl 2>/dev/null; k=\$((k+1)); done; }
    for n in $CL \$( [ $NODES = 2 ] && echo $CL-worker1 ); do
      [ -s $R/pulled/gpu-\$n.txt ] || timeout 60 \$S \$n 'hostname; nvidia-smi --query-gpu=index,uuid,name,driver_version --format=csv,noheader' > $R/pulled/gpu-\$n.txt 2>/dev/null
      timeout 60 \$S \$n 'date -u +%FT%TZ; nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader; ps -eo pid,args --no-headers | grep -E \"ray::|sglang|yeto.rl.adapters.miles.island_entry|raylet\" | grep -v grep | cut -c1-140' >> $R/pulled/apps-\$n.txt 2>/dev/null
    done
  fi
  sleep 10
done" > $R/puller.log 2>&1 &
echo $! > $R/puller.pid
case $C in m4a|m4a1) true;; *) false;; esac && { KILL_RID=2 setsid nohup $D/s1kill.sh $R > $R/s1kill.out 2>&1 & echo $! > $R/s1kill.pid; }   # after rounds 0,1 (round cut at rid 2 exists)
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
# provisioning never succeeded (capacity): no cluster exists -> skip the m5 post step (it would ssh into nothing for up to
# M5_POST_HARD) and kill the per-run watchdog, which would otherwise `sky down` a later retry that reuses this cluster name
if grep -qE "failed to provision|ResourcesUnavailableError" $R/launch.log; then
  kill $(cat $R/watchdog.pid) 2>/dev/null; echo "provision failed: watchdog killed, post skipped" > $R/provision_failed.txt
elif [ $C = m5 ]; then
  timeout ${M5_POST_HARD:-1500} $D/s1m5post.sh $R > $R/m5post.out 2>&1; echo "m5post rc=$?" >> $R/m5post.out
fi
echo "done $P $(cat $R/rc.txt)"

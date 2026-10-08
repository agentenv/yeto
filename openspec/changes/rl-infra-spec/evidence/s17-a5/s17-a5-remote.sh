#!/bin/bash
# S17 N7 I2 = rl-infra-spec 3.8 (gpu-plan-v2 A5): two strict-avg islands, each Modal 3xH100! T1R1S1 (c0 started, c1 declared
#   not started), Nebius eu-north1 CPU VM head (controller + syncer). Derived from s15-island1b-remote.sh (no killer/stopper).
#   CASE=base   : no requests, 6 rounds
#   CASE=switch : island 0 up (train r1) -> down (train r3) -> up during the last round (train r5, finalization must refuse)
#   CASE=quorum : quorum 120 s, pause margin 2.0, 150 s start_cells delay; island 0 up (train r1, deadline 230 s), 4 rounds
#   A sidecar (s17-a5-prep/sidecar.py) is exec'd into every island container: triggers (island 0), GPU / router samplers,
#   elastic-state snapshots; all files land in ~/yeto-output -> tape volume.
# usage: s17-a5-remote.sh <run_id>   env: CASE, PLAN_ONLY=1, HARD (s), THREAD_MAX
set -u
CASE=${CASE:?CASE=base|switch|quorum|merged}; P=${1:?run id}; HARD=${HARD:-2700}; WD=$(( HARD + 600 ))
REPO=${REPO:-/home/michael/work/s17-infra-i2}; B=/home/michael/work/s1-runs; R=$B/$P; PREP=$B/s17-a5-prep
SKY=/home/michael/work/gpu-head/venv/bin/sky; PY=/home/michael/work/gpu-head/venv/bin/python
APP=yeto-$P; TAPEVOL=yeto-event-tapes; HEAD_REGION=nebius/eu-north1
export CARGO_HOME=/home/michael/work/gpu-default-modal/cargo RUSTUP_HOME=/home/michael/work/gpu-default-modal/rustup
export PATH=$CARGO_HOME/bin:$PATH
IMAGE=$(cd "$REPO" && PYTHONPATH=. /usr/bin/python3 -c "import yeto.rl as r; print(r.MILES_NEXT_IMAGE if r.MILES_NEXT_IMAGE.startswith('docker:') else 'docker:'+r.MILES_NEXT_IMAGE)" 2>/dev/null)
[ -n "$IMAGE" ] || { echo "abort: no MILES_NEXT_IMAGE"; exit 65; }
T=$(ps -u michael -L -o pid= | wc -l); [ "$T" -lt ${THREAD_MAX:-3000} ] || { echo "abort: $T user threads"; exit 3; }
pgrep -x raylet >/dev/null && { echo "abort: local Ray running"; exit 3; }
case $CASE in
  base)   STEPS=6; EXTRA=""; TRIG='{"triggers": []}' ;;
  switch) STEPS=6; EXTRA=""; TRIG='{"triggers": [["train",1,"up1",{"target":"T1R2S0","expected_config_epoch":0,"deadline_s":600}],["train",3,"dn1",{"target":"T1R1S1","expected_config_epoch":1,"deadline_s":600}],["train",5,"fin1",{"target":"T1R2S0","expected_config_epoch":2,"deadline_s":600}]]}' ;;
  quorum) STEPS=4; EXTRA="--rl-elastic-quorum-timeout-s 120 --rl-elastic-pause-margin 2.0 --rl-test-inject-start-delay-s 150"; TRIG='{"triggers": [["train",1,"up1",{"target":"T1R2S0","expected_config_epoch":0,"deadline_s":230}]]}' ;;
  merged) STEPS=6; EXTRA="--rl-elastic-attestation a5-attestation.json --rl-elastic-quorum-timeout-s 120 --rl-elastic-pause-margin 4.0 --rl-test-inject-start-delay-s 150"; TRIG='{"triggers": [["train",1,"up1",{"target":"T1R2S0","expected_config_epoch":0,"deadline_s":450}],["train",3,"dn1",{"target":"T1R1S1","expected_config_epoch":1,"deadline_s":600}],["train",5,"fin1",{"target":"T1R2S0","expected_config_epoch":2,"deadline_s":600}]]}' ;;
  *) echo "bad CASE"; exit 64 ;;
esac
if [ "${PLAN_ONLY:-0}" = 1 ]; then R=/tmp/$P-plan; rm -rf $R; fi
[ -e $R/start_utc.txt ] && { echo "abort: $R already used"; exit 67; }
mkdir -p $R/home $R/runs $R/yeto $R/head
for d in .sky .ssh .nebius .cache .huggingface .config; do [ -e /home/michael/$d ] && ln -sfn /home/michael/$d $R/home/$d; done
git -C $REPO archive ${SHA:-HEAD} | tar x -C $R/yeto
cp /home/michael/work/gpu-default-modal/yeto/gsm8k_reward.py $R/yeto/; touch $R/yeto/yeto-rl-echo-events
cp $PREP/resources-3.json $R/yeto/a5-resources.json; cp $PREP/attestation-a5.json $R/yeto/a5-attestation.json
MODEL="--model Qwen/Qwen3-0.6B --model-revision c1899de289a04d12100db370d81485cdf75e47ca"; LORA="--tuning lora --lora-r 16 --lora-targets all-linear"
MODALX="--modal-gpu-exact --modal-retries 0 --modal-timeout-s $HARD --modal-tape-volume $TAPEVOL"
ELASTIC="--rl-sync-preset strict-avg --rl-placement fixed-partition --rl-rollout-gpus 1 --rl-standby-gpus 1 --rl-elastic --rl-elastic-declare-cells --rl-elastic-cells c0,c1 --rl-elastic-resources a5-resources.json --rl-elastic-initial-config T1R1S1 $EXTRA"
ARGS="launch --controller head --syncer-region $HEAD_REGION --training-mode rl --rl-engine ports $ELASTIC --rl-stall-timeout 2400 --rl-observe-timeline --on-demand --gpu modal:3xh100,modal:3xh100 --cluster-prefix $P $MODALX --rl-image $IMAGE $MODEL --data zhuzilin/gsm8k --data-revision 0cbd9f31d91ac21a7613dcbc7fef992adac459ae --reward-function gsm8k_reward:score $LORA --fragments 1 --pipeline 1 --rollout-batch-size 4 --n-samples-per-prompt 8 --rollout-max-response-len 384 --seq-len 1024 --inner-lr 1e-5 --seed 17 --apply-chat-template-kwargs '{\"enable_thinking\": false}' --trust-remote-code --total-steps $STEPS"
git -C $REPO rev-parse ${SHA:-HEAD} > $R/yeto_sha.txt; echo "$ARGS" > $R/args.txt; echo "$TRIG" > $R/triggers.json; echo "$APP" > $R/app.txt; cp $PREP/sidecar.py $R/
eval "$(/usr/bin/python3 - <<'PYC'
import shlex, tomllib
t = tomllib.load(open("/home/michael/.modal.toml", "rb"))
prof = next((v for v in t.values() if isinstance(v, dict) and v.get("active")), None) or next(v for v in t.values() if isinstance(v, dict))
print(f"export MODAL_TOKEN_ID={shlex.quote(prof['token_id'])} MODAL_TOKEN_SECRET={shlex.quote(prof['token_secret'])}")
PYC
)"
export YETO_ISLAND_HMAC_KEY=${YETO_ISLAND_HMAC_KEY:-$(openssl rand -hex 32)}
[ -n "${MODAL_TOKEN_ID:-}" ] || { echo "abort: no modal token"; exit 65; }
export HOME=$R/home YETO_RUNS_DIR=$R/runs PYTHONPATH=$R/yeto
HEAD_CL=$(cd $R/yeto && /usr/bin/python3 -c "from yeto.launcher import sky_cluster_name; print(sky_cluster_name('$P-head'))")
echo "$APP $HEAD_CL" > $R/resources.txt
if [ "${PLAN_ONLY:-0}" = 1 ]; then
  echo "PLAN_ONLY case=$CASE run=$P app=$APP head=$HEAD_CL steps=$STEPS hard=${HARD}s code=$(cut -c1-8 $R/yeto_sha.txt) threads=$T"
  (cd $R/yeto && $PY - "$ARGS" <<'PY'
import shlex, sys
from yeto import cli, launcher
args = cli.parse_args(shlex.split(sys.argv[1])[1:])
launcher.prepare_launch_args(args)
print("controller:", args.controller, "| preset:", args.rl_sync_preset, "| elastic:", args.rl_elastic, "| steps:", args.total_steps)
print("syncer cmd:", launcher.syncer_command(args, 2))
from yeto.gpu_spec import parse_gpu_spec
for i, s in enumerate(parse_gpu_spec(args.gpu)):
    t = launcher.make_miles_island_task(args, s, i, 2, "127.0.0.1:29400")
    c = launcher.build_modal_island_config(args, s, i, t, "1.2.3.4:29400")
    print(f"island {i}: gpu={c.gpu} n={getattr(c, 'gpus_per_node', None)} test_env={ {k: v for k, v in c.envs.items() if k.startswith('YETO_RL_TEST')} } timeout={getattr(c, 'timeout_s', None)}")
PY
  ) 2>&1 | grep -v "^\s*$" | tail -20
  exit 0
fi
date -u +%FT%TZ > $R/start_utc.txt
cat > $R/teardown.sh <<EOF
cd /tmp; export HOME=/home/michael
timeout 300 $PY -m modal app stop --yes $APP > $R/\${1:-final}-stop.out 2>&1
timeout 900 $SKY down -y $HEAD_CL > $R/\${1:-final}-down.out 2>&1; echo "down rc=\$?" >> $R/\${1:-final}-down.out
sleep 15; timeout 120 $PY -m modal app list --json > $R/\${1:-final}-applist.json 2>&1
timeout 120 $SKY status $HEAD_CL > $R/\${1:-final}-skystatus.txt 2>&1
EOF
setsid nohup bash -c "sleep $WD; bash $R/teardown.sh watchdog; touch $R/WATCHDOG_FIRED" >/dev/null 2>&1 &
echo $! > $R/watchdog.pid
cd $R/yeto
( eval "timeout $HARD $PY -m yeto.cli $(cat $R/args.txt)" 2>&1 | tee $R/launch.stream.log | awk '{ print strftime("%FT%TZ", systime(), 1) " " $0; fflush() }' > $R/launch.ts.log; echo "rc=${PIPESTATUS[0]}" > $R/submit_rc.txt ) &
SUB=$!
# arm the sidecar in each island container once it exists (from this machine; Modal token from the real HOME)
(
  export HOME=/home/michael; armed=""; b64=$(base64 -w0 $R/sidecar.py); tb=$(printf '%s' "$TRIG" | base64 -w0)
  for i in $(seq 1 240); do
    ids=$(cd /tmp && timeout 60 $PY -m modal container list --json 2>/dev/null | /usr/bin/python3 -c "
import json,sys
for c in json.load(sys.stdin):
    if '$APP' in json.dumps(c): print(c.get('Container ID') or c.get('container_id'))" 2>/dev/null)
    for id in $ids; do
      case " $armed " in *" $id "*) continue ;; esac
      (cd /tmp && timeout 120 $PY -m modal container exec $id -- sh -c "mkdir -p /root/yeto-rl /root/yeto-output && echo $b64 | base64 -d > /root/yeto-rl/s17sidecar.py && (nohup python3 /root/yeto-rl/s17sidecar.py \"\$(echo $tb | base64 -d)\" > /root/yeto-output/s17-sidecar.log 2>&1 &) ; sleep 2; ps -eo pid,args | grep [s]17sidecar") >> $R/arm.log 2>&1 \
        && { armed="$armed $id"; echo "armed $id $(date -u +%FT%TZ)" >> $R/arm.log; }
    done
    [ $(echo $armed | wc -w) -ge 2 ] && break
    kill -0 $SUB 2>/dev/null || break; sleep 15
  done
  echo "arm done: $armed" >> $R/arm.log
) &
wait $SUB
JOB=$(grep -oP "submitted: job \K\d+" $R/launch.stream.log 2>/dev/null | head -1); echo "head_job=$JOB" > $R/head_job.txt
if [ -n "$JOB" ]; then
  END=$(( $(date +%s) + 900 ))
  while [ $(date +%s) -lt $END ]; do
    st=$(cd /tmp && HOME=/home/michael timeout 120 $SKY queue $HEAD_CL 2>/dev/null | awk -v j=$JOB '$1==j{print}')
    echo "$st" | grep -qE "RUNNING|PENDING|SETTING_UP|INIT" || break; sleep 30
  done
fi
date -u +%FT%TZ > $R/end_utc.txt
export HOME=/home/michael
S='ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20'
if timeout 60 $SKY status $HEAD_CL 2>/dev/null | grep -q ' UP '; then
  (cd /tmp && timeout 600 $SKY logs $HEAD_CL $JOB --no-follow) > $R/launch.log 2>&1
  timeout 600 $S $HEAD_CL 'cd ~ && tar czf - yeto-output yeto-syncer.log sky_logs 2>/dev/null' > $R/head/head.tgz 2>/dev/null
  (cd $R/head && tar xzf head.tgz 2>/dev/null)
fi
[ -s $R/launch.log ] || cp $R/launch.stream.log $R/launch.log
bash $R/teardown.sh final
mkdir -p $R/tape-direct; (cd /tmp && timeout 600 $PY -m modal volume get --force $TAPEVOL $APP $R/tape-direct) > $R/tape-direct.out 2>&1
for f in $R/tape-direct/*/l*/rank0/s17-elastic-state-*.b64.txt; do [ -f "$f" ] && base64 -d "$f" | tar xz -C $(dirname $f); done
echo "rc=$(cat $R/submit_rc.txt 2>/dev/null)" > $R/rc.txt
kill $(cat $R/watchdog.pid) 2>/dev/null
echo "done $P"

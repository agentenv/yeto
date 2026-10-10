#!/bin/bash
# S19 #10 rl-spot-cost-saving 5.2: anchor island Nebius 1xH100 on-demand + droppable island Modal 1xH100!;
# one active reclaim = `modal container stop` on the droppable island's container (Modal reassigns the input
# to a new container -> rejoin from syncer base). Derived from s19-lpg-neg.sh.
# usage: s19-spot52.sh <run_id>   env: STEPS, HARD, PLAN_ONLY=1, KILL_AFTER (droppable rounds before reclaim)
# (orig header) S19 rerun of S18 LPG 4.3 (on main; teardown pulls head logs FIRST; watchdog = systemd-run --user timer)
# S18 LPG 4.3 (launch-preflight-guards): N12/N14 negative islands on real hardware.
#   Derived from s17-g1-island.sh: Nebius eu-north1 CPU head + 2x Modal H100!, Qwen3-0.6B gsm8k LoRA.
#   Changes vs g1: no killer / stopper / pause; NO --modal-launcher-relaunch / --recover-timeout (a refused
#   island is torn down at once, never relaunched); adds --rl-negative-test-run --rl-island-override $OV;
#   the local thread gate is the launcher's own preflight (first field use) -- `ps` is only logged.
# usage: s18-lpg-neg.sh <run_id>   env: SCHED (legacy|elastic), OV (e.g. 1:rl_lr_schedule=constant), STEPS, HARD, PLAN_ONLY=1
set -u
SCHED=${SCHED:-elastic}; KILL_ISLAND=${KILL_ISLAND:-0}; DELAY=${DELAY:-}; LEASE=${LEASE:-90}; STOP_S=${STOP_S:-$(( LEASE + 30 ))}; P=${1:?run id}; HARD=${HARD:-3600}; STEPS=${STEPS:-40}; KILL_AFTER=${KILL_AFTER:-2}; GPUS=nebius:1xh100@eu-north1,modal:1xh100; KILL_AT_V=${KILL_AT_V:-2}; WD=$(( HARD + 600 )); MODAL_TIMEOUT=$HARD; KILL_DELAY=${KILL_DELAY:-20}
REPO=${REPO:-/home/michael/work/spot52}; B=/home/michael/work/s1-runs; R=$B/$P
SKY=/home/michael/work/gpu-head/venv/bin/sky; PY=/home/michael/work/gpu-head/venv/bin/python
APP=yeto-$P; TAPEVOL=yeto-event-tapes; EXPECT_GPU=H100; HEAD_REGION=nebius/eu-north1
export CARGO_HOME=/home/michael/work/gpu-default-modal/cargo RUSTUP_HOME=/home/michael/work/gpu-default-modal/rustup
export PATH=$CARGO_HOME/bin:$PATH
IMAGE=$(cd "$REPO" && PYTHONPATH=. /usr/bin/python3 -c "import yeto.rl as r; print(r.MILES_NEXT_IMAGE if r.MILES_NEXT_IMAGE.startswith('docker:') else 'docker:'+r.MILES_NEXT_IMAGE)" 2>/dev/null)
[ -n "$IMAGE" ] || { echo "abort: no MILES_NEXT_IMAGE"; exit 65; }
T=$(ps -u michael -L -o pid= | wc -l)  # logged only; the launcher preflight is the gate
pgrep -x raylet >/dev/null && { echo "abort: local Ray running"; exit 3; }
grep -q -- "--rl-elastic-debug-pause" $REPO/yeto/cli.py || { echo "abort: $REPO cli lacks --rl-elastic-debug-pause"; exit 64; }
grep -q 'secrets=secrets or None' $REPO/yeto/cli.py || { echo "abort: $REPO lacks head secrets (s13-g3remote)"; exit 64; }
HEAD_CL=$(cd $REPO && PYTHONPATH=$REPO /usr/bin/python3 -c "from yeto.launcher import sky_cluster_name; print(sky_cluster_name('$P-head'))")
L1=$(cd $REPO && PYTHONPATH=$REPO /usr/bin/python3 -c "
from yeto.launcher import learner_cluster_names; from yeto.gpu_spec import parse_gpu_spec
print(learner_cluster_names('$P', parse_gpu_spec('$GPUS'))[1])")
L0=$(cd $REPO && PYTHONPATH=$REPO /usr/bin/python3 -c "
from yeto.launcher import learner_cluster_names; from yeto.gpu_spec import parse_gpu_spec
print(learner_cluster_names('$P', parse_gpu_spec('$GPUS'))[0])")
if [ "${PLAN_ONLY:-0}" = 1 ]; then R=/tmp/$P-plan; rm -rf $R; fi
[ -e $R/start_utc.txt ] && { echo "abort: $R already used"; exit 67; }
mkdir -p $R/home $R/runs $R/yeto $R/head
# client HOME: everything sky/yeto need EXCEPT ~/.modal.toml (Modal goes by env -> sky secret)
for d in .sky .ssh .nebius .cache .huggingface .config; do [ -e /home/michael/$d ] && ln -sfn /home/michael/$d $R/home/$d; done
git -C $REPO archive ${SHA:-HEAD} | tar x -C $R/yeto
cp /home/michael/work/gpu-default-modal/yeto/gsm8k_reward.py $R/yeto/; touch $R/yeto/yeto-rl-echo-events; [ -n "${SPEC:-}" ] && cp $SPEC $R/yeto/g1-spec.json
MODEL="--model Qwen/Qwen3-0.6B --model-revision c1899de289a04d12100db370d81485cdf75e47ca"; LORA="--tuning lora --lora-r 16 --lora-targets all-linear"
MODALX="--modal-retries 0 --modal-timeout-s $MODAL_TIMEOUT --modal-tape-volume $TAPEVOL"
ARGS="launch --controller head --syncer-region $HEAD_REGION --training-mode rl --rl-engine ports --rl-island-scheduling $SCHED $( [ "$SCHED" = elastic ] && echo "--rl-soft-deadline-s ${SOFT:-180} ")${DELAY:+--rl-elastic-debug-delay-s $DELAY }$( [ "${PAUSE:-0}" = 1 ] && echo "--rl-elastic-debug-pause ${PAUSE_ISLAND:-1}:${PAUSE_AT_V:-1}:$STOP_S ")${SPEC:+--rl-algorithm-spec g1-spec.json }$( [ "$SCHED" = elastic ] && echo "--rl-island-lease-s $LEASE ")--rl-island-role 1:droppable --on-demand --preflight-threads wait --rl-stall-timeout 2400 --rl-observe-timeline --gpu $GPUS --cluster-prefix $P $MODALX --rl-image $IMAGE $MODEL --data zhuzilin/gsm8k --data-revision 0cbd9f31d91ac21a7613dcbc7fef992adac459ae --reward-function gsm8k_reward:score $LORA --fragments 1 --pipeline 1 --rollout-batch-size 4 --n-samples-per-prompt 8 --rollout-max-response-len 384 --seq-len 1024 --inner-lr 1e-5 --seed 17 --apply-chat-template-kwargs '{\"enable_thinking\": false}' --trust-remote-code --total-steps $STEPS"
git -C $REPO rev-parse ${SHA:-HEAD} > $R/yeto_sha.txt; echo "$ARGS" > $R/args.txt; echo "$APP $HEAD_CL $L0" > $R/resources.txt
# secrets: process env only (never echoed, never written)
eval "$(/usr/bin/python3 - <<'PYC'
import json, shlex, tomllib
t = tomllib.load(open("/home/michael/.modal.toml", "rb"))
prof = next((v for v in t.values() if isinstance(v, dict) and v.get("active")), None) or next(v for v in t.values() if isinstance(v, dict))
print(f"export MODAL_TOKEN_ID={shlex.quote(prof['token_id'])} MODAL_TOKEN_SECRET={shlex.quote(prof['token_secret'])}")
PYC
)"
# island HMAC key (0.20): process env only -> shipped as sky/Modal secret; never written or echoed
export YETO_ISLAND_HMAC_KEY=${YETO_ISLAND_HMAC_KEY:-$(openssl rand -hex 32)}
[ -n "${MODAL_TOKEN_ID:-}" ] || { echo "abort: no modal token"; exit 65; }
export HOME=$R/home YETO_RUNS_DIR=$R/runs PYTHONPATH=$R/yeto
if [ "${PLAN_ONLY:-0}" = 1 ]; then
  echo "PLAN_ONLY run=$P app=$APP head=$HEAD_CL sched=$SCHED steps=$STEPS hard=${HARD}s wd=${WD}s code=$REPO@$(cat $R/yeto_sha.txt|cut -c1-8) ps_threads=$T"
  (cd $R/yeto && eval "$PY -m yeto.cli $(cat $R/args.txt) --dry-run") > $R/plan.json 2> $R/plan.err; echo "dry-run rc=$?"; tail -3 $R/plan.err
  /usr/bin/python3 -c "import json,sys;p=json.load(open(sys.argv[1]));[print(i['learner_id'],i.get('cloud'),i['memory_estimate'].get('peak_gib'),i['memory_estimate'].get('fits'),[t for t in i['learner_command'].split(' --') if t.startswith(('rl-lr','rl-max-policy','rl-island-sched'))]) for i in p['island_requests']]" $R/plan.json
  exit 0
fi
date -u +%FT%TZ > $R/start_utc.txt
# teardown of THIS run only (Modal app + head cluster; never `sky down -a`)
cat > $R/teardown.sh <<EOF
cd /tmp; export HOME=/home/michael
S='ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20'
JOB=\$(grep -oP 'head_job=\K\d+' $R/head_job.txt 2>/dev/null)
# 1) pull head evidence BEFORE anything is stopped (launch.log, syncer log, output, queue)
if timeout 60 $SKY status $HEAD_CL 2>/dev/null | grep -q ' UP '; then
  timeout 600 $SKY logs $HEAD_CL \$JOB --no-follow > $R/\${1:-final}-launch.log 2>&1
  timeout 300 $SKY queue $HEAD_CL > $R/head/queue.txt 2>&1
  timeout 600 \$S $HEAD_CL 'cd ~ && tar czf - yeto-output yeto-syncer.log sky_logs 2>/dev/null' > $R/head/head-\${1:-final}.tgz 2>/dev/null
  (cd $R/head && tar xzf head-\${1:-final}.tgz 2>/dev/null)
  timeout 60 \$S $HEAD_CL 'hostname; nproc; free -g; ss -ltnp 2>/dev/null|grep -E ":29400"' > $R/head/host.txt 2>&1
  [ -s $R/launch.log ] || cp $R/\${1:-final}-launch.log $R/launch.log
fi
echo "pulled \$(date -u +%FT%TZ) syncer_log_bytes=\$(stat -c %s $R/head/yeto-syncer.log 2>/dev/null || echo 0)" > $R/\${1:-final}-pull.out
# 2) stop Modal app, 3) down THIS head only (never -a)
timeout 300 $PY -m modal app stop --yes $APP > $R/\${1:-final}-stop.out 2>&1
# 3) learners of a head run are downed FROM the head, then the head (yeto down does both, in order; never sky down -a)
(cd $R/yeto && YETO_RUNS_DIR=$R/runs PYTHONPATH=$R/yeto timeout 1500 $PY -m yeto.cli down $P) > $R/\${1:-final}-down.out 2>&1; echo "yeto down rc=\$?" >> $R/\${1:-final}-down.out
# 4) fallback (yeto down could not reach the head's sky on 10-09 try a): delete THIS run's Nebius anchor instance by name prefix via the cloud API, then down THIS head locally
NP=project-e00eqrj3pr00622zrgdeyc
for id in \$(timeout 120 nebius compute instance list --parent-id \$NP --format json 2>/dev/null | /usr/bin/python3 -c "import json,sys;[print(i['metadata']['id']) for i in json.load(sys.stdin).get('items',[]) if i['metadata']['name'].startswith('$P-l0')]"); do echo "fallback delete \$id" >> $R/\${1:-final}-down.out; timeout 600 nebius compute instance delete --id \$id >> $R/\${1:-final}-down.out 2>&1; done
timeout 900 $SKY down -y $HEAD_CL >> $R/\${1:-final}-down.out 2>&1; echo "sky down head rc=\$?" >> $R/\${1:-final}-down.out
sleep 30
timeout 120 nebius compute instance list --parent-id project-e00eqrj3pr00622zrgdeyc --format json 2>/dev/null | /usr/bin/python3 -c "import json,sys;d=json.load(sys.stdin);print([ (i['metadata']['name'],i.get('status',{}).get('state')) for i in d.get('items',[])])" > $R/\${1:-final}-nebius-instances.txt 2>&1
sleep 15; timeout 120 $PY -m modal app list --json > $R/\${1:-final}-applist.json 2>&1
timeout 120 $SKY status $HEAD_CL > $R/\${1:-final}-skystatus.txt 2>&1
EOF
systemd-run --user --unit=$P-wd --on-active=${WD}s /bin/bash -c "bash $R/teardown.sh watchdog; touch $R/WATCHDOG_FIRED" > $R/watchdog.out 2>&1
echo "$P-wd.timer" > $R/watchdog.pid
echo "ps_threads_at_start=$T" > $R/threads-ps.txt
cd $R/yeto
( eval "timeout $HARD $PY -m yeto.cli $(cat $R/args.txt)" 2>&1 | tee $R/launch.stream.log | awk '{ print strftime("%FT%TZ", systime(), 1) " " $0; fflush() }' > $R/launch.ts.log; echo "rc=${PIPESTATUS[0]}" > $R/submit_rc.txt ) &
SUB=$!
# once the controller job is submitted: start the killer job on the VM
JOB=""
for i in $(seq 1 360); do
  JOB=$(grep -oP "submitted: job \K\d+" $R/launch.stream.log 2>/dev/null | head -1); [ -n "$JOB" ] && break
  kill -0 $SUB 2>/dev/null || break; sleep 10
done
echo "head_job=$JOB" > $R/head_job.txt
# active reclaim (once): anchor island (l0) trained >=1 round AND droppable (l1) trained >=KILL_AFTER rounds
( cd /tmp; export HOME=/home/michael
  for i in $(seq 1 600); do
    kill -0 $SUB 2>/dev/null || { echo "$(date -u +%FT%TZ) launch ended before reclaim" >> $R/reclaim.log; exit 0; }
    a=$(grep -a -- '-l0-' $R/launch.stream.log | grep -ac '"event":"rl_round_trained"')
    d=$(grep -a -- '-l1-' $R/launch.stream.log | grep -ac '"event":"rl_round_trained"')
    if [ "$a" -ge 1 ] && [ "$d" -ge $KILL_AFTER ]; then
      CID=$(grep -a -oP '\[modal-island 1\] rank 0 container \K\S+' $R/launch.stream.log | tail -1)
      echo "$(date -u +%FT%TZ) trigger anchor_rounds=$a droppable_rounds=$d container=$CID outer=$(grep -a -oP 'kind="outer_step" outer_version=\K\d+' $R/launch.stream.log | tail -1)" >> $R/reclaim.log
      [ -n "$CID" ] && timeout 120 $PY -m modal container stop --yes $CID >> $R/reclaim.log 2>&1; echo "stop rc=$? $(date -u +%FT%TZ)" >> $R/reclaim.log
      exit 0
    fi
    sleep 10
  done ) &
wait $SUB
# the stream may drop early: wait (bounded) for the head job to reach a terminal state
if [ -n "$JOB" ]; then
  END=$(( $(date +%s) + HARD ))
  while [ $(date +%s) -lt $END ]; do
    st=$(cd /tmp && timeout 120 $SKY queue $HEAD_CL 2>/dev/null | awk -v j=$JOB '$1==j{print}')
    echo "$st" | grep -qE "RUNNING|PENDING|SETTING_UP|INIT" || break; sleep 30
  done
fi
date -u +%FT%TZ > $R/end_utc.txt
export HOME=/home/michael
bash $R/teardown.sh final
[ -s $R/launch.log ] || cp $R/launch.stream.log $R/launch.log
# rc: the head job's terminal status (SUCCEEDED -> 0)
grep -E "^ *$JOB " $R/head/queue.txt 2>/dev/null | grep -q SUCCEEDED && echo "rc=0" > $R/rc.txt || echo "rc=1 ($(grep -E "^ *$JOB " $R/head/queue.txt 2>/dev/null | tr -s ' ' | cut -c1-120))" > $R/rc.txt
AID=$(/usr/bin/python3 -c "import json,sys;print(next((a['app_id'] for a in json.load(open(sys.argv[1])) if a.get('description')==sys.argv[2]),''))" $R/final-applist.json $APP 2>/dev/null); [ -n "$AID" ] && (cd /tmp && timeout 180 $PY -m modal app logs $AID > $R/modal-app-logs.txt 2>&1)
mkdir -p $R/tape-direct; timeout 600 $PY -m modal volume get --force $TAPEVOL $APP $R/tape-direct > $R/tape-direct.out 2>&1
systemctl --user stop $P-wd.timer 2>/dev/null
echo "done $P $(cat $R/rc.txt)"

#!/bin/bash
# Miles eval cell E3: two synced trainer islands (elastic OuterStep; head + syncer on a
# Nebius CPU VM) feed one sampler island (a separate no-sync launch) through merged
# releases in one Modal Volume store. Copied from s1-runs/s25-c3/run.sh (S25 C3, passed in
# run s25-c3-20261020b); parameterised by the eval driver (tools/eval/miles.py).
# env: P (run id), R (raw dir), REPO, GPU (modal gpu, e.g. a10g), HARD, STEPS, STOREVOL,
#      PY, SKY, PLAN_ONLY=1 (dry-run only)
set -u
: "${P:?}" "${R:?}" "${REPO:?}" "${PY:?}" "${SKY:?}"
HARD=${HARD:-2700}; WD=$(( HARD + 300 )); STEPS=${STEPS:-4}; SROUNDS=$(( 2 * STEPS + 6 )); GPU=${GPU:-a10g}
# launcher thread preflight stays on; loosen only with explicit flags, e.g.
# PREFLIGHT="--preflight-thread-start 2950" (the launcher logs and records the loosened values)
PREFLIGHT=${PREFLIGHT:-}
SHA=${SHA:-HEAD}; MODAL="$PY -m modal"
APP=yeto-$P; SP=$P-s; SAPP=yeto-$SP; TAPEVOL=yeto-event-tapes; STOREVOL=${STOREVOL:-yeto-eval-store}; HEAD_REGION=nebius/eu-north1
export CARGO_HOME=/home/michael/work/gpu-default-modal/cargo RUSTUP_HOME=/home/michael/work/gpu-default-modal/rustup
export PATH=$CARGO_HOME/bin:$PATH
T=$(ps -u michael -L --no-headers | wc -l)
[ "${PLAN_ONLY:-0}" = 1 ] || [ "$T" -lt 3000 ] || { echo "abort: $T user threads (>= 3000)"; exit 3; }
pgrep -x raylet >/dev/null && { echo "abort: local Ray running"; exit 3; }
HEAD_CL=$(cd $REPO && PYTHONPATH=$REPO /usr/bin/python3 -c "from yeto.launch.sky_ops import sky_cluster_name; print(sky_cluster_name('$P-head'))")
[ -e $R/start_utc.txt ] && { echo "abort: $R already used"; exit 67; }
mkdir -p $R/runs $R/sruns $R/head $R/tapes $R/store
# a real clone: the code pack needs a git checkout. HOME stays the real one: the remote sky API
# server cannot unpack a file mount reached through a symlinked HOME dir (run a, FileExistsError).
git clone -q --no-local $REPO $R/yeto && git -C $R/yeto checkout -q $SHA
touch $R/yeto/yeto-rl-echo-events
cp $REPO/yeto/rl/gsm8k_reward.py $R/yeto/gsm8k_reward.py
(cd $R/yeto && git add -f gsm8k_reward.py yeto-rl-echo-events && git -c user.email=eval@yeto -c user.name=t-eval commit -qm 'run inputs')
MODEL="--model Qwen/Qwen3-0.6B --model-revision c1899de289a04d12100db370d81485cdf75e47ca"
DATA="--data zhuzilin/gsm8k --data-revision 0cbd9f31d91ac21a7613dcbc7fef992adac459ae"
LORA="--tuning lora --lora-r 16"
MODALX="--modal-gpu-exact --modal-retries 0 --modal-cpu 8 --modal-memory 64 --modal-tape-volume $TAPEVOL"
STORE="--rl-checkpoint-store modal-volume://$STOREVOL/$P"
COMMON="--training-mode rl --rl-engine ports $MODALX $MODEL $DATA $LORA $STORE --rl-algorithm-spec examples/rl_algorithms/decoupled-ppo-c2.json --rl-max-policy-age 2 --rl-observe-timeline --reward-function gsm8k_reward:score --fragments 1 --pipeline 1 --rollout-batch-size 32 --n-samples-per-prompt 4 --rollout-max-response-len 1024 --seq-len 2048 --inner-lr 1e-5 --seed 17 --apply-chat-template-kwargs '{\"enable_thinking\": false}' --trust-remote-code"
TARGS="launch --controller head --syncer-region $HEAD_REGION $PREFLIGHT --rl-stall-timeout 2400 --on-demand --no-island-relaunch --modal-timeout-s $HARD --cluster-prefix $P --gpu modal:1x$GPU,modal:1x$GPU --rl-island-scheduling elastic --rl-sample-role trainer --rl-sample-wait-s 900 $COMMON --total-steps $STEPS"
SARGS="launch $PREFLIGHT --controller local --cluster-prefix $SP --gpu modal:1x$GPU --rl-single-island-no-sync --rl-sample-role sampler --rl-sampler-max-backlog-rounds 4 $COMMON --total-steps $SROUNDS"
git -C $REPO rev-parse $SHA > $R/yeto_sha.txt; echo "$TARGS" > $R/args.txt; echo "$SARGS" > $R/sargs.txt
echo "$APP $SAPP $HEAD_CL" > $R/resources.txt
eval "$(/usr/bin/python3 - <<'PYC'
import shlex, tomllib
t = tomllib.load(open("/home/michael/.modal.toml", "rb"))
prof = next((v for v in t.values() if isinstance(v, dict) and v.get("active")), None) or next(v for v in t.values() if isinstance(v, dict))
print(f"export MODAL_TOKEN_ID={shlex.quote(prof['token_id'])} MODAL_TOKEN_SECRET={shlex.quote(prof['token_secret'])}")
PYC
)"
export YETO_ISLAND_HMAC_KEY=${YETO_ISLAND_HMAC_KEY:-$(openssl rand -hex 32)}
[ -n "${MODAL_TOKEN_ID:-}" ] || { echo "abort: no modal token"; exit 65; }
export PYTHONPATH=$R/yeto
if [ "${PLAN_ONLY:-0}" = 1 ]; then
  echo "PLAN_ONLY run=$P apps=$APP,$SAPP head=$HEAD_CL steps=$STEPS sampler_rounds=$SROUNDS hard=${HARD}s code=$(cut -c1-8 $R/yeto_sha.txt) threads=$T"
  (cd $R/yeto && eval "YETO_RUNS_DIR=$R/runs $PY -m yeto.cli $TARGS --dry-run") > $R/plan.json 2> $R/plan.err; echo "trainer dry-run rc=$?"; tail -3 $R/plan.err
  (cd $R/yeto && eval "YETO_RUNS_DIR=$R/sruns $PY -m yeto.cli $SARGS --dry-run") > $R/splan.json 2> $R/splan.err; echo "sampler dry-run rc=$?"; tail -3 $R/splan.err
  exit 0
fi
date -u +%FT%TZ > $R/start_utc.txt
cat > $R/teardown.sh <<EOF
cd /tmp; export HOME=/home/michael
S='ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20'
JOB=\$(grep -oP 'head_job=\K\d+' $R/head_job.txt 2>/dev/null)
if timeout 60 $SKY status $HEAD_CL 2>/dev/null | grep -q ' UP '; then
  timeout 600 $SKY logs $HEAD_CL \$JOB --no-follow > $R/\${1:-final}-launch.log 2>&1
  timeout 300 $SKY queue $HEAD_CL > $R/head/queue.txt 2>&1
  timeout 600 \$S $HEAD_CL 'cd ~ && tar czf - yeto-output yeto-syncer.log 2>/dev/null' > $R/head/head-\${1:-final}.tgz 2>/dev/null
  (cd $R/head && tar xzf head-\${1:-final}.tgz 2>/dev/null)
fi
for a in $APP $SAPP; do case \$a in yeto-$P|yeto-$P-s) timeout 300 $MODAL app stop --yes \$a;; *) echo REFUSE-STOP \$a;; esac; done > $R/\${1:-final}-stop.out 2>&1
timeout 900 $SKY down -y $HEAD_CL > $R/\${1:-final}-down.out 2>&1; echo "down rc=\$?" >> $R/\${1:-final}-down.out
sleep 15; timeout 120 $MODAL app list --json > $R/\${1:-final}-applist.json 2>&1
timeout 120 $SKY status $HEAD_CL > $R/\${1:-final}-skystatus.txt 2>&1
EOF
echo "ps_threads_at_start=$T" > $R/threads-ps.txt
# container tape + NVML puller (both apps)
( while sleep 20; do c=$(cd /tmp && HOME=/home/michael timeout 60 $MODAL container list --json 2>/dev/null | python3 -c "import json,sys; print(' '.join(x['container_id'] for x in json.load(sys.stdin) if x.get('app_name') in ('$APP','$SAPP')))" 2>/dev/null)
    echo "$(date -u +%FT%TZ) $c" >> $R/containers.log
    for x in $c; do
      (cd /tmp && HOME=/home/michael timeout 60 $MODAL container exec $x -- sh -c "cat /root/yeto-output/rl-island-*.jsonl 2>/dev/null") > $R/tapes/.$x 2>/dev/null && [ -s $R/tapes/.$x ] && grep -q '"event"' $R/tapes/.$x && mv $R/tapes/.$x $R/tapes/$x.jsonl
      (cd /tmp && HOME=/home/michael timeout 30 $MODAL container exec $x -- sh -c "nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,power.draw --format=csv,noheader") 2>/dev/null | sed "s/^/$x,/" >> $R/nvml.csv; done; done ) 9>&- &
PULLER=$!
cd $R/yeto
exec 9>/home/michael/work/infra-drafts/GPU-LAUNCH.lock; flock 9
date -u +%FT%TZ > $R/lock_utc.txt
systemd-run --user --unit=$P-wd --on-active=${WD}s /bin/bash -c "bash $R/teardown.sh watchdog; touch $R/WATCHDOG_FIRED" > $R/watchdog.out 2>&1
( eval "YETO_RUNS_DIR=$R/runs timeout $HARD $PY -m yeto.cli $TARGS" 2>&1 | tee $R/launch.stream.log | awk '{ print strftime("%FT%TZ", systime(), 1) " " $0; fflush() }' > $R/launch.ts.log; echo "rc=${PIPESTATUS[0]}" > $R/submit_rc.txt ) 9>&- &
SUB=$!
JOB=""
for i in $(seq 1 360); do
  JOB=$(grep -oP "submitted: job \K\d+" $R/launch.stream.log 2>/dev/null | head -1); [ -n "$JOB" ] && break
  kill -0 $SUB 2>/dev/null || break; sleep 10
done
echo "head_job=$JOB" > $R/head_job.txt
for i in $(seq 1 90); do [ "$(grep -c 'modal-island [01]\] requested' $R/launch.stream.log 2>/dev/null)" -ge 2 ] && break; kill -0 $SUB 2>/dev/null || break; sleep 10; done
# the sampler starts once both trainer islands are requested (it waits for the first release)
( eval "YETO_RUNS_DIR=$R/sruns timeout $HARD $PY -m yeto.cli $SARGS" 2>&1 | awk '{ print strftime("%FT%TZ", systime(), 1) " " $0; fflush() }' > $R/slaunch.ts.log; echo "rc=${PIPESTATUS[0]}" > $R/ssubmit_rc.txt ) 9>&- &
SSUB=$!
sleep 60; exec 9>&-
( cd /tmp; HOME=/home/michael timeout $HARD $MODAL app logs --timestamps $APP > $R/modal-live-logs.txt 2>&1 ) 9>&- &
wait $SUB
if [ -n "$JOB" ]; then
  END=$(( $(date +%s) + HARD ))
  while [ $(date +%s) -lt $END ]; do
    st=$(cd /tmp && HOME=/home/michael timeout 120 $SKY queue $HEAD_CL 2>/dev/null | awk -v j=$JOB '$1==j{print}')
    echo "$st" | grep -qE "RUNNING|PENDING|SETTING_UP|INIT" || break; sleep 30
  done
fi
date -u +%FT%TZ > $R/end_utc.txt
kill $PULLER 2>/dev/null
export HOME=/home/michael
bash $R/teardown.sh final
kill $SSUB 2>/dev/null
grep -E "^ *$JOB " $R/head/queue.txt 2>/dev/null | grep -q SUCCEEDED && echo "rc=0" > $R/rc.txt || echo "rc=1 ($(grep -E "^ *$JOB " $R/head/queue.txt 2>/dev/null | tr -s ' ' | cut -c1-120))" > $R/rc.txt
for a in $APP $SAPP; do
  AID=$(/usr/bin/python3 -c "import json,sys;print(next((a['app_id'] for a in json.load(open(sys.argv[1])) if a.get('description')==sys.argv[2]),''))" $R/final-applist.json $a 2>/dev/null)
  [ -n "$AID" ] && (cd /tmp && timeout 180 $MODAL app logs --timestamps $AID > $R/modal-app-logs-$a.txt 2>&1)
  mkdir -p $R/tape-direct/$a; (cd /tmp && timeout 600 $MODAL volume get --force $TAPEVOL $a $R/tape-direct/$a) >> $R/tape-direct.out 2>&1
done
(cd /tmp && timeout 900 $MODAL volume get --force $STOREVOL $P/policy-releases $R/store; timeout 900 $MODAL volume get --force $STOREVOL $P/sample-buffer/index.jsonl $R/store; timeout 300 $MODAL volume get --force $STOREVOL $P/sample-buffer/lanes $R/store; timeout 300 $MODAL volume ls --json $STOREVOL $P) > $R/store.out 2>&1
systemctl --user stop $P-wd.timer 2>/dev/null
echo "done $P $(cat $R/rc.txt)"

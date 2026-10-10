#!/bin/bash
# S19 batch-3 #7/#8 (from s18-aru3-ab.sh): agentic-rollout-utilization 5.5 criterion 5 + rl-spot-cost-saving 4.4.
# M1 arm only (OVS=12, limit 1), Qwen3.5-4B 12288/6144, TB2 46 tasks, codex_openenv, 1xH200 per island, 8 rounds.
# 4.1 in-flight save on (YETO_SPOT_INFLIGHT_SAVE_DIR on the tape volume); KILL_AT=k: modal container stop
# KILL_DELAY s after 'Suspended agentic groups at rollout k' (one real reclaim).
# ARU-3 arm legend kept for reference:
#   MB: OVS=6 (no over-sampling), limit 0      M0: OVS=12, limit 0 (cut-off, discard)
#   M1: OVS=12, --rl-max-policy-age 1 (cut-off, suspend between model turns, continue next round)
# All arms: same code (worktree s18-aru-stage3), same image ddce209-2fa8801 (Miles ddce20992),
# same algorithm spec (m1-spec + execution.max_policy_staleness 1), sandbox idle timeout 900 s.
# usage: s18-aru3-ab.sh run_id   env: PLAN_ONLY=1, OVS, LIMIT
set -u
P=${1:-s17-n17-codex-20261009a}; HARD=${HARD:-3000}; WD=$(( HARD + 240 )); MODAL_TIMEOUT=$HARD
QUEUE_MAX=${QUEUE_MAX:-2400}
REPO=${REPO:-/home/michael/work/s19-agentic5-run}; B=/home/michael/work/s1-runs; R=$B/$P
PY=/home/michael/work/gpu-head/venv/bin/python
CB=${CB:-/home/michael/work/s1-runs/s17-n17-bundle}
APP=yeto-$P; TAPEVOL=yeto-event-tapes; EXPECT_GPU=H200; GPUS=1
CONTAINER_MARK="\[modal-island 0\] requested ${EXPECT_GPU}!*:${GPUS}, got"
MEM_GIB=${MEM_GIB:-128}; CPUS=${CPUS:-16}
PROFILE=${PROFILE:-qwen35}; TITO=${TITO:-qwen35}; STEPS=${STEPS:-8}; OVS=${OVS:-12}
RBS=${RBS:-6}; NSAMP=${NSAMP:-4}; RESP=${RESP:-6144}; SEQ=${SEQ:-12288}; LR=${LR:-1e-5}
DATA_FILE=${DATA_FILE:-$CB/data/tb2-train.jsonl}
CAP=${CAP:-6}
PRICE=$(python3 -c "print(round($GPUS*0.001261*3600 + $CPUS*0.0472 + $MEM_GIB*0.00000222*3600, 2))")
WORST=$(python3 -c "print(round($PRICE*$WD/3600,2))")
python3 -c "import sys; sys.exit(0 if $WORST <= $CAP else 1)" || { echo "abort: worst case \$$WORST exceeds cap \$$CAP"; exit 69; }
IMAGE=$(cd "$REPO" && PYTHONPATH=. /usr/bin/python3 -c "import yeto.rl as r; print(r.MILES_NEXT_IMAGE if r.MILES_NEXT_IMAGE.startswith('docker:') else 'docker:'+r.MILES_NEXT_IMAGE)" 2>/dev/null)
case "$IMAGE" in *fa2413be4c4fcf066437f946365f01392d6f884002f8fa19c770c12c08f2acf0*) ;; *) echo "abort: image $IMAGE is not fa2413be"; exit 65;; esac
T=$(ps -u michael -L -o pid= | wc -l); if [ "$T" -ge ${THREAD_MAX:-2800} ]; then echo "abort: $T user threads (max ${THREAD_MAX:-2800})"; [ "${PLAN_ONLY:-0}" = 1 ] || exit 3; fi
pgrep -u michael -x "raylet|gcs_server" >/dev/null && { echo "abort: local Ray running"; [ "${PLAN_ONLY:-0}" = 1 ] || exit 3; }
grep -q "_install_inflight_save" $REPO/yeto/rl/adapters/miles/entry.py && grep -q "SCORING_ROUTING_KEY" $REPO/yeto/rl/adapters/miles/carry_over.py && grep -q "agentic-suspend-between-turns" $REPO/yeto/rl/adapters/miles/policy_age.py && grep -q "^async def suspend" $REPO/yeto/rl/harness/codex/codex_openenv_subprocess_agent_function.py || { echo "abort: $REPO lacks ARU-3/4.1/#176 code"; exit 64; }
[ -z "$(git -C $REPO status --porcelain)" ] || { echo "abort: $REPO has uncommitted changes"; exit 64; }
[ "$(wc -l < $DATA_FILE)" = 46 ] || { echo "abort: data is not the 46-task train set"; exit 64; }
[ "$(ls $CB/codex/tb2-tasks | wc -l)" = 46 ] && [ -f "$CB/hmac.key" ] || { echo "abort: bundle $CB incomplete"; exit 64; }
export YETO_CODEX_BUNDLE_DIR=$CB/codex
export YETO_HARNESS_ENVIRONMENT_PROVIDER=yeto.cloud.modal_reward_env:modal_provider
export YETO_HARNESS_TB2_TASKS_DIR=/opt/yeto/codex/tb2-tasks
export YETO_HARNESS_TB2_SANDBOX_TTL_S=1800
export SECRLENV_MAX_TURNS=${SECRLENV_MAX_TURNS:-12}
export YETO_HARNESS_TB2_IDLE_TIMEOUT_S=900  # > suspension survival limit (600 s) + one model turn
export OPENENV_RUN_ID=$P
unset YETO_CODEX_COMPACTION_ENABLED
MODEL="--model Qwen/Qwen3.5-4B --model-revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
DATA="--data $DATA_FILE --rl-allow-local-data --reward-function yeto.rl.harness.codex.tbench_reward:reward_func"
CODEX="--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate --custom-agent-function-path yeto.rl.harness.codex.codex_openenv_subprocess_agent_function.run --codex-backend-profile $PROFILE --codex-reasoning-effort xhigh --use-session-server --tito-model $TITO --tito-allowed-append-roles tool user --agent-max-seq-len $SEQ"
LORA="--tuning lora --lora-r 16 --lora-targets attention"
HYP="--rollout-batch-size $RBS --n-samples-per-prompt $NSAMP --rollout-max-response-len $RESP --seq-len $SEQ --inner-lr $LR --seed ${SEED:-17} --rl-algorithm-spec $B/s19-agentic5/aru3-m1-age1.spec.json --rl-lr-schedule constant --over-sampling-batch-size $OVS --rl-max-policy-age ${LIMIT:-1}"
OBS="--rl-observe-timeline --rl-resource-sample-interval 10"
MODALX="--modal-gpu-exact --modal-retries 0 --modal-timeout-s $MODAL_TIMEOUT --modal-tape-volume $TAPEVOL --modal-memory-gib $MEM_GIB --modal-cpu $CPUS --modal-env YETO_MODAL_HOSTMEM_SAMPLE_S=10 --modal-env YETO_SPOT_INFLIGHT_SAVE_DIR=/yeto-tape/inflight-$P --modal-env YETO_SPOT_INFLIGHT_SAVE_VOLUME=$TAPEVOL --modal-sandbox-secret yeto-sandbox-modal"
ARGS="launch --controller local --training-mode rl --rl-engine ports --rl-single-island-no-sync --on-demand --gpu modal:${GPUS}xh200 --cluster-prefix $P --no-island-relaunch $MODALX --rl-image $IMAGE $MODEL $DATA $CODEX $LORA --fragments 1 --pipeline 1 $HYP $OBS --trust-remote-code --total-steps $STEPS ${EXTRA:-}"
if [ "${PLAN_ONLY:-0}" = 1 ]; then
  echo "PLAN_ONLY run=$P app=$APP gpu=${EXPECT_GPU}x$GPUS image=$IMAGE steps=$STEPS batch=${RBS}x${NSAMP} seq=$SEQ resp=$RESP hard=${HARD}s code=$REPO@$(git -C $REPO rev-parse --short HEAD) threads=$T price=\$$PRICE/h worst=\$$WORST"
  echo "$ARGS"
  (cd $REPO && HOME=/home/michael PYTHONPATH=$REPO:/tmp/s15-noray \
     TBENCH_REWARD_HMAC_KEY=$(cat $CB/hmac.key) MODAL_TOKEN_ID=plan MODAL_TOKEN_SECRET=plan CB=$CB OVS=$OVS LIMIT=${LIMIT:-1} \
     $PY - "$ARGS" <<'PY'
import os, shlex, sys
from yeto import launcher
from yeto.cli import parse_args
from yeto.gpu_spec import parse_gpu_spec
args = parse_args(shlex.split(sys.argv[1])[1:])
args.source_sha256 = "c" * 64; args.reward_sha256 = "d" * 64
launcher._prepare_rl_args(args)
print("codex profile:", args.codex_backend_profile, "| tito:", args.tito_model, "| chat kwargs:", args.apply_chat_template_kwargs,
      "| over_sampling:", args.over_sampling_batch_size, "| lr schedule:", args.rl_lr_schedule)
specs = parse_gpu_spec(args.gpu)
launcher.check_cloud_prerequisites(specs, args=args)
task = launcher.make_miles_island_task(args, specs[0], 0, 1, "none")
cfg = launcher.build_modal_island_config(args, specs[0], 0, task, "none")
cfg.validate()
print("modal cfg:", {k: getattr(cfg, k) for k in ("app_name", "gpu_request", "gpu_exact", "retries", "timeout_s", "cpu_request", "memory_request_mib", "image_ref")})
learner = next(l for l in task.run.splitlines() if "island_entry" in l)
print("learner flags:", learner.split("island_entry", 1)[1].strip()[:2500])
assert "--rl-lr-schedule constant" in learner and f"--over-sampling-batch-size {os.environ['OVS']}" in learner, "lr/over-sampling flag not forwarded"
assert os.environ['LIMIT'] == '0' or f"--rl-max-policy-age {os.environ['LIMIT']}" in learner, "policy-age limit not forwarded"
print("rl_max_policy_age:", args.rl_max_policy_age)
from pathlib import Path as _P
from yeto.rl.harness.codex import tb2_provider as _tb2
rows = _tb2.preflight_task_prompts(_P(args.data if isinstance(args.data, str) else args.data[0]), _P(os.environ["CB"]) / "codex" / "tb2-tasks")
print("task prompts resolved:", len(rows))
PY
  ) || exit 66
  exit 0
fi

export TBENCH_REWARD_HMAC_KEY=$(cat $CB/hmac.key)
export MODAL_TOKEN_ID=${MODAL_TOKEN_ID:-$(grep -m1 token_id ~/.modal.toml | cut -d'"' -f2)}
export MODAL_TOKEN_SECRET=${MODAL_TOKEN_SECRET:-$(grep -m1 token_secret ~/.modal.toml | cut -d'"' -f2)}
[ -e $R/start_utc.txt ] && { echo "abort: $R already used"; exit 67; }
grep -q "$P" /home/michael/work/infra-drafts/gpu-spend.md || { echo "abort: $P not pre-registered in gpu-spend.md"; exit 68; }
mkdir -p $R/home $R/runs $R/yeto
for d in .modal.toml .sky .cache; do ln -sfn /home/michael/$d $R/home/$d; done
[ -e /home/michael/.huggingface ] && ln -sfn /home/michael/.huggingface $R/home/.huggingface
git -C $REPO archive ${SHA:-HEAD} | tar x -C $R/yeto; touch $R/yeto/yeto-rl-echo-events
git -C $REPO rev-parse ${SHA:-HEAD} > $R/yeto_sha.txt; echo "$ARGS" > $R/args.txt; echo $APP > $R/app.txt
env | grep -E "^(YETO_HARNESS|YETO_CODEX|SECRLENV|OPENENV)" | sed 's/=.*//' > $R/env.txt
echo "price_per_h=$PRICE hard=$HARD wd=$WD worst=$WORST cap=$CAP" > $R/cost-params.txt
# Watchdog (container-time): wait for the first container line (<= QUEUE_MAX+300 s), then WD s, then stop the app.
cat > $R/watchdog.sh <<WDEOF
t0=\$(date +%s)
until [ -f $R/launch.log ] && grep -q '$CONTAINER_MARK' $R/launch.log; do
  [ \$(( \$(date +%s) - t0 )) -ge $(( QUEUE_MAX + 300 )) ] && break
  sleep 10
done
date -u +%FT%TZ > $R/watchdog-armed_utc.txt
sleep $WD
cd /tmp; export HOME=/home/michael
$PY -m modal app stop --yes $APP > $R/watchdog.out 2>&1
$PY -m modal app list --json > $R/watchdog-applist.json 2>&1
touch $R/WATCHDOG_FIRED
WDEOF
setsid nohup bash $R/watchdog.sh >/dev/null 2>&1 9>&- &
echo $! > $R/watchdog.pid
(
export HOME=$R/home YETO_RUNS_DIR=$R/runs PYTHONPATH=$R/yeto
eval "$(/usr/bin/python3 - <<'PY'
import base64, json, shlex
a = json.load(open("/home/michael/.docker/config.json"))["auths"]["ghcr.io"]["auth"]
u, t = base64.b64decode(a).decode().split(":", 1)
print(f"export SKYPILOT_DOCKER_USERNAME={shlex.quote(u)} SKYPILOT_DOCKER_PASSWORD={shlex.quote(t)} SKYPILOT_DOCKER_SERVER=ghcr.io")
PY
)"
exec 9>/home/michael/work/infra-drafts/GPU-LAUNCH.lock; flock 9
T2=$(ps -eLo user= | grep -c '^michael'); if [ "$T2" -ge 2800 ]; then echo "abort under lock: $T2 threads" > $R/abort.txt; echo "rc=3" > $R/rc.txt; exit 3; fi
cd $R/yeto; date -u +%FT%TZ > $R/start_utc.txt
# Local kill (container-time): the CLI runs in its own process group; rc=124 mimics `timeout`.
setsid bash -c "exec $PY -m yeto.cli $(cat $R/args.txt)" 9>&- > >(tee $R/launch.log 9>&- | awk '{ print strftime("%FT%TZ", systime(), 1) " " $0; fflush() }' 9>&- > $R/launch.ts.log) 2>&1 &
CLI=$!
if [ -n "${KILL_AT:-}" ]; then
( cd /tmp; export HOME=/home/michael
  until grep -aq "Suspended agentic groups at rollout ${KILL_AT}:" $R/launch.log 2>/dev/null; do kill -0 $CLI 2>/dev/null || { echo "$(date -u +%FT%TZ) launch ended before reclaim" >> $R/reclaim.log; exit 0; }; sleep 2; done
  echo "$(date -u +%FT%TZ) saw suspend at rollout ${KILL_AT}" >> $R/reclaim.log
  sleep ${KILL_DELAY:-8}
  CID=$(grep -a -oP '\[modal-island 0\] rank 0 container \K\S+' $R/launch.log | tail -1)
  echo "$(date -u +%FT%TZ) trigger container=$CID" >> $R/reclaim.log
  [ -n "$CID" ] && timeout 120 $PY -m modal container stop --yes $CID >> $R/reclaim.log 2>&1; echo "stop rc=$? $(date -u +%FT%TZ)" >> $R/reclaim.log
) 9>&- &
fi
t0=$(date +%s); seen=0
while kill -0 $CLI 2>/dev/null; do
  grep -q "$CONTAINER_MARK" $R/launch.log 2>/dev/null && { seen=1; break; }
  [ $(( $(date +%s) - t0 )) -ge $QUEUE_MAX ] && break
  sleep 5
done
exec 9>&-  # lock released once the container is up (or the queue wait ended)
if [ $seen = 1 ]; then date -u +%FT%TZ > $R/container_utc.txt; deadline=$(( $(date +%s) + HARD )); else echo "queue_max ${QUEUE_MAX}s exceeded or cli exited before container" > $R/timeout.txt; deadline=$(date +%s); fi
while kill -0 $CLI 2>/dev/null && [ $(date +%s) -lt $deadline ]; do sleep 5; done
if kill -0 $CLI 2>/dev/null; then
  echo "local timeout at $(date -u +%FT%TZ) (container-time HARD=${HARD}s, seen=$seen)" >> $R/timeout.txt
  kill -TERM -- -$CLI 2>/dev/null; sleep 30; kill -KILL -- -$CLI 2>/dev/null; wait $CLI 2>/dev/null; echo "rc=124" > $R/rc.txt
else
  wait $CLI; echo "rc=$?" > $R/rc.txt
fi
date -u +%FT%TZ > $R/end_utc.txt
)
cd /tmp; export HOME=/home/michael
timeout 300 $PY -m modal app stop --yes $APP > $R/final-stop.out 2>&1
sleep 15; timeout 120 $PY -m modal app list --json > $R/final-applist.json 2>&1
mkdir -p $R/tape-direct; timeout 600 $PY -m modal volume get --force $TAPEVOL $APP $R/tape-direct > $R/tape-direct.out 2>&1
kill $(cat $R/watchdog.pid) 2>/dev/null
echo "done $P $(cat $R/rc.txt)"

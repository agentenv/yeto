#!/bin/bash
# S14 G2 (fix-decoupled-lr-schedule 3.2): head-mode two-island DECOUPLED rerun, config = yeto-hp929d
#   (--rl-sync-preset decoupled --total-steps 4 --fragments 4 --pipeline 2 --local-rl-rounds-per-sync 2 --inner-lr 1e-5),
#   derived from s13-g3-remote.sh: ONE Nebius eu-north1 CPU VM hosts controller + actor syncer (:29400), two Modal
#   H100! islands dial its public IP. Differences from s13-g3: no critic (no spec, no critic syncer), no killer /
#   kill-resume (3.2 does not ask for it), no --keep, --rl-stall-timeout 2400 (S13 §7: 900 s fired during a 14 min
#   WAN upload; here only LoRA deltas cross the WAN but the margin is kept), ENGINE selects --rl-engine.
#   Modal's --modal-timeout-s counts from container start (queue time is not billed and not counted).
# usage: ENGINE=ports|legacy s14-dlr-remote.sh [run_id]   env: PLAN_ONLY=1, REPO, SHA, THREAD_MAX, HARD
set -u
ENGINE=${ENGINE:-ports}; P=${1:-s14-dlr-$ENGINE-20261007a}; HARD=${HARD:-2700}; WD=$(( HARD + 600 )); MODAL_TIMEOUT=$HARD
REPO=${REPO:-/home/michael/work/s14-dlr}; B=/home/michael/work/s1-runs; R=$B/$P
SKY=/home/michael/work/gpu-head/venv/bin/sky; PY=/home/michael/work/gpu-head/venv/bin/python
APP=yeto-$P; TAPEVOL=yeto-event-tapes; EXPECT_GPU=H100; HEAD_REGION=nebius/eu-north1
export CARGO_HOME=/home/michael/work/gpu-default-modal/cargo RUSTUP_HOME=/home/michael/work/gpu-default-modal/rustup
export PATH=$CARGO_HOME/bin:$PATH
# both engines run in the public MILES_NEXT_IMAGE (legacy's default MILES_IMAGE is a private ghcr image Modal cannot pull;
# the legacy setup step clones agentenv/miles + the vendored bundle into it, as the 2026-09-29 L40S sandbox did)
IMAGE=$(cd "$REPO" && PYTHONPATH=. /usr/bin/python3 -c "import yeto.rl as r; print(r.MILES_NEXT_IMAGE if r.MILES_NEXT_IMAGE.startswith('docker:') else 'docker:'+r.MILES_NEXT_IMAGE)" 2>/dev/null)
[ -n "$IMAGE" ] || { echo "abort: no MILES_NEXT_IMAGE"; exit 65; }
T=$(ps -u michael -L -o pid= | wc -l); [ "$T" -lt ${THREAD_MAX:-3000} ] || { echo "abort: $T user threads"; exit 3; }
pgrep -x raylet >/dev/null && { echo "abort: local Ray running"; exit 3; }
grep -q 'secrets=secrets or None' $REPO/yeto/cli.py || { echo "abort: $REPO lacks head secrets"; exit 64; }
HEAD_CL=$(cd $REPO && PYTHONPATH=$REPO /usr/bin/python3 -c "from yeto.launcher import sky_cluster_name; print(sky_cluster_name('$P-head'))")
if [ "${PLAN_ONLY:-0}" = 1 ]; then R=/tmp/$P-plan; rm -rf $R; fi
[ -e $R/start_utc.txt ] && { echo "abort: $R already used"; exit 67; }
mkdir -p $R/home $R/runs $R/yeto $R/head
for d in .sky .ssh .nebius .cache .huggingface .config; do [ -e /home/michael/$d ] && ln -sfn /home/michael/$d $R/home/$d; done
git -C $REPO archive ${SHA:-HEAD} | tar x -C $R/yeto
cp /home/michael/work/gpu-default-modal/yeto/gsm8k_reward.py $R/yeto/; touch $R/yeto/yeto-rl-echo-events
MODEL="--model Qwen/Qwen3-0.6B --model-revision c1899de289a04d12100db370d81485cdf75e47ca"; LORA="--tuning lora --lora-r 16 --lora-targets all-linear"
SYNC="--rl-sync-preset decoupled --total-steps 4 --fragments 4 --pipeline 2 --local-rl-rounds-per-sync 2"
MODALX="--modal-gpu-exact --modal-retries 0 --modal-launcher-relaunch --recover-timeout 1800 --modal-timeout-s $MODAL_TIMEOUT --modal-tape-volume $TAPEVOL"
ARGS="launch --controller head --syncer-region $HEAD_REGION --training-mode rl --rl-engine $ENGINE --rl-stall-timeout 2400 --on-demand --gpu modal:1xh100,modal:1xh100 --cluster-prefix $P $MODALX --rl-image $IMAGE $MODEL --data zhuzilin/gsm8k --data-revision 0cbd9f31d91ac21a7613dcbc7fef992adac459ae --reward-function gsm8k_reward:score $LORA $SYNC --rollout-batch-size 4 --n-samples-per-prompt 8 --rollout-max-response-len 384 --seq-len 1024 --inner-lr 1e-5 --seed 17 --apply-chat-template-kwargs '{\"enable_thinking\": false}' --trust-remote-code"
git -C $REPO rev-parse ${SHA:-HEAD} > $R/yeto_sha.txt; echo "$ARGS" > $R/args.txt; echo "$APP $HEAD_CL" > $R/resources.txt
eval "$(/usr/bin/python3 - <<'PYC'
import base64, json, shlex, tomllib
t = tomllib.load(open("/home/michael/.modal.toml", "rb"))
prof = next((v for v in t.values() if isinstance(v, dict) and v.get("active")), None) or next(v for v in t.values() if isinstance(v, dict))
a = json.load(open("/home/michael/.docker/config.json"))["auths"]["ghcr.io"]["auth"]
u, p = base64.b64decode(a).decode().split(":", 1)
print(f"export MODAL_TOKEN_ID={shlex.quote(prof['token_id'])} MODAL_TOKEN_SECRET={shlex.quote(prof['token_secret'])}")
print(f"export SKYPILOT_DOCKER_USERNAME={shlex.quote(u)} SKYPILOT_DOCKER_PASSWORD={shlex.quote(p)} SKYPILOT_DOCKER_SERVER=ghcr.io")
PYC
)"
[ -n "${MODAL_TOKEN_ID:-}" ] || { echo "abort: no modal token"; exit 65; }
export HOME=$R/home YETO_RUNS_DIR=$R/runs PYTHONPATH=$R/yeto
if [ "${PLAN_ONLY:-0}" = 1 ]; then
  echo "PLAN_ONLY run=$P engine=$ENGINE app=$APP head=$HEAD_CL ($HEAD_REGION) hard=${HARD}s wd=${WD}s code=$REPO@$(cat $R/yeto_sha.txt|cut -c1-8) threads=$T"
  (cd $R/yeto && $PY - "$ARGS" <<'PY'
import shlex, sys
import sky
from yeto import cli, launcher
args = cli.parse_args(shlex.split(sys.argv[1])[1:])
launcher.prepare_launch_args(args)
print("controller:", args.controller, "engine:", args.rl_engine, "preset:", args.rl_sync_preset, "stall:", args.rl_stall_timeout,
      "keep:", getattr(args, "keep", None), "syncer ports:", launcher.syncer_ports(args), "needs critic:", launcher.rl_needs_critic(args))
print("lr schedule:", {k: getattr(args, k, None) for k in vars(args) if "lr" in k and "lora" not in k})
print("syncer cmd (head LocalSyncer):", launcher.syncer_command(args, 2))
mounts, envs = launcher.head_cloud_credentials(launcher.fleet_clouds(args))
print("head cred file mounts:", sorted(mounts), "| cred env NAMES (sent as secrets):", sorted(envs))
assert "~/.modal.toml" not in mounts and {"MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"} <= set(envs)
task = cli._make_head_task(args, {})
cfg = task.to_yaml_config()
print("head resources:", cfg.get("resources"), "| secret NAMES:", sorted(cfg.get("secrets") or {}), "| env NAMES:", sorted(cfg.get("envs") or {}))
req = sky.launch(task, cluster_name="dryrun-dlr", dryrun=True)
sky.stream_and_get(req)
PY
  ) 2>&1 | grep -v "^\s*$" | tail -40
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
JOB=""
for i in $(seq 1 360); do
  JOB=$(grep -oP "submitted: job \K\d+" $R/launch.stream.log 2>/dev/null | head -1); [ -n "$JOB" ] && break
  kill -0 $SUB 2>/dev/null || break; sleep 10
done
echo "head_job=$JOB" > $R/head_job.txt
wait $SUB
if [ -n "$JOB" ]; then
  END=$(( $(date +%s) + HARD ))
  while [ $(date +%s) -lt $END ]; do
    st=$(cd /tmp && timeout 120 $SKY queue $HEAD_CL 2>/dev/null | awk -v j=$JOB '$1==j{print}')
    echo "$st" | grep -qE "RUNNING|PENDING|SETTING_UP|INIT" || break; sleep 30
  done
fi
date -u +%FT%TZ > $R/end_utc.txt
export HOME=/home/michael
S='ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20'
if timeout 60 $SKY status $HEAD_CL 2>/dev/null | grep -q ' UP '; then
  (cd /tmp && timeout 600 $SKY logs $HEAD_CL $JOB --no-follow) > $R/launch.log 2>&1
  (cd /tmp && timeout 300 $SKY queue $HEAD_CL) > $R/head/queue.txt 2>&1
  timeout 600 $S $HEAD_CL 'cd ~ && tar czf - yeto-output yeto-syncer.log sky_logs 2>/dev/null' > $R/head/head.tgz 2>/dev/null
  (cd $R/head && tar xzf head.tgz 2>/dev/null)
  timeout 60 $S $HEAD_CL 'hostname; nproc; free -g; nvidia-smi -L 2>&1|head -2; ss -ltnp 2>/dev/null|grep -E ":29400"' > $R/head/host.txt 2>&1
fi
[ -s $R/launch.log ] || cp $R/launch.stream.log $R/launch.log
grep -E "^ *$JOB " $R/head/queue.txt 2>/dev/null | grep -q SUCCEEDED && echo "rc=0" > $R/rc.txt || echo "rc=1 ($(grep -E "^ *$JOB " $R/head/queue.txt 2>/dev/null | tr -s ' ' | cut -c1-120))" > $R/rc.txt
bash $R/teardown.sh final
mkdir -p $R/tape-direct; timeout 600 $PY -m modal volume get --force $TAPEVOL $APP $R/tape-direct > $R/tape-direct.out 2>&1
kill $(cat $R/watchdog.pid) 2>/dev/null
/usr/bin/python3 $B/s14-dlr-judge.py $R $EXPECT_GPU > $R/judgment.json 2>&1
echo "done $P $(cat $R/rc.txt) judgment=$(head -c 300 $R/judgment.json)"

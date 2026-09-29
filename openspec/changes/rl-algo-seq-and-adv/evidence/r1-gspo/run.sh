#!/bin/bash
# R1 channel GPU recheck: gspo_s2 on integ-decl 501d71d (plan.md). usage: run.sh
set -u; G=$(cd "$(dirname "$0")" && pwd); E=$G/run; mkdir -p $E; T=/tmp/a2-r1
[ "$(git -C $T rev-parse --short=7 HEAD)" = "$(cat $G/YETO_SHA)" ] || { echo "tree not at YETO_SHA"; exit 9; }
PREFIX=${PREFIX:-algo2a-r1-gspo}; APP=yeto-$PREFIX; RUNS=/tmp/algo2a/r1runs; mkdir -p /tmp/algo2a
eval "$(python3 - <<'PY'
import base64, json, os, shlex
a = json.load(open(os.path.expanduser("~/.docker/config.json")))["auths"]["ghcr.io"]["auth"]
u, p = base64.b64decode(a).decode().split(":", 1)
print(f"export SKYPILOT_DOCKER_USERNAME={shlex.quote(u)} SKYPILOT_DOCKER_PASSWORD={shlex.quote(p)} SKYPILOT_DOCKER_SERVER=ghcr.io")
PY
)"
cp /home/michael/work/algo-2a/openspec/changes/rl-algo-seq-and-adv/evidence/g3/harness/watchdog.sh /tmp/algo2a/algo2a-r1-watchdog.sh
setsid nohup /tmp/algo2a/algo2a-r1-watchdog.sh $(( $(date +%s) + 2700 )) $APP /nonexistent-syncer $E/watchdog.log >/dev/null 2>&1 < /dev/null &
echo $! > $E/watchdog.pid
date -u +%FT%TZ > $E/start_time.txt
cd $T && YETO_RUNS_DIR=$RUNS PYTHONPATH=$T timeout 2400 /home/michael/work/gpu-head/venv/bin/python -m yeto.cli launch \
 --training-mode rl --rl-engine ports --rl-single-island-no-sync --controller local \
 --gpu modal:1xh100 --modal-gpu-exact --cluster-prefix $PREFIX \
 --model Qwen/Qwen3-0.6B --model-revision c1899de289a04d12100db370d81485cdf75e47ca \
 --data zhuzilin/gsm8k --data-revision 0cbd9f31d91ac21a7613dcbc7fef992adac459ae \
 --reward-function yeto.rl.algos.gdpo_reward:correctness_reward \
 --tuning lora --lora-r 16 --lora-targets all-linear --total-steps 3 \
 --rollout-batch-size 4 --n-samples-per-prompt 8 --rollout-max-response-len 384 --seq-len 1024 \
 --inner-lr 1e-5 --seed 17 --apply-chat-template-kwargs '{"enable_thinking": false}' --trust-remote-code \
 --rl-optimizer-steps 2 --rl-algorithm-spec openspec/changes/rl-algo-seq-and-adv/examples/gspo.json \
 > $E/launch.log 2>&1
echo $? > $E/rc; date -u +%FT%TZ > $E/end_time.txt
mkdir -p $E/events; cp $RUNS/$PREFIX/events/*.jsonl $E/events/ 2>/dev/null
cd $T && YETO_RUNS_DIR=$RUNS PYTHONPATH=$T timeout 300 /home/michael/work/gpu-head/venv/bin/python -m yeto.cli down $PREFIX > $E/down.log 2>&1
timeout 120 /tmp/modal-venv/bin/modal app stop -y $APP >> $E/down.log 2>&1
kill -9 $(cat $E/watchdog.pid) 2>/dev/null
timeout 60 /tmp/modal-venv/bin/modal app list --json 2>/dev/null | python3 -c "import json,sys; [print(a['app_id'],a['description'],a['state'],a['tasks'],a.get('stopped_at')) for a in json.load(sys.stdin) if a['description']=='$APP']" > $E/app_after_stop.txt
echo "r1 rc=$(cat $E/rc)"

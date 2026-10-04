#!/bin/bash
# usage: run.sh a|c
set -u
X=$1; P=algo1b-g1f-$X${ATTEMPT:+-$ATTEMPT}; APP=yeto-$P; D=/tmp/algo1b-g1f; mkdir -p $D; SRC=$(pwd); cd $D; Y=/home/michael/work/algo-1b-tok
M=/tmp/modal-venv/bin/modal
mkdir -p out-$X; date -u +%FT%TZ > out-$X/t_start; git -C $Y rev-parse HEAD > out-$X/YETO_SHA
setsid nohup bash -c "sleep 3900; $M app stop -y $APP > $D/out-$X/watchdog.log 2>&1" >/dev/null 2>&1 & echo $! > out-$X/watchdog_pid
COMMON="launch --training-mode rl --rl-engine ports --rl-single-island-no-sync --controller local --gpu modal:1xh100 --cluster-prefix $P --model Qwen/Qwen3-0.6B --model-revision c1899de289a04d12100db370d81485cdf75e47ca --data zhuzilin/gsm8k --data-revision 0cbd9f31d91ac21a7613dcbc7fef992adac459ae --reward-function yeto.rl.math_reward:reward_func --tuning lora --lora-r 16 --lora-targets all-linear --n-samples-per-prompt 8 --rollout-max-response-len 384 --seq-len 1024 --seed 17 --trust-remote-code --apply-chat-template-kwargs {\"enable_thinking\":false}"
ARGS="$COMMON --rollout-batch-size 4 --inner-lr 1e-5 --total-steps 3 --rl-algorithm-spec $SRC/$X.json"
if [ $X = token ]; then ARGS="$ARGS --rl-allow-unverified-mechanism loss_aggregations:token"; fi
echo "yeto $ARGS" > out-$X/cmd.txt
setsid nohup $SRC/noprogress.sh $X >/dev/null 2>&1 &
# Private ports image: registry credentials decoded in-process from
# ~/.docker/config.json (ghcr.io auth); never printed or logged.
eval "$(python3 - <<'PY'
import base64, json, shlex
a = json.load(open("/home/michael/.docker/config.json"))["auths"]["ghcr.io"]["auth"]
u, p = base64.b64decode(a).decode().split(":", 1)
print(f"export SKYPILOT_DOCKER_USERNAME={shlex.quote(u)} SKYPILOT_DOCKER_PASSWORD={shlex.quote(p)} SKYPILOT_DOCKER_SERVER=ghcr.io")
PY
)"
cd $Y && PYTHONPATH=$Y timeout 3600 /home/michael/work/gpu-head/venv/bin/python -m yeto.cli $ARGS > $D/out-$X/launch.log 2>&1; echo "rc=$?" >> $D/out-$X/launch.log
# The launcher detaches a `yeto _worker`; a local timeout does not stop it.
(cd $Y && PYTHONPATH=$Y timeout 300 /home/michael/work/gpu-head/venv/bin/python -m yeto.cli down $P) >> $D/out-$X/teardown.log 2>&1
$M app stop -y $APP >> $D/out-$X/teardown.log 2>&1
pkill -P $(cat $D/out-$X/watchdog_pid) sleep 2>/dev/null; kill $(cat $D/out-$X/watchdog_pid) 2>/dev/null
date -u +%FT%TZ > $D/out-$X/t_end
rm -rf $SRC/out-$X; cp -r $D/out-$X $SRC/out-$X

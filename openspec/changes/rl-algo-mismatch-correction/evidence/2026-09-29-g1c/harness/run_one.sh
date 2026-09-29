#!/bin/bash
# run_one.sh <run> <spec.json> <allow...>   (from evidence/2026-09-29-g1c)
set -u; R=$1; SPEC=$2; shift 2
E=$(pwd)/runs/$R; mkdir -p $E; T=/tmp/algo1a/tree-$R
PFX=algo1a-g1c-$R; APP=yeto-$PFX
rm -rf $T; mkdir -p $T; git -C /home/michael/work/algo-1a archive $(cat YETO_SHA) | tar x -C $T
cp harness/gsm8k_reward.py $T/; cp $SPEC $E/spec.json
eval "$(/tmp/yeto-venv/bin/python - <<'PY'
import base64,json,os,shlex
a=json.load(open(os.path.expanduser("~/.docker/config.json")))["auths"]["ghcr.io"]["auth"]
u,p=base64.b64decode(a).decode().split(":",1)
print(f"export SKYPILOT_DOCKER_USERNAME={shlex.quote(u)} SKYPILOT_DOCKER_PASSWORD={shlex.quote(p)} SKYPILOT_DOCKER_SERVER=ghcr.io")
PY
)"
ALLOW=(); for a in "$@"; do ALLOW+=(--rl-allow-unverified-mechanism "$a"); done
cleanup(){ timeout 120 /tmp/modal-venv/bin/modal app stop -y $APP >/dev/null 2>&1; }
trap cleanup EXIT
setsid nohup bash -c "sleep 3000; /tmp/modal-venv/bin/modal app stop -y $APP" >/dev/null 2>&1 < /dev/null &
echo $! > $E/watchdog.pid
( while sleep 20; do c=$(timeout 60 /tmp/modal-venv/bin/modal container list --json 2>/dev/null | python3 -c "import json,sys; print(' '.join(x['container_id'] for x in json.load(sys.stdin) if x['app_name']=='$APP'))" 2>/dev/null)
    for x in $c; do timeout 60 /tmp/modal-venv/bin/modal container exec $x -- sh -c "cat /root/yeto-output/rl-island-0.jsonl 2>/dev/null" > $E/.tape 2>/dev/null && [ -s $E/.tape ] && grep -q '"event"' $E/.tape && mv $E/.tape $E/tape.jsonl; done; done ) &
PULLER=$!
date -u +%FT%TZ > $E/start_time.txt
cd $T && PYTHONPATH=$T timeout 2700 /tmp/yeto-venv/bin/python -m yeto.cli launch --training-mode rl --rl-engine ports \
  --gpu modal:1xh100 --modal-gpu-exact --controller local --rl-single-island-no-sync \
  --rl-algorithm-spec $E/spec.json "${ALLOW[@]}" --cluster-prefix $PFX \
  --model Qwen/Qwen3-0.6B --model-revision c1899de289a04d12100db370d81485cdf75e47ca \
  --data zhuzilin/gsm8k --data-revision 0cbd9f31d91ac21a7613dcbc7fef992adac459ae \
  --reward-function gsm8k_reward:score --tuning lora --lora-r 16 --lora-targets all-linear \
  --total-steps 3 --fragments 1 --pipeline 1 --rollout-batch-size 4 --n-samples-per-prompt 8 \
  --rollout-max-response-len 384 --seq-len 1024 --inner-lr 1e-5 --seed 17 \
  --apply-chat-template-kwargs '{"enable_thinking": false}' --trust-remote-code > $E/launch.log 2>&1
echo $? > $E/rc; date -u +%FT%TZ > $E/end_time.txt
kill $PULLER 2>/dev/null; kill $(cat $E/watchdog.pid) 2>/dev/null
cleanup; timeout 60 /tmp/modal-venv/bin/modal app list 2>/dev/null | grep "$APP" > $E/app_after_stop.txt
echo "$R rc=$(cat $E/rc)"

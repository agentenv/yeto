#!/bin/bash
set -u; G=$(pwd); E=$G/run; mkdir -p $E; T=/tmp/algo1a/tree-g3; H=/tmp/algo1a/g3home; APP=yeto-algo1a-g3b
rm -rf $T $H; mkdir -p $T $H/yeto-output
git -C /home/michael/work/algo-1a archive $(cat YETO_SHA) | tar x -C $T; rm -rf $T/openspec $T/tests $T/docs; cp harness/gsm8k_reward.py harness/run_local_head.py $T/
cp /home/michael/work/gpu-default-modal/home/yeto-syncer $H/; ln -s /home/michael/.modal.toml $H/.modal.toml; ln -s /home/michael/.sky $H/.sky
ss -ltn | grep -q ":29410 " && { echo "port 29410 busy" | tee $E/port_check.txt; exit 5; }; echo "29410 free $(date -u +%FT%TZ)" > $E/port_check.txt
eval "$(/tmp/yeto-venv/bin/python - <<'PY'
import base64,json,os,shlex
a=json.load(open(os.path.expanduser("~/.docker/config.json")))["auths"]["ghcr.io"]["auth"]
u,p=base64.b64decode(a).decode().split(":",1)
print(f"export SKYPILOT_DOCKER_USERNAME={shlex.quote(u)} SKYPILOT_DOCKER_PASSWORD={shlex.quote(p)} SKYPILOT_DOCKER_SERVER=ghcr.io")
PY
)"
cat > $E/args.txt <<ARGS
launch --training-mode rl --rl-engine ports --rl-sync-preset strict-avg --gpu modal:1xh100,modal:1xh100 --modal-gpu-exact --cluster-prefix algo1a-g3b --rl-algorithm-spec $G/tis.json --model Qwen/Qwen3-0.6B --model-revision c1899de289a04d12100db370d81485cdf75e47ca --data zhuzilin/gsm8k --data-revision 0cbd9f31d91ac21a7613dcbc7fef992adac459ae --reward-function gsm8k_reward:score --tuning lora --lora-r 16 --lora-targets all-linear --total-steps 3 --fragments 1 --pipeline 1 --rollout-batch-size 4 --n-samples-per-prompt 8 --rollout-max-response-len 384 --seq-len 1024 --inner-lr 1e-5 --seed 17 --apply-chat-template-kwargs '{"enable_thinking": false}' --trust-remote-code
ARGS
cleanup(){ timeout 120 /tmp/modal-venv/bin/modal app stop -y $APP >/dev/null 2>&1; pkill -f "$H/yeto-syncer" 2>/dev/null; }
trap cleanup EXIT
setsid nohup bash -c "sleep 3300; /tmp/modal-venv/bin/modal app stop -y $APP; pkill -f $H/yeto-syncer" >/dev/null 2>&1 < /dev/null & echo $! > $E/watchdog.pid
( while sleep 20; do for c in $(timeout 60 /tmp/modal-venv/bin/modal container list --json 2>/dev/null | python3 -c "import json,sys; print(' '.join(x['container_id'] for x in json.load(sys.stdin) if x['app_name']=='$APP'))" 2>/dev/null); do
    for i in 0 1; do timeout 60 /tmp/modal-venv/bin/modal container exec $c -- sh -c "cat /root/yeto-output/rl-island-$i.jsonl 2>/dev/null" > $E/.t.$i 2>/dev/null; grep -q '"event"' $E/.t.$i 2>/dev/null && mv $E/.t.$i $E/island-$i.jsonl; done; done; done ) & PULLER=$!
date -u +%FT%TZ > $E/start_time.txt
cd $T && HOME=$H YETO_RUNS_DIR=$H/runs SYNCER_PUBLIC_IP=185.189.44.160 PYTHONPATH=$T timeout 3000 /home/michael/work/gpu-head/venv/bin/python $T/run_local_head.py $E/args.txt > $E/launch.log 2>&1
echo $? > $E/rc; date -u +%FT%TZ > $E/end_time.txt; sleep 30; kill $PULLER; kill $(cat $E/watchdog.pid) 2>/dev/null
sleep 5; cp $H/yeto-output/yeto-tape.jsonl $H/yeto-syncer.log $E/ 2>/dev/null; mkdir -p $E/events; cp $H/runs/algo1a-g3b/events/* $E/events/ 2>/dev/null
cleanup; timeout 60 /tmp/modal-venv/bin/modal app list 2>/dev/null | grep "$APP" > $E/app_after_stop.txt; ss -ltn | grep -c ":29410 " > $E/port_after.txt
echo "g3 rc=$(cat $E/rc)"

#!/bin/bash
# 7.6 MaxRL two-island strict-avg G3 (plan.md). Run from evidence/g3.
set -u; G=$(pwd); E=$G/run; mkdir -p $E; T=/tmp/algo2a/tree-g3; H=/tmp/algo2a/g3home; APP=yeto-algo2a-g3; PORT=29420
rm -rf $T $H; mkdir -p $T $H/yeto-output /tmp/algo2a
git -C /home/michael/work/algo-2a archive $(cat YETO_SHA) | tar x -C $T; rm -rf $T/tests $T/docs
find $T/openspec -mindepth 1 -maxdepth 1 ! -name changes -exec rm -rf {} +; cp harness/run_local_head.py $T/
cp /home/michael/work/gpu-default-modal/home/yeto-syncer $H/; ln -s /home/michael/.modal.toml $H/.modal.toml; ln -s /home/michael/.sky $H/.sky
ss -ltn | grep -q ":$PORT " && { echo "port $PORT busy" | tee $E/port_check.txt; exit 5; }; echo "$PORT free $(date -u +%FT%TZ)" > $E/port_check.txt
eval "$(/tmp/yeto-venv/bin/python - <<'PY'
import base64,json,os,shlex
a=json.load(open(os.path.expanduser("~/.docker/config.json")))["auths"]["ghcr.io"]["auth"]
u,p=base64.b64decode(a).decode().split(":",1)
print(f"export SKYPILOT_DOCKER_USERNAME={shlex.quote(u)} SKYPILOT_DOCKER_PASSWORD={shlex.quote(p)} SKYPILOT_DOCKER_SERVER=ghcr.io")
PY
)"
cat > $E/args.txt <<ARGS
launch --training-mode rl --rl-engine ports --rl-sync-preset strict-avg --gpu modal:1xh100,modal:1xh100 --modal-gpu-exact --cluster-prefix algo2a-g3 --rl-algorithm-spec $T/openspec/changes/rl-algo-seq-and-adv/examples/maxrl.json --model Qwen/Qwen3-0.6B --model-revision c1899de289a04d12100db370d81485cdf75e47ca --data zhuzilin/gsm8k --data-revision 0cbd9f31d91ac21a7613dcbc7fef992adac459ae --reward-function yeto.rl.algos.gdpo_reward:correctness_reward --tuning lora --lora-r 16 --lora-targets all-linear --total-steps 3 --fragments 1 --pipeline 1 --rollout-batch-size 4 --n-samples-per-prompt 8 --rollout-max-response-len 384 --seq-len 1024 --inner-lr 1e-5 --seed 17 --apply-chat-template-kwargs '{"enable_thinking": false}' --trust-remote-code
ARGS
cleanup(){ timeout 120 /tmp/modal-venv/bin/modal app stop -y $APP >/dev/null 2>&1; pkill -f "$H/yeto-syncer" 2>/dev/null; }
trap cleanup EXIT
cp harness/watchdog.sh /tmp/algo2a/algo2a-g3-watchdog.sh
setsid nohup /tmp/algo2a/algo2a-g3-watchdog.sh $(( $(date +%s) + 3300 )) $APP $H/yeto-syncer $E/watchdog.log >/dev/null 2>&1 < /dev/null &
echo $! > $E/watchdog.pid
# the harness re-checks the watchdog every minute while the head runs and restarts it (same deadline)
( dl=$(( $(date +%s) + 3300 )); while sleep 60; do kill -0 $(cat $E/watchdog.pid) 2>/dev/null || { echo "watchdog gone $(date -u +%FT%TZ); restarting" >> $E/watchdog.log; setsid nohup /tmp/algo2a/algo2a-g3-watchdog.sh $dl $APP $H/yeto-syncer $E/watchdog.log >/dev/null 2>&1 < /dev/null & echo $! > $E/watchdog.pid; }; done ) & GUARD=$!
date -u +%FT%TZ > $E/start_time.txt
cd $T && HOME=$H YETO_RUNS_DIR=$H/runs SYNCER_PUBLIC_IP=185.189.44.160 PYTHONPATH=$T timeout 3000 /home/michael/work/gpu-head/venv/bin/python $T/run_local_head.py $E/args.txt > $E/launch.log 2>&1
echo $? > $E/rc; date -u +%FT%TZ > $E/end_time.txt; kill $GUARD 2>/dev/null; kill -9 $(cat $E/watchdog.pid) 2>/dev/null
mkdir -p $E/events; cp $H/runs/algo2a-g3/events/*.jsonl $E/events/ 2>/dev/null; cp $H/yeto-output/*.jsonl $E/ 2>/dev/null; cp $H/yeto-syncer.log $E/ 2>/dev/null
cp $H/runs/algo2a-g3/*.jsonl $E/ 2>/dev/null; ls -R $H/runs > $E/runs_listing.txt 2>&1
cleanup; timeout 60 /tmp/modal-venv/bin/modal app list --json 2>/dev/null | python3 -c "import json,sys; [print(a['app_id'],a['description'],a['state'],a['tasks']) for a in json.load(sys.stdin) if a['description']=='$APP']" > $E/app_after_stop.txt
ss -ltn | grep -c ":$PORT " > $E/port_after.txt
echo "g3 rc=$(cat $E/rc)"

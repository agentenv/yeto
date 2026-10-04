#!/bin/bash
# usage: arm.sh <prefix> <yeto-sha> <gpu> <fault-json-or-empty> <extra launch args...>
set -u
T=$(ps -u michael -L -o pid= | wc -l); if [ "$T" -ge 3000 ]; then echo "abort: $T user threads (limit 4096)"; exit 3; fi
P=$1; SHA=$2; GPU=$3; FAULT=$4; shift 4; EXTRA="$*"
R=/home/michael/work/gpu-b1-runs/$P
mkdir -p $R/home $R/runs $R/pulled $R/yeto
cp ~/.modal.toml $R/home/; cp /home/michael/work/gpu-default-modal/home/yeto-syncer $R/home/
git -C /home/michael/work/gpu-b1 archive $SHA | tar x -C $R/yeto
cp /home/michael/work/gpu-default-modal/yeto/gsm8k_reward.py $R/yeto/
[ -n "$FAULT" ] && printf '%s' "$FAULT" > $R/yeto/yeto-rl-fault-injection.json
echo $SHA > $R/yeto_sha.txt; touch $R/yeto/yeto-rl-echo-events
cp /home/michael/work/infra-a-gpu/b12/run_local_head.py $R/
echo "launch --training-mode rl --rl-sync-preset strict-avg --gpu $GPU --modal-gpu-exact --cluster-prefix $P --no-island-relaunch --model Qwen/Qwen3-0.6B --model-revision c1899de289a04d12100db370d81485cdf75e47ca --data zhuzilin/gsm8k --data-revision 0cbd9f31d91ac21a7613dcbc7fef992adac459ae --reward-function gsm8k_reward:score --tuning lora --lora-r 16 --lora-targets all-linear --total-steps ${STEPS:-3} --fragments 1 --pipeline 1 --rollout-batch-size 4 --n-samples-per-prompt 8 --rollout-max-response-len 384 --seq-len 1024 --inner-lr 1e-5 --seed ${SEED:-17} --apply-chat-template-kwargs '{\"enable_thinking\": false}' --trust-remote-code --modal-retries 0 --modal-timeout-s $(( ${HARD:-2700} + 300 )) $EXTRA" > $R/args.txt
# watchdog (independent)
setsid nohup bash -c "sleep ${WD:-3000}; HOME=$R/home /tmp/modal-venv/bin/modal app stop -y yeto-$P > $R/watchdog.out 2>&1" >/dev/null 2>&1 &
echo $! > $R/watchdog.pid
# puller
setsid nohup bash -c "
export HOME=$R/home
while [ ! -f $R/rc.txt ]; do
  for c in \$(/tmp/modal-venv/bin/modal container list --json 2>/dev/null | /usr/bin/python3 -c 'import json,sys
try: d=json.load(sys.stdin)
except Exception: d=[]
[print(x[\"container_id\"]) for x in d if x.get(\"app_name\")==\"yeto-$P\"]'); do
    timeout 60 /tmp/modal-venv/bin/modal container exec \$c -- sh -c 'cat /root/yeto-output/rl-island-0.jsonl 2>/dev/null' > $R/pulled/.tmp 2>/dev/null
    [ -s $R/pulled/.tmp ] && mv $R/pulled/.tmp $R/pulled/rl-island-0.jsonl
    [ -s $R/pulled/gpu.txt ] || timeout 60 /tmp/modal-venv/bin/modal container exec \$c -- sh -c 'nvidia-smi --query-gpu=index,uuid,name,driver_version --format=csv,noheader' > $R/pulled/gpu.txt 2>/dev/null
    timeout 60 /tmp/modal-venv/bin/modal container exec \$c -- sh -c 'date -u +%FT%TZ; nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader; nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader' >> $R/pulled/compute-apps.txt 2>/dev/null
  done
  sz=\$(stat -c %s $R/pulled/rl-island-0.jsonl 2>/dev/null || echo 0); now=\$(date +%s)
  if [ \"\$sz\" != \"\${lastsz:-x}\" ]; then lastsz=\$sz; lastt=\$now; fi
  if [ \$(( now - \${lastt:-\$now} )) -ge 1200 ] && [ ! -f $R/progress_stop.txt ]; then
    echo \"no new events for 20 min at \$(date -u +%FT%TZ); pulling and stopping\" > $R/progress_stop.txt
    cp $R/pulled/rl-island-0.jsonl $R/pulled/rl-island-0.at-stop.jsonl 2>/dev/null
    HOME=$R/home /tmp/modal-venv/bin/modal app stop -y yeto-$P >> $R/progress_stop.txt 2>&1
  fi
  sleep 10
done" > $R/puller.log 2>&1 &
# launch
(
export HOME=$R/home YETO_RUNS_DIR=$R/runs SYNCER_PUBLIC_IP=185.189.44.160 PYTHONPATH=$R/yeto
eval "$(/usr/bin/python3 - <<'PY'
import base64, json, shlex
a = json.load(open("/home/michael/.docker/config.json"))["auths"]["ghcr.io"]["auth"]
u, t = base64.b64decode(a).decode().split(":", 1)
print(f"export SKYPILOT_DOCKER_USERNAME={shlex.quote(u)} SKYPILOT_DOCKER_PASSWORD={shlex.quote(t)} SKYPILOT_DOCKER_SERVER=ghcr.io")
PY
)"
cd $R/yeto
date -u +%FT%TZ > $R/start_utc.txt
timeout ${HARD:-2700} /home/michael/work/gpu-head/venv/bin/python $R/run_local_head.py $R/args.txt > $R/launch.log 2>&1
echo "rc=$?" > $R/rc.txt.tmp
date -u +%FT%TZ > $R/end_utc.txt
sleep 15; mv $R/rc.txt.tmp $R/rc.txt
)

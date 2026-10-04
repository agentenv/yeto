#!/bin/bash
# usage: mrun.sh <prefix> <gpu e.g. modal:1xl40s> <hard_s> <watchdog_s> <launch args...>  (Modal island, --controller local, no-sync)
set -u
T=$(ps -u michael -L -o pid= | wc -l); if [ "$T" -ge 3000 ]; then echo "abort: $T user threads"; exit 3; fi
P=$1; GPU=$2; HARD=$3; WD=$4; shift 4; EXTRA="$*"
B=/home/michael/work/gpu-b1-runs; R=$B/$P; M=/tmp/modal-venv/bin/modal
IMG=ghcr.io/michaellchung/yeto-miles-ports@sha256:2cc5cc52de2444e59ddefba4f9546d1aaa13f9807ab441f56e2c70a7e7936eff
mkdir -p $R/home $R/runs $R/pulled $R/yeto
cp ~/.modal.toml $R/home/
git -C /home/michael/work/gpu-b1 archive ${SHA:-${SHA:-1a5ccd5}} | tar x -C $R/yeto
cp /home/michael/work/gpu-default-modal/yeto/gsm8k_reward.py $R/yeto/
touch $R/yeto/yeto-rl-echo-events; echo ${SHA:-${SHA:-1a5ccd5}} > $R/yeto_sha.txt
echo "launch --controller local --training-mode rl --rl-single-island-no-sync --gpu $GPU --cluster-prefix $P --no-island-relaunch --model Qwen/Qwen3-0.6B --model-revision c1899de289a04d12100db370d81485cdf75e47ca --data zhuzilin/gsm8k --data-revision 0cbd9f31d91ac21a7613dcbc7fef992adac459ae --reward-function gsm8k_reward:score --tuning lora --lora-r 16 --lora-targets all-linear --fragments 1 --pipeline 1 --rollout-batch-size 4 --n-samples-per-prompt 8 --rollout-max-response-len 384 --seq-len 1024 --inner-lr 1e-5 --seed 17 --apply-chat-template-kwargs '{\"enable_thinking\": false}' --trust-remote-code --modal-retries 0 --modal-timeout-s $(( ${HARD:-2700} + 300 )) $EXTRA" > $R/args.txt
setsid nohup bash -c "sleep $WD; HOME=$R/home $M app stop -y yeto-$P > $R/watchdog.out 2>&1" >/dev/null 2>&1 &
echo $! > $R/watchdog.pid
setsid nohup bash -c "
export HOME=$R/home
while [ ! -f $R/rc.txt ]; do
  for c in \$($M container list --json 2>/dev/null | /usr/bin/python3 -c 'import json,sys
try: d=json.load(sys.stdin)
except Exception: d=[]
[print(x[\"container_id\"]) for x in d if x.get(\"app_name\")==\"yeto-$P\"]'); do
    echo \$c > $R/pulled/container_id.txt
    timeout 60 $M container exec \$c -- sh -c 'cat /root/yeto-output/rl-island-0.jsonl 2>/dev/null' > $R/pulled/.tmp 2>/dev/null
    [ -s $R/pulled/.tmp ] && mv $R/pulled/.tmp $R/pulled/rl-island-0.jsonl
    [ -s $R/pulled/gpu.txt ] || timeout 60 $M container exec \$c -- sh -c 'nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader' > $R/pulled/gpu.txt 2>/dev/null
    timeout 60 $M container exec \$c -- sh -c 'date -u +%FT%TZ; nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader' >> $R/pulled/compute-apps.txt 2>/dev/null
    if [ -n \"${MANIFEST:-}\" ] && [ ! -s $R/pulled/manifest.json ]; then
      timeout 300 $M container exec \$c -- sh -c 'cd ~/sky_workdir && PYTHONPATH=~/miles:~/sglang/python:. python3 -m yeto.rl.engine.runtime_manifest --image $IMG --out /tmp/m.json > /tmp/m.check 2>&1; echo rc=\$? >> /tmp/m.check; cat /tmp/m.json' > $R/pulled/manifest.json 2>/dev/null
      timeout 60 $M container exec \$c -- sh -c 'cat /tmp/m.check' > $R/pulled/manifest-check.txt 2>/dev/null
    fi
    timeout 60 $M container exec \$c -- sh -c 'date -u +%FT%TZ; ps -eo pid,args --no-headers | grep -E \"ray::|sglang|scheduler\" | grep -v grep | cut -c1-160' >> $R/pulled/ps.txt 2>/dev/null
    timeout 60 $M container exec \$c -- sh -c 'cat ~/yeto-rl/inwatch.log 2>/dev/null' > $R/pulled/inwatch.log 2>/dev/null
    if [ -n \"${PULL_STATE:-}\" ]; then
      timeout 60 $M container exec \$c -- sh -c 'cd ~/yeto-rl 2>/dev/null && tar czf - elastic-state | base64 -w0' > $R/pulled/.st 2>/dev/null && [ -s $R/pulled/.st ] && mv $R/pulled/.st $R/pulled/elastic-state.tgz.b64
    fi
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
echo $! > $R/puller.pid
(
export HOME=$R/home YETO_RUNS_DIR=$R/runs PYTHONPATH=$R/yeto
eval "$(/usr/bin/python3 - <<'PY'
import base64, json, shlex
a = json.load(open("/home/michael/.docker/config.json"))["auths"]["ghcr.io"]["auth"]
u, t = base64.b64decode(a).decode().split(":", 1)
print(f"export SKYPILOT_DOCKER_USERNAME={shlex.quote(u)} SKYPILOT_DOCKER_PASSWORD={shlex.quote(t)} SKYPILOT_DOCKER_SERVER=ghcr.io")
PY
)"
cd $R/yeto
date -u +%FT%TZ > $R/start_utc.txt
eval "timeout $HARD /home/michael/work/gpu-head/venv/bin/python -m yeto.cli $(cat $R/args.txt)" > $R/launch.log 2>&1
echo "rc=$?" > $R/rc.txt.tmp
date -u +%FT%TZ > $R/end_utc.txt
sleep 15; mv $R/rc.txt.tmp $R/rc.txt
)

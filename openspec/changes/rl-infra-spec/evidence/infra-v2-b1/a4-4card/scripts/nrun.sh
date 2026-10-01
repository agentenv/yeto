#!/bin/bash
# usage: nrun.sh <prefix> <hard_timeout_s> <watchdog_s> <launch args...>   (Nebius/sky island, local head)
set -u
P=$1; HARD=$2; WD=$3; shift 3; EXTRA="$*"
B=/home/michael/work/gpu-b1-runs; R=$B/$P; SKY=/home/michael/work/gpu-head/venv/bin/sky
mkdir -p $R/home $R/runs $R/pulled $R/yeto
for d in .sky .nebius .ssh; do ln -sfn /home/michael/$d $R/home/$d; done
cp /home/michael/work/gpu-default-modal/home/yeto-syncer $R/home/
git -C /home/michael/work/gpu-b1 archive ${SHA:-47efd25} | tar x -C $R/yeto
cp /home/michael/work/gpu-default-modal/yeto/gsm8k_reward.py $R/yeto/
touch $R/yeto/yeto-rl-echo-events; echo ${SHA:-47efd25} > $R/yeto_sha.txt
cp /home/michael/work/infra-a-gpu/b12/run_local_head.py $R/
echo "launch --training-mode rl --on-demand --gpu nebius:${NGPU:-1}xh100@eu-north1 --cluster-prefix $P --no-island-relaunch --modal-retries 0 --model Qwen/Qwen3-0.6B --model-revision c1899de289a04d12100db370d81485cdf75e47ca --data zhuzilin/gsm8k --data-revision 0cbd9f31d91ac21a7613dcbc7fef992adac459ae --reward-function gsm8k_reward:score --tuning lora --lora-r 16 --lora-targets all-linear --fragments 1 --pipeline 1 --rollout-batch-size 4 --n-samples-per-prompt 8 --rollout-max-response-len 384 --seq-len 1024 --inner-lr 1e-5 --seed 17 --apply-chat-template-kwargs '{\"enable_thinking\": false}' --trust-remote-code $EXTRA" > $R/args.txt
CL=$P-l0-eu-north1; echo $CL > $R/cluster.txt
setsid nohup bash -c "sleep $WD; HOME=/home/michael $SKY down -y $CL > $R/watchdog.out 2>&1" >/dev/null 2>&1 &
echo $! > $R/watchdog.pid
setsid nohup bash -c "
export HOME=/home/michael
as=0
while [ ! -f $R/rc.txt ]; do
  if timeout 60 $SKY status $CL 2>/dev/null | grep -q ' UP '; then
    [ \$as = 1 ] || { timeout 120 $SKY autostop -y -i 10 --down $CL > $R/autostop.out 2>&1 && as=1; }
    timeout 60 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 $CL 'cat ~/yeto-output/rl-island-0.jsonl 2>/dev/null' > $R/pulled/.tmp 2>/dev/null && [ -s $R/pulled/.tmp ] && mv $R/pulled/.tmp $R/pulled/rl-island-0.jsonl
    timeout 60 ssh -o ConnectTimeout=20 $CL 'date -u +%FT%TZ; nvidia-smi -L; nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader' >> $R/pulled/gpu-samples.txt 2>/dev/null
    [ -n \"\${PULL_STATE:-}\" ] && timeout 60 ssh -o ConnectTimeout=20 $CL 'cd ~/yeto-rl && tar czf - elastic-state 2>/dev/null' > $R/pulled/.st.tgz 2>/dev/null && [ -s $R/pulled/.st.tgz ] && mv $R/pulled/.st.tgz $R/pulled/elastic-state.tgz
  fi
  sleep 10
done" > $R/puller.log 2>&1 &
echo $! > $R/puller.pid
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
timeout $HARD /home/michael/work/gpu-head/venv/bin/python $R/run_local_head.py $R/args.txt > $R/launch.log 2>&1
echo "rc=$?" > $R/rc.txt.tmp
date -u +%FT%TZ > $R/end_utc.txt
sleep 15; mv $R/rc.txt.tmp $R/rc.txt
)

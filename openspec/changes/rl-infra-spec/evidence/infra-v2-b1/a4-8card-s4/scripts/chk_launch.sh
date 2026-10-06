#!/bin/bash
# usage: chk_launch.sh <prefix> <ngpu> <hard_s>   (a8go_strict.sh case chk; chain env RUN_ROOT CLUSTER_PREFIX CHAIN_DIR as set by chain8.sh)
# Runs chk_launch.py under the hard timeout with the same workdir/HOME layout as n2run (git archive $SHA, docker login env), writes <run>/rc.txt + item_done;
# on failure/timeout also <chain>/ABORT so chain8 stops and releases the cluster.  Never starts training.
set -u
P=$1; NG=$2; HARD=$3; B=/home/michael/work/gpu-b1-runs; R=${RUN_ROOT:-$B}/$P; CP=${CLUSTER_PREFIX:-$P}; CL=$CP-l0-eu-north1
mkdir -p $R/home $R/runs $R/pulled $R/yeto; for d in .sky .nebius .ssh; do ln -sfn /home/michael/$d $R/home/$d; done
git -C /home/michael/work/gpu-b1 archive ${SHA:?} | tar x -C $R/yeto; cp /home/michael/work/gpu-default-modal/yeto/gsm8k_reward.py $R/yeto/; echo $SHA > $R/yeto_sha.txt; echo $CL > $R/cluster.txt
echo "launch --controller local --training-mode rl --rl-single-island-no-sync --on-demand --gpu ${GPU_SPEC:-nebius:${NG}xh100@eu-north1} --cluster-prefix $CP --keep --no-island-relaunch --modal-retries 0 --model Qwen/Qwen3-0.6B --model-revision c1899de289a04d12100db370d81485cdf75e47ca --data zhuzilin/gsm8k --data-revision 0cbd9f31d91ac21a7613dcbc7fef992adac459ae --reward-function gsm8k_reward:score --tuning lora --lora-r 16 --lora-targets all-linear --fragments 1 --pipeline 1 --rollout-batch-size 4 --n-samples-per-prompt 8 --rollout-max-response-len 384 --seq-len 1024 --inner-lr 1e-5 --seed 17 --apply-chat-template-kwargs '{\"enable_thinking\": false}' --trust-remote-code --total-steps 1" > $R/args.txt
(
export HOME=$R/home YETO_RUNS_DIR=$R/runs PYTHONPATH=$R/yeto CHK_OUT=$R/chk_result.json
eval "$(/usr/bin/python3 - <<'PY'
import base64, json, shlex
a = json.load(open("/home/michael/.docker/config.json"))["auths"]["ghcr.io"]["auth"]
u, t = base64.b64decode(a).decode().split(":", 1)
print(f"export SKYPILOT_DOCKER_USERNAME={shlex.quote(u)} SKYPILOT_DOCKER_PASSWORD={shlex.quote(t)} SKYPILOT_DOCKER_SERVER=ghcr.io")
PY
)"
cd $R/yeto; date -u +%FT%TZ > $R/start_utc.txt
timeout $HARD ${YETO_PY:-/home/michael/work/gpu-head/venv/bin/python} ${CHK_PY_SCRIPT:-$B/chk_launch.py} $R/args.txt $CL > $R/launch.log 2>&1; rc=$?
date -u +%FT%TZ > $R/end_utc.txt
if [ $rc != 0 ]; then
  why=$(grep -m3 '\[chk\] FAIL' $R/launch.log | tr '\n' ';'); [ $rc = 124 ] && why="hard timeout ${HARD}s"
  [ -n "${CHAIN_DIR:-}" ] && echo "{\"item\":\"$P\",\"reason\":\"chk_failed\",\"rc\":$rc,\"why\":\"${why:-see launch.log}\",\"ts\":\"$(date -u +%FT%TZ)\"}" > $CHAIN_DIR/ABORT
fi
echo "rc=$rc" > $R/rc.txt; touch $R/item_done
)

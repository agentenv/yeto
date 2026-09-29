#!/bin/bash
# G1 via the launcher (plan.md "Attempt 5 plan"). usage: launch_run.sh <run> <outdir>
set -u
run=$1; out=$2; mkdir -p "$out"
cd /home/michael/work/algo-2a
X=openspec/changes/rl-algo-seq-and-adv/examples
A=--rl-allow-unverified-mechanism
R=yeto.rl.algos.gdpo_reward:correctness_reward
case $run in
  gspo_s2) EXTRA=(--rl-optimizer-steps 2 --rl-algorithm-spec $X/gspo.json $A advantage_estimators:gspo $A features:eps_clip $A features:clip_higher);;
  gspo_s1) EXTRA=(--rl-optimizer-steps 1 --rl-algorithm-spec $X/gspo.json $A advantage_estimators:gspo $A features:eps_clip $A features:clip_higher);;
  rpp) EXTRA=(--rl-optimizer-steps 1 --rl-algorithm-spec $X/rpp.json $A advantage_estimators:reinforce_plus_plus $A features:whiten_advantages);;
  rpp_baseline) EXTRA=(--rl-optimizer-steps 1 --rl-algorithm-spec $X/rpp_baseline.json $A advantage_estimators:reinforce_plus_plus_baseline $A features:whiten_advantages);;
  gdpo) R=yeto.rl.algos.gdpo_reward:reward_func; EXTRA=(--rl-optimizer-steps 1 --rl-algorithm-spec $X/gdpo.json $A features:gdpo $A reward_postprocessors:custom_reward_postprocess);;
esac
PREFIX=algo2a-g1-${run//_/-}
COMMON=(--training-mode rl --rl-engine ports --rl-single-island-no-sync --controller local
 --gpu modal:1xh100 --modal-gpu-exact --cluster-prefix $PREFIX
 --model Qwen/Qwen3-0.6B --model-revision c1899de289a04d12100db370d81485cdf75e47ca
 --data zhuzilin/gsm8k --data-revision 0cbd9f31d91ac21a7613dcbc7fef992adac459ae
 --reward-function $R
 --tuning lora --lora-r 16 --lora-targets all-linear --total-steps 3
 --rollout-batch-size 4 --n-samples-per-prompt 8 --rollout-max-response-len 384 --seq-len 1024
 --inner-lr 1e-5 --seed 17 --apply-chat-template-kwargs '{"enable_thinking": false}' --trust-remote-code)
echo "$PREFIX" > "$out/prefix.txt"; git rev-parse HEAD > "$out/yeto_sha.txt"
# independent reclamation: stop the run's Modal app after 45 min whatever happens here
setsid nohup bash -c "sleep 2700; /tmp/modal-venv/bin/modal app stop -y yeto-$PREFIX" >/dev/null 2>&1 < /dev/null &
echo $! > "$out/watchdog.pid"
# Private image pull credentials: decoded from the ghcr.io entry of ~/.docker/config.json
# into this process's environment only (never printed or written).
eval "$(python3 - <<'PY'
import base64, json, os, shlex
auth = json.load(open(os.path.expanduser("~/.docker/config.json")))["auths"]["ghcr.io"]["auth"]
user, token = base64.b64decode(auth).decode().split(":", 1)
print(f"export SKYPILOT_DOCKER_USERNAME={shlex.quote(user)} SKYPILOT_DOCKER_PASSWORD={shlex.quote(token)} SKYPILOT_DOCKER_SERVER=ghcr.io")
PY
)"
PYTHONPATH=$PWD timeout 2400 /home/michael/work/gpu-head/venv/bin/python -m yeto.cli launch "${COMMON[@]}" "${EXTRA[@]}" > "$out/launch.log" 2>&1
echo "launcher rc=$?" | tee "$out/launcher_rc.txt"
PYTHONPATH=$PWD timeout 300 /home/michael/work/gpu-head/venv/bin/python -m yeto.cli down $PREFIX > "$out/down.log" 2>&1
/tmp/modal-venv/bin/modal app stop -y yeto-$PREFIX >> "$out/down.log" 2>&1
/tmp/modal-venv/bin/modal app list 2>&1 | grep -E "yeto-$PREFIX|App ID" > "$out/app_list_after.txt"

#!/bin/bash
# S27 IMG: GPU check of the ports image ed5f64f-2fa8801 (Miles ed5f64fd7, B8).
# Qwen3-0.6B full parameters, 1 colocated H100!, Adam state on the host
# (--rl-optimizer-offload-fraction 1.0), a round cut every round, 2 rounds.
# Pass: rl_round_cut events for both rounds and cut shards in the store.
# Review: infra-drafts/SESSION27/gpu-review/IMG.md. usage: run.sh RUN_ID (s27-img-*)
set -u
P=${1:?run id}; case $P in s27-img-*) ;; *) echo "abort: run id must start with s27-img-"; exit 65;; esac
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../.." && pwd)
R=/home/michael/work/s1-runs/s27-img/$P
STOREVOL=yeto-eval-store
ARGS="launch --training-mode rl --rl-engine ports --preflight-threads off --modal-gpu-exact --modal-retries 0 \
--modal-cpu 8 --modal-memory 64 --modal-tape-volume yeto-event-tapes --controller local --rl-observe-timeline \
--model Qwen/Qwen3-0.6B --data zhuzilin/gsm8k --data-revision 0cbd9f31d91ac21a7613dcbc7fef992adac459ae \
--rl-checkpoint-store modal-volume://$STOREVOL/$P --reward-function gsm8k_reward:score --fragments 1 --pipeline 1 \
--rollout-batch-size 8 --n-samples-per-prompt 4 --rollout-max-response-len 256 --seq-len 512 --inner-lr 1e-6 \
--seed 17 --total-steps 2 --trust-remote-code --cluster-prefix $P --gpu modal:1xh100 \
--rl-single-island-no-sync --tuning full --rl-optimizer-offload-fraction 1.0 --rl-cut-every 1"
mkdir -p "$(dirname $R)"
export P R REPO ARGS STOREVOL ISLANDS=1 HARD=${HARD:-1500} STALL_S=600 PY=/home/michael/work/gpu-head/venv/bin/python
exec bash "$HERE/miles_local.sh"

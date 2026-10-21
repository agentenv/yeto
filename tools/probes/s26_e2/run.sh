#!/bin/bash
# S26 E2 async defaults sweep: one run of one group. Same setup as S26 E0.3 v2 (L0/L1), so
# E0.3's v2 runs are this sweep's age-0 and age-1/TIS-2 groups: Qwen3-0.6B on GSM8K, Miles,
# full parameters, Modal H100!, 1 train + 1 engine card (fixed partition, generation overlaps
# training through Miles fully_async), 10 rounds x 256 samples (32 prompts x 8), token TIS.
# Review: infra-drafts/SESSION26/gpu-review/E2.md.
#
# usage: SEED=17 run.sh GROUP RUN_ID      (RUN_ID starts with s26-e2-)
#   GROUP aAtT  max policy age A (1|2|4), TIS upper T (2|5); buffer = A rounds of groups
#         (--rl-async-buffer-rounds A) so generation can run A rounds ahead
#   GROUP aAtTbK  same with a buffer of K rounds instead of A
#   GROUP aAais5  age A, adaptive TIS (AIS, yeto.rl.algos.adaptive_tis), upper 5
# env: SEED (default 17), DRY=1 (launcher --dry-run only), HARD (s)
set -u
G=${1:?group}; P=${2:?run id}; case $P in s26-e2-*) ;; *) echo "abort: run id must start with s26-e2-"; exit 65;; esac
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../.." && pwd)
D=tools/probes/s26_e2
[[ $G =~ ^a([124])(t2|t5|ais5)(b([124]))?$ ]] || { echo "abort: group $G"; exit 65; }
AGE=${BASH_REMATCH[1]}; BUF=${BASH_REMATCH[4]:-$AGE}
case ${BASH_REMATCH[2]} in t2) SPEC=algo-tis2-age4.json;; t5) SPEC=algo-tis5-age4.json;; ais5) SPEC=algo-ais5-age4.json;; esac
ARGS="launch --training-mode rl --rl-engine ports --modal-gpu-exact --modal-retries 0 --modal-cpu 8 \
--modal-tape-volume yeto-event-tapes --model Qwen/Qwen3-0.6B --model-revision c1899de289a04d12100db370d81485cdf75e47ca \
--rl-allow-local-data --data tools/probes/s26_e03/gsm8k-tasks512.jsonl --reward-function yeto.rl.gsm8k_reward:score --pipeline 1 \
--n-samples-per-prompt 8 --rollout-max-response-len 1024 --seq-len 1536 --inner-lr 1e-5 --seed ${SEED:-17} \
--total-steps 10 --apply-chat-template-kwargs '{\"enable_thinking\": false}' --trust-remote-code \
--rl-algorithm-spec $D/$SPEC --rl-placement fixed-partition --rl-rollout-gpus 1 --cluster-prefix $P \
--controller local --preflight-threads off --rl-observe-timeline --gpu modal:2xh100 \
--rl-single-island-no-sync --tuning full --modal-memory 64 --rollout-batch-size 32 \
--rl-max-policy-age $AGE --rl-async-buffer-rounds $BUF"
grep -q "S26 E2 异步默认值" /home/michael/work/infra-drafts/gpu-spend.md || [ -n "${DRY:-}" ] || { echo "abort: E2 not pre-registered"; exit 68; }
export P REPO ARGS PY=/home/michael/work/gpu-head/venv/bin/python
export R=/home/michael/work/s1-runs/s26-e2/$P STOREVOL=yeto-eval-store ISLANDS=1 HARD=${HARD:-2400}
# no-progress watchdog: no Miles training step for 15 min (first step: 20 min) -> stop the app
export STALL_S=900 FIRST_STEP_S=1200
[ -n "${DRY:-}" ] && R=/tmp/$P-plan && rm -rf $R
# no checkpoint store, as in E0.3 v2 (keeps the setup identical to the reused baseline runs)
exec bash "$HERE/miles_local.sh"

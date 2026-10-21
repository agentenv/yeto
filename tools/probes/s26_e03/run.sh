#!/bin/bash
# S26 E0.3 bias-stacking ablation: one run of one layer. Qwen3-0.6B on GSM8K, Miles, full
# parameters, Modal H100!, 10 rounds x 64 samples (8 prompts x 8), TIS token-level upper 2.0 in
# every layer, same prompts per round in every layer (gsm8k-tasks256.jsonl interleaves the two
# islands' halves, so one island reads the union of both islands' prompts in the same round).
# Review: infra-drafts/SESSION26/gpu-review/E0.3.md.
#
# usage: SEED=17 run.sh LAYER RUN_ID      (RUN_ID starts with s26-e03-)
#   L0  1 island, 1 train + 1 engine card (fixed partition), synchronous (policy age 0)
#   L1  L0 + fully_async (policy age 1)
#   L2  L0 + 2 islands DiLoCo (E3 config C); the full-parameter sweep needs policy age 0, so this
#       layer stacks on L0, not on L1 (see the review)
#   L3  L2 + FP8 rollout tier 1 (Miles trainer source: block FP8 + ignore list, KV BF16)
#   L3s L1 + FP8 rollout tier 1 (single-island FP8 on top of async; optional)
# env: SEED (default 17), DRY=1 (launcher --dry-run only), HARD (s)
set -u
L=${1:?layer}; P=${2:?run id}; case $P in s26-e03-*) ;; *) echo "abort: run id must start with s26-e03-"; exit 65;; esac
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../.." && pwd)
D=tools/probes/s26_e03
COMMON="launch --training-mode rl --rl-engine ports --modal-gpu-exact --modal-retries 0 --modal-cpu 8 \
--modal-tape-volume yeto-event-tapes --model Qwen/Qwen3-0.6B --model-revision c1899de289a04d12100db370d81485cdf75e47ca \
--rl-allow-local-data --data $D/gsm8k-tasks256.jsonl --reward-function yeto.rl.gsm8k_reward:score --pipeline 1 \
--n-samples-per-prompt 8 --rollout-max-response-len 1024 --seq-len 1536 --inner-lr 1e-5 --seed ${SEED:-17} \
--total-steps 10 --apply-chat-template-kwargs '{\"enable_thinking\": false}' --trust-remote-code \
--rl-algorithm-spec $D/algo-tis2.json --rl-placement fixed-partition --rl-rollout-gpus 1 --cluster-prefix $P"
FP8="--rl-rollout-quant-tier 1 --rl-rollout-quant-fp8-source trainer"
SINGLE="$COMMON --controller local --preflight-threads off --rl-observe-timeline --gpu modal:2xh100 \
--rl-single-island-no-sync --tuning full --modal-memory 64 --rollout-batch-size 8"
# E3 config C (E3-REPORT.md): fragment pipeline + background checkpoint + bf16 push +
# checkpoint every 4P, 8 WAN streams; link probe (E3b, 2509fdef): an island below 20 MB/s
# either way is replaced (s26-e03-L2-a: the initial broadcast stalled at ~2.6 MB/s)
MULTI="$COMMON --controller head --syncer-region nebius/eu-north1 --preflight-threads wait \
--preflight-thread-start 2950 --rl-stall-timeout 2400 --on-demand --modal-memory-gib 64 --no-island-relaunch \
--gpu modal:2xh100,modal:2xh100 --rl-sync-preset full-parameter \
--rl-parameter-layout-sha256 3d5a0936038d8c9587afffc4bc043769d094695c8916ebef8696c0758d83a3b5 --fragments 2 \
--rl-sync-fragment-pipeline 2 --rl-sync-async-checkpoint --rl-sync-push-dtype bf16 --rl-sync-checkpoint-every 8 \
--wan-streams 8 --wan-probe-min-mbps 20 --wan-probe-min-down-mbps 20 --wan-probe-mib 128 --rl-max-policy-age 0 --rollout-batch-size 4 --rl-island-tasks 'l0=gsm1-128;l1=gsm129-256'"
case $L in
  L0)  ARGS="$SINGLE" ;;
  L1)  ARGS="$SINGLE --rl-max-policy-age 1" ;;
  L3s) ARGS="$SINGLE --rl-max-policy-age 1 $FP8" ;;
  L2)  ARGS="$MULTI" ;;
  L3)  ARGS="$MULTI $FP8" ;;
  *) echo "abort: layer $L"; exit 65 ;;
esac
export P REPO ARGS PY=/home/michael/work/gpu-head/venv/bin/python
case $L in
  L2|L3)
    export HARD=${HARD:-2700} PLAN_ONLY=${DRY:+1}
    exec bash "$HERE/head_local.sh" ;;
  *)
    grep -q "E0.3 偏差叠加消融" /home/michael/work/infra-drafts/gpu-spend.md || { echo "abort: E0.3 not pre-registered"; exit 68; }
    export R=/home/michael/work/s1-runs/s26-e03/$P STOREVOL=yeto-eval-store ISLANDS=1 HARD=${HARD:-2400}
    [ -n "${DRY:-}" ] && R=/tmp/$P-plan && rm -rf $R
    ARGS="$ARGS --rl-checkpoint-store modal-volume://$STOREVOL/$P"
    exec bash "$REPO/tools/probes/s26_e6/miles_local.sh" ;;
esac

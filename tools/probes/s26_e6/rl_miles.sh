#!/bin/bash
# S26 E6 step 3 (Miles): one RL run of a group on Modal H100!, Qwen3-4B-Instruct-2507 on GSM8K,
# 10 rounds x 32 prompts x 4 samples, TIS token-level upper 2.0 (algo-tis2.json, as E0.1).
# Review: infra-drafts/SESSION26/gpu-review/E6.md. usage: rl_miles.sh GROUP RUN_ID
#   t0-lora / t1-lora   1 colocated island, LoRA r=32, tier 0 / tier 1 (SGLang online FP8 + ignore list)
#   t1n-lora            1 colocated island, LoRA r=32, tier 1 trainer source (engines on Miles'
#                       block-FP8 base + ignore list)
#   t0-full / t1n-full  1 island, full parameters, 1 training card + 1 dedicated engine card
#                       (engines on the island need DP=1; Adam state on the host, 200 GiB)
#                       (fixed partition): Miles' own full-parameter update every round; tier 0 /
#                       tier 1 trainer source (Miles block-quantizes each update, ignore list)
#   t0-pair / t2-pair   trainer island (2 H100) + sampler island (1 actor + 1 engine H100), full
#                       parameters, tier 0 / tier 2
# env: DRY=1 (launcher --dry-run only), SEED (default 17; the repeat of a group uses another
#      seed, since Miles is nearly deterministic here and two equal-seed runs give one point)
set -u
G=${1:?group}; P=${2:?run id}; case $P in e6-*) ;; *) echo "abort: run id must start with e6-"; exit 65;; esac
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../.." && pwd)
R=/home/michael/work/s1-runs/s26-e6/$P
STOREVOL=yeto-eval-store
COMMON="launch --training-mode rl --rl-engine ports --preflight-threads off --modal-gpu-exact --modal-retries 0 \
--modal-cpu 8 --modal-tape-volume yeto-event-tapes --controller local --rl-observe-timeline \
--model Qwen/Qwen3-4B-Instruct-2507 --model-revision cdbee75f17c01a7cc42f958dc650907174af0554 \
--data zhuzilin/gsm8k --data-revision 0cbd9f31d91ac21a7613dcbc7fef992adac459ae \
--rl-checkpoint-store modal-volume://$STOREVOL/$P --reward-function gsm8k_reward:score --fragments 1 --pipeline 1 \
--rollout-batch-size 32 --n-samples-per-prompt 4 --rollout-max-response-len 512 --seq-len 1024 --inner-lr 1e-5 \
--seed ${SEED:-17} --total-steps 10 --trust-remote-code --rl-algorithm-spec tools/probes/s26_e6/algo-tis2.json"
LORA="--tuning lora --lora-r 32 --modal-memory 64"
ISLANDS=1
case $G in
  t0-lora)  ARGS="$COMMON --cluster-prefix $P --gpu modal:1xh100 --rl-single-island-no-sync $LORA" ;;
  t1-lora)  ARGS="$COMMON --cluster-prefix $P --gpu modal:1xh100 --rl-single-island-no-sync $LORA --rl-rollout-quant-tier 1" ;;
  t1n-lora) ARGS="$COMMON --cluster-prefix $P --gpu modal:1xh100 --rl-single-island-no-sync $LORA --rl-rollout-quant-tier 1 --rl-rollout-quant-fp8-source trainer" ;;
  t0-full|t1n-full)
    Q=""; [ $G = t1n-full ] && Q="--rl-rollout-quant-tier 1 --rl-rollout-quant-fp8-source trainer"
    ARGS="$COMMON --cluster-prefix $P --gpu modal:2xh100 --rl-single-island-no-sync --tuning full \
--modal-memory 200 --rl-optimizer-offload-fraction 1.0 --rl-placement fixed-partition --rl-rollout-gpus 1 $Q"
    # the round cut's DP-invariant state export refuses the host (precision-aware) optimizer
    # (e6-t1n-full-b): this group runs without the checkpoint store
    ARGS=${ARGS/--rl-checkpoint-store modal-volume:\/\/$STOREVOL\/$P /} ;;
  t0-pair|t2-pair)
    # Two launches on one store (as S25 C3): the trainer island reads samples and starts no
    # engine (1 card); a full-parameter sampler island needs dedicated engines (fixed
    # partition, 1 actor + 1 engine card). One launch with --rl-island-sample-role cannot
    # give the two islands different placements for full parameters.
    ROLE="$COMMON --rl-single-island-no-sync --tuning full --modal-memory 128 --rl-max-policy-age 1"
    Q=""; [ $G = t2-pair ] && Q="--rl-rollout-quant-tier 2"
    # full-parameter Adam state of 4B does not fit next to activations on one H100
    # (e6-t0-pair-a: CUDA OOM in round 2), and with it on the host the worker was killed
    # (e6-t0-pair-c): 2 training cards shard it (distributed optimizer)
    ARGS="$ROLE --cluster-prefix $P --gpu modal:2xh100 --rl-sample-role trainer --rl-sample-wait-s 900 $Q"
    SARGS="$ROLE --cluster-prefix $P-s --gpu modal:2xh100 --rl-placement fixed-partition --rl-rollout-gpus 1 \
--rl-sample-role sampler --rl-sampler-max-backlog-rounds 2 --total-steps 12 $Q"
    ISLANDS=2 ;;
  *) echo "abort: group $G"; exit 65 ;;
esac
mkdir -p "$(dirname $R)"
export P R REPO ARGS ${SARGS:+SARGS} STOREVOL ISLANDS HARD=${HARD:-3600} PY=/home/michael/work/gpu-head/venv/bin/python
exec bash "$HERE/miles_local.sh"

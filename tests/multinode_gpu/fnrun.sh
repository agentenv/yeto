#!/usr/bin/env bash
# Flash-Next elastic launch argument set (fn-elastic). PRINT-ONLY: this script never
# calls sky; it renders the `yeto launch` argv for a case so the plan in
# infra-drafts/FN-ELASTIC-GPU-PLAN.md can be reviewed and dry-checked on CPU.
#   fn32s : stage B, 4x8 H200, initial FN-T16R8S8 (1 TP8/EP8 engine + 8 standby), recommend mode
#   fn32b : stage B, 4x8 H200, initial FN-T16R16S0 (2 engines), recommend mode
# env: IMAGE (digest-pinned --rl-image), ATTEST (attestation json; absent = nothing certified),
#      COSTS (5.7 edge cost table; absent = recommend holds with "unknown transition cost"),
#      STEPS (default 12), PREFIX (cluster prefix, default fn).
set -eu
C=${1:?case}; D=$(cd "$(dirname "$0")" && pwd)
STORE="--model-store nebius-fs://computefilesystem-e00nm64w4cqpkqd0ch"
MODEL="--model Qwen/Qwen3.8-Flash-Next --model-revision de4b8e4d43b917e7706784d8bb445c9af86a3540"
LORA="--tuning lora --lora-r 16 --lora-targets all-linear"
PAR="--tensor-parallel 2 --pipeline-parallel 8 --expert-parallel 2 --rollout-num-gpus-per-engine 8"
OBS="--rl-observe-timeline --rl-recommend-mode recommend${COSTS:+ --rl-edge-costs-path $COSTS}${ATTEST:+ --rl-elastic-attestation $ATTEST}"
case $C in
  fn32s) GPU=nebius:4x8xh200@eu-north1; EX="--rl-placement fixed-partition --rl-rollout-gpus 8 --rl-standby-gpus 8 --rl-elastic --rl-elastic-resources $D/resources-fn-4x8.json --rl-elastic-initial-config FN-T16R8S8 $OBS";;
  fn32b) GPU=nebius:4x8xh200@eu-north1; EX="--rl-placement fixed-partition --rl-rollout-gpus 16 --rl-elastic --rl-elastic-resources $D/resources-fn-4x8.json --rl-elastic-initial-config FN-T16R16S0 $OBS";;
  *) echo "unknown case $C" >&2; exit 64;;
esac
echo "launch --controller local --training-mode rl --rl-engine ports --rl-single-island-no-sync --on-demand --gpu $GPU --cluster-prefix ${PREFIX:-fn} --no-island-relaunch ${IMAGE:+--rl-image $IMAGE }$STORE $MODEL --data zhuzilin/gsm8k --data-revision 0cbd9f31d91ac21a7613dcbc7fef992adac459ae --reward-function gsm8k_reward:score $LORA $PAR --fragments 1 --pipeline 1 --rollout-batch-size 8 --n-samples-per-prompt 8 --rollout-max-response-len 1024 --seq-len 2048 --inner-lr 1e-5 --seed 17 --trust-remote-code --total-steps ${STEPS:-12} $EX"

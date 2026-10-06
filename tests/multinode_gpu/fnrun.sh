#!/usr/bin/env bash
# Flash-Next elastic launch argument set (fn-elastic). PRINT-ONLY: this script never
# calls sky; it renders the `yeto launch` argv for a case so the plan in
# infra-drafts/FN-ELASTIC-GPU-PLAN.md can be reviewed and dry-checked on CPU.
#   fn32s : stage B, 4x8 H200, initial FN-T16R8S8 (1 TP8/EP8 engine + 8 standby), recommend mode
#   fn32b : stage B, 4x8 H200, initial FN-T16R16S0 (2 engines), recommend mode
#   fn8s  : stage A, 1x8 H200 small slice: Flash-Next-4layer, colocated, trainer TP2 PP2 EP4
#           + 2 TP4/EP4 engines (run_qwen3_8_next.py 4-layer 8-GPU layout), observe only
#           (recommend needs --rl-elastic, i.e. a fixed partition + resources file)
# --rl-megatron-ref-load: the raw recipe loads <type>_torch_dist (run_qwen3_8_next.py
# --ref-load), pre-converted (scripts/convert_qwen3_8_next.sh) onto the model FS
# mounted at /mnt/yeto-models; the learner refuses a missing/non-release dir.
# env: IMAGE (digest-pinned --rl-image), ATTEST (attestation json; absent = nothing certified),
#      COSTS (5.7 edge cost table; absent = recommend holds with "unknown transition cost"),
#      STEPS (default 12), PREFIX (cluster prefix, default fn),
#      BOOT_ONLY=1 (fn8s only: append --rl-boot-only -- S11 fnboot provisions the FS node, the learner
#      checks everything torch_dist-free, writes FN_BOOT_ONLY_OK and exits 0 so --keep keeps the cluster).
set -eu
C=${1:?case}; D=$(cd "$(dirname "$0")" && pwd)
STORE="--model-store nebius-fs://computefilesystem-e00nm64w4cqpkqd0ch"
TD=/mnt/yeto-models/torch_dist
MODEL="--model Qwen/Qwen3.8-Flash-Next --model-revision de4b8e4d43b917e7706784d8bb445c9af86a3540 --rl-megatron-ref-load $TD/qwen3.8-flash-next_torch_dist"
LORA="--tuning lora --lora-r 16 --lora-targets all-linear --rl-lora-expert-rank 8"
PAR="--tensor-parallel 2 --pipeline-parallel 8 --expert-parallel 2 --rollout-num-gpus-per-engine 8"
OBS="--rl-observe-timeline --rl-recommend-mode recommend${COSTS:+ --rl-edge-costs-path $COSTS}${ATTEST:+ --rl-elastic-attestation $ATTEST}"
case $C in
  fn32s) GPU=nebius:4x8xh200@eu-north1; EX="--rl-placement fixed-partition --rl-rollout-gpus 8 --rl-standby-gpus 8 --rl-elastic --rl-elastic-resources $D/resources-fn-4x8.json --rl-elastic-initial-config FN-T16R8S8 $OBS";;
  fn32b) GPU=nebius:4x8xh200@eu-north1; EX="--rl-placement fixed-partition --rl-rollout-gpus 16 --rl-elastic --rl-elastic-resources $D/resources-fn-4x8.json --rl-elastic-initial-config FN-T16R16S0 $OBS";;
  fn8s) case ${FN_GPU:-h200} in h100|h200) ;; *) echo "abort: FN_GPU must be h100|h200" >&2; exit 64;; esac
        GPU=nebius:1x8x${FN_GPU:-h200}@eu-north1   # FN_GPU=h100: 80GB, see infra-drafts/FN-A-PRELAUNCH-REVIEW.md "H100 变体"
        MODEL="--model CharyZeng/Qwen3.8-Flash-Next-4layer --model-revision d19a6b60c0df8f90faf92c7c592b37df2e15b060 --rl-megatron-ref-load $TD/qwen3.8-flash-next-4layer_torch_dist"
        PAR="--tensor-parallel 2 --pipeline-parallel 2 --expert-parallel 4 --rollout-num-gpus-per-engine 4"
        EX="--rl-placement colocated --rl-offload-train --sglang-mem-fraction-static 0.7 --rl-observe-timeline"
        [ "${BOOT_ONLY:-0}" = 1 ] && EX="$EX --rl-boot-only";;
  *) echo "unknown case $C" >&2; exit 64;;
esac
[ "${BOOT_ONLY:-0}" = 1 ] && [ $C != fn8s ] && { echo "abort: BOOT_ONLY=1 only applies to fn8s" >&2; exit 64; }
echo "launch --controller local --training-mode rl --rl-engine ports --rl-single-island-no-sync --on-demand --gpu $GPU --cluster-prefix ${PREFIX:-fn} --no-island-relaunch ${IMAGE:+--rl-image $IMAGE }$STORE $MODEL --data zhuzilin/gsm8k --data-revision 0cbd9f31d91ac21a7613dcbc7fef992adac459ae --reward-function gsm8k_reward:score $LORA $PAR --fragments 1 --pipeline 1 --rollout-batch-size 8 --n-samples-per-prompt 8 --rollout-max-response-len 1024 --seq-len 2048 --inner-lr 1e-5 --seed 17 --trust-remote-code --total-steps ${STEPS:-12} $EX"

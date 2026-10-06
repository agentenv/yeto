#!/usr/bin/env bash
# Flash-Next formal RL training (fn-train). PRINT/CHECK-ONLY: never calls sky.
# Plan + day-of checklist: infra-drafts/FN-TRAIN-PLAN.md.
#   fntrain.sh print [s|b]  : the `yeto launch` argv. s = initial FN-T16R8S8 (1 TP8 engine +
#                             8 standby, recommend sees an idle node), b = FN-T16R16S0 (default).
#   fntrain.sh convert      : stage B0 HF->torch_dist command for the FULL model (dry-run render
#                             through scripts/convert_qwen3_8_next.sh; run it on a GPU node).
#   fntrain.sh dry          : print + pytest tests/test_rl_fn_train_args.py (launcher dry-run).
# Training recipe follows miles-m3 scripts/run_qwen3_8_next.py (dapo-math-17k, math reward,
# n=8, lr 1e-6, max response 4096; ports path decays LR linearly to 0 over --total-steps) on the ports path so the elastic hook is live.
# env: IMAGE (digest-pinned --rl-image), RUN (run id; checkpoint prefix), STEPS (default 200 = LR-decay
#      horizon; LR decays linearly to 0 over it, so never shorten it to the gate), RBS (rollout prompts/step, default 16), COSTS (5.7 edge cost table; absent =
#      recommend holds "unknown transition cost"), WINDOW (--rl-elastic-window-s, set to ~2x the
#      measured round time), ATTEST (absent = nothing certified), PREFIX (default fnt).
set -eu
MODE=${1:?print|convert|dry}; SHAPE=${2:-b}
D=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$D/../.." && pwd)
TD=/mnt/yeto-models/torch_dist
REF=$TD/qwen3.8-flash-next_torch_dist
RUN=${RUN:-fnt-$(date -u +%Y%m%d)a}
STORE="--model-store nebius-fs://computefilesystem-e00nm64w4cqpkqd0ch"
MODEL="--model Qwen/Qwen3.8-Flash-Next --model-revision de4b8e4d43b917e7706784d8bb445c9af86a3540 --rl-megatron-ref-load $REF"
DATA="--data zhuzilin/dapo-math-17k --data-revision 2e65612930298bde4c5d58fd97b3f23a483aaff9 --rl-prompt-column prompt --rl-label-column label --reward-function yeto.rl.math_reward:reward_func"
LORA="--tuning lora --lora-r 16 --lora-targets all-linear --rl-lora-expert-rank 8"
PAR="--tensor-parallel 2 --pipeline-parallel 8 --expert-parallel 2 --rollout-num-gpus-per-engine 8"
HYP="--rollout-batch-size ${RBS:-16} --n-samples-per-prompt 8 --rollout-max-response-len 4096 --seq-len 8192 --inner-lr 1e-6 --seed 17"
OBS="--rl-observe-timeline --rl-recommend-mode recommend${WINDOW:+ --rl-elastic-window-s $WINDOW}${COSTS:+ --rl-edge-costs-path $COSTS}${ATTEST:+ --rl-elastic-attestation $ATTEST}"
CKPT="--rl-checkpoint-store s3://yeto-rl-ckpt-ddde6f79/fn-train/$RUN"
case $SHAPE in
  s) PLACE="--rl-placement fixed-partition --rl-rollout-gpus 8 --rl-standby-gpus 8 --rl-elastic-initial-config FN-T16R8S8";;
  b) PLACE="--rl-placement fixed-partition --rl-rollout-gpus 16 --rl-elastic-initial-config FN-T16R16S0";;
  *) echo "unknown shape $SHAPE (s|b)" >&2; exit 64;;
esac
argv() {
  echo "launch --controller local --training-mode rl --rl-engine ports --rl-single-island-no-sync --on-demand --gpu nebius:4x8xh200@eu-north1 --cluster-prefix ${PREFIX:-fnt} --no-island-relaunch --modal-retries 0 ${IMAGE:+--rl-image $IMAGE }$STORE $MODEL $DATA $LORA $PAR --fragments 1 --pipeline 1 $HYP --trust-remote-code --total-steps ${STEPS:-200} --rl-elastic --rl-elastic-resources $D/resources-fn-4x8.json $PLACE $CKPT $OBS"
}
case $MODE in
  print) argv;;
  # TP2 PP1 (4layer CI layout) would put ~180B/2 params x 2 B ~= 180 GB on each rank > 141 GB
  # (estimate, unverified): split the 48 layers over PP4 -> ~45 GB/rank. torch_dist re-shards at load.
  convert) YETO_Q38N_NUM_NODES=4 YETO_Q38N_CONVERT_PP=${CONVERT_PP:-4} bash "$ROOT/scripts/convert_qwen3_8_next.sh" --variant full \
             --hf-checkpoint "${FN_HF:-/mnt/yeto-models/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/de4b8e4d43b917e7706784d8bb445c9af86a3540}" \
             --output "$REF" --nproc 8 --dry-run;;
  dry) argv; cd "$ROOT" && PYTHONPATH=. "${PY:-python3}" -m pytest -q tests/test_rl_fn_train_args.py;;
  *) echo "unknown mode $MODE" >&2; exit 64;;
esac

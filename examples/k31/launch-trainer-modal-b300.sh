#!/usr/bin/env bash
# Kimi-K3.1 async GRPO, trainer island on Modal B300 (draft; CPU-rehearsed wiring, not GPU-run).
#
# Full-parameter form: the selected routed experts of every MoE layer (64 of 896 is the
# production setting; 128 also works) train in place next to the frozen MXFP4 experts
# (Miles --partial-experts-selection). LoRA is for infrastructure tests only.
#
# Stage 1 (default): a 4-layer smoke checkpoint you cut yourself
# (scripts/prune_layers.py --layers 4 of the K3.1-Pro checkpoint) on one B300 node, TP1 EP8 (DP 8).
# Stage 2 (STAGE=2): the full 93-layer Kimi-K3.1-Pro on a B300 cluster, 6 nodes x 8 = 48 GPUs as
# PP6 x TP8 x EP8 (DP 1; the synced forms need DP 1 on the k31 pin). 93 layers do not split
# evenly over PP6: yeto derives the uneven split first 17 / last 16 (4 middle stages x 15,
# --decoder-first/last-pipeline-num-layers) and translate_run_config refuses any split that
# does not add up before a GPU starts.
#
# Forms (FORM):
#   no-sync (default): one trainer island, --rl-single-island-no-sync; it releases its own
#            policy.
#   synced-1: one trainer island behind the Rust syncer (head + syncer; learners 1 / quorum 1,
#            the merge is the identity): the decoupled DiLoCo path with a single learner.
#   synced-2: two trainer islands behind the syncer: each reads its own buffer lane, pushes its
#            full-parameter pseudo-gradient (the selected experts' parameters) at every
#            asynchronous round, the syncer merges them and learner 0 releases the merged policy.
# Both synced forms keep --tuning full (the K3.1 recipe is full-parameter only): the
# full-parameter preset with asynchronous rounds and a parameter-layout sha; they take no
# --rl-allow-unverified-mechanism (a run with outer sync uses declared mechanisms only), so
# the algorithm spec declares the staleness correction itself (algorithm-spec.async-age2-tis.json).
#
# The samplers are National Compute MI355X nodes (examples/k31/sampler-spec.nc-mi355x.json):
# this island reads their groups from the object store and releases every policy into it
# with the selection's sha256 in the release manifest. Every value below is a placeholder;
# the expert_selection.json comes from the sampler's routed-expert statistics, or, for a first
# run before any statistics exist, a uniform selection:
#   python -m yeto.rl.adapters.miles.models.partial_experts.selection --model kimi_k31 \
#     --num-experts 896 --num-trainable 64 --ep-size 8 --moe-layer-offset 1 \
#     --mode uniform --num-moe-layers 92 --output expert_selection.json
#   (a 4-layer cut has 3 MoE layers: --num-moe-layers 3 --skip-layer-check)
# The dry run (append --dry-run) validates the selection against --expert-parallel and prints a
# rough per-GPU memory estimate (kimi_k31_full.memory), with warnings for long sequences.
# 262k context: SEQ_LEN=262144; the response length follows SEQ_LEN - PROMPT_LEN.
# Round cuts are off (--rl-round-cuts off): a full-parameter cut holds bf16 weights, fp32 master and
# both Adam moments (about 5 TB). A relaunch continues from the store's newest release with a fresh
# optimizer: re-running this script is enough (every trainer node runs init_from_release
# --if-released before Ray and loads the newest release when the store holds one; a first launch
# starts from the base). A trainer that finds releases but loaded the base refuses to start.
set -euo pipefail

STAGE=${STAGE:-1}
FORM=${FORM:-no-sync}
: "${MODEL:?HF repo or path of the Kimi-K3.1 checkpoint (stage 1: the 4-layer cut)}"
: "${MODEL_REVISION:?pinned revision}"
: "${DATA:?prompt dataset}"
: "${DATA_REVISION=}"   # pinned data revision (HF datasets); a trainer island never reads --data rows
: "${STORE_URI:?s3://<bucket>/<run>}"
: "${STORE_ENDPOINT:?https://<account>.r2.cloudflarestorage.com}"
: "${STORE_SECRET:?Modal Secret holding AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY}"
: "${EXPERT_SELECTION:?path of expert_selection.json (sha256 in every release manifest)}"
: "${RL_IMAGE:?docker:REPO@sha256:DIGEST of the k31 Miles image}"
HERE=$(cd "$(dirname "$0")" && pwd)

if [ "$STAGE" = "1" ]; then
  ISLAND=${ISLAND:-modal:1x8xb300}      # one node: 8 x B300, TP1 EP8
  TP=${TP:-1}; PP=${PP:-1}
  SEQ_LEN=${SEQ_LEN:-12288}            # PROMPT_LEN 4096 + the sampler spec's max_tokens 8192
  TOTAL_STEPS=${TOTAL_STEPS:-8}
else
  ISLAND=${ISLAND:-modal:6x8xb300}      # whole-node cluster, RDMA (yeto.modal_runner clustered)
  TP=${TP:-8}; PP=${PP:-6}             # PP6 x TP8 x EP8 = 48 GPUs, DP 1 (split 17 / 4 x 15 / 16)
  SEQ_LEN=${SEQ_LEN:-16384}
  TOTAL_STEPS=${TOTAL_STEPS:-64}
fi
PROMPT_LEN=${PROMPT_LEN:-4096}                    # prompt budget inside SEQ_LEN
RESPONSE_LEN=${RESPONSE_LEN:-$((SEQ_LEN - PROMPT_LEN))}

case "$FORM" in
  no-sync)
    GPU=${GPU:-$ISLAND}
    SYNC_FLAGS=(--rl-single-island-no-sync)
    SPEC=${SPEC:-$HERE/algorithm-spec.async-age2.json}
    # score_centering/tis is an unverified mechanism: allowed on a single island without outer sync
    LOSS_FLAGS=(--rl-loss score_centering --rl-score-centering-is tis
                --rl-allow-unverified-mechanism losses:score_centering
                --rl-allow-unverified-mechanism features:rollout_logprobs_as_old
                --rl-allow-unverified-mechanism features:no_grpo_std_normalization
                --rl-allow-unverified-mechanism loss_aggregations:token)
    ;;
  synced-1|synced-2)
    # One or two trainer islands behind the Rust syncer (the launcher starts the head + syncer).
    # The syncer merges the islands' full-parameter pseudo-gradients each asynchronous round;
    # with one island it runs learners=1/quorum=1 and the merge is the identity.
    if [ "$FORM" = "synced-2" ]; then GPU=${GPU:-$ISLAND,$ISLAND}; else GPU=${GPU:-$ISLAND}; fi
    : "${PARAMETER_LAYOUT_SHA256:?sha256 of the full-parameter layout (--rl-parameter-layout-sha256)}"
    SYNC_FLAGS=(--rl-sync-preset full-parameter --rl-sync-full-parameter-rounds async
                --rl-parameter-layout-sha256 "$PARAMETER_LAYOUT_SHA256"
                --rl-sync-quorum 1 --rl-sync-final-cut ack)
    SPEC=${SPEC:-$HERE/algorithm-spec.async-age2-tis.json}
    LOSS_FLAGS=()                      # declared mechanisms only: the spec carries the tis correction
    ;;
  *) echo "FORM must be no-sync, synced-1 or synced-2 (got $FORM)" >&2; exit 2 ;;
esac

yeto launch --training-mode rl \
  --gpu "$GPU" --cluster-prefix k31-stage"$STAGE"-"$FORM" \
  --rl-image "$RL_IMAGE" \
  --model "$MODEL" --model-revision "$MODEL_REVISION" \
  --data "$DATA" ${DATA_REVISION:+--data-revision "$DATA_REVISION"} \
  --rl-prompt-column prompt --rl-label-column answer \
  --reward-function yeto.rl.numeric_answer_reward:score \
  --trust-remote-code --tuning full --rl-model-recipe kimi_k31 --rl-expert-selection "$EXPERT_SELECTION" \
  --expert-parallel 8 --tensor-parallel "$TP" --pipeline-parallel "$PP" --context-parallel "${CP:-1}" \
  --rl-sample-role trainer "${SYNC_FLAGS[@]}" --controller local \
  --rl-checkpoint-store "$STORE_URI" --rl-object-store-endpoint "$STORE_ENDPOINT" \
  --rl-round-cuts off \
  --modal-store-secret "$STORE_SECRET" \
  --inner-lr "${INNER_LR:-1e-6}" --rl-max-policy-age 2 --rl-algorithm-spec "$SPEC" \
  --rl-rollout-top-logprobs-num 128 ${LOSS_FLAGS[@]+"${LOSS_FLAGS[@]}"} \
  --rollout-batch-size 32 --n-samples-per-prompt 8 --rollout-max-response-len "$RESPONSE_LEN" \
  --seq-len "$SEQ_LEN" --total-steps "$TOTAL_STEPS" \
  --rl-release-delta-full-every 8 \
  "$@"

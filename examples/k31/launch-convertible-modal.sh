#!/usr/bin/env bash
# Kimi-K3.1 convertible island on one Modal B300 node (draft; CPU-rehearsed wiring, not GPU-run).
#
# Production flow: the N-card node (this script) and the A-card NC MI355X nodes
# (examples/k31/sampler-node-nc-mi355x.sh) all start as engine samplers writing their own shard
# of the same object-store buffer. After SAMPLE_MINUTES (or SAMPLE_GROUPS groups, or the store
# marker convertible/<node>/switch.json written by an operator) this node stops its SGLang server
# and runs the Miles trainer island on the same GPUs: it reads every island's shard from the
# newest release's trained buffer position (round cuts are off: --rl-round-cuts off; before the
# trainer command runs, the container loads the store's newest release when it holds one, so a
# re-run of the spec continues from it), trains
# (score_centering + GRPO advantage, --rl-max-policy-age gating the staleness) and releases whole checkpoints with the selection sha into the store; the A-card
# samplers keep generating and pull each release. With THEN_SAMPLER=1 the node comes back as a
# sampler on the newest release when the trainer phase ends.
#
# The container runs `python -m yeto.rl.engine.role_switch --spec SPEC`, where SPEC is the
# engine sampler spec of this node (sampler-spec.nc-mi355x.json shaped, platform cuda) plus a
# role_switch section whose trainer.command is the trainer island's learner command. The learner
# command is what `yeto launch --training-mode rl ... --rl-sample-role trainer --dry-run` prints
# for the flags below (launch-trainer-modal-b300.sh): copy it into the spec unchanged.
# Every value below is a placeholder; nothing site-specific is written here.
set -euo pipefail

: "${MODEL:?HF repo or path of the Kimi-K3.1 checkpoint (stage 1: a 4-layer cut from scripts/prune_layers.py)}"
: "${MODEL_REVISION:?pinned revision}"
: "${DATA:?prompt dataset}"
: "${DATA_REVISION:?pinned data revision}"
: "${STORE_URI:?s3://<bucket>/<run>}"
: "${STORE_ENDPOINT:?https://<account>.r2.cloudflarestorage.com}"
: "${STORE_SECRET:?Modal Secret holding AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY}"
: "${EXPERT_SELECTION:?path of expert_selection.json (sha256 in every release manifest)}"
: "${RL_IMAGE:?docker:REPO@sha256:DIGEST of the k31 Miles image}"
: "${SAMPLER_SPEC:?engine sampler spec of this node (sglang-server, platform cuda, buffer shard = NODE)}"
NODE=${NODE:-n-island-0}
SAMPLE_MINUTES=${SAMPLE_MINUTES:-30}      # sampler phase budget; 0 = groups / marker only
SAMPLE_GROUPS=${SAMPLE_GROUPS:-0}         # or: switch after this many groups written by this node
THEN_SAMPLER=${THEN_SAMPLER:-0}           # 1: back to sampler on the released weights
GPU=${GPU:-modal:1x8xb300}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-/tmp/k31-convertible}
mkdir -p "$OUT"

# 1. The trainer island's learner command (dry run of the trainer launch; unchanged flags of
#    launch-trainer-modal-b300.sh). The printed command becomes role_switch.trainer.command.
yeto launch --training-mode rl --dry-run \
  --gpu "$GPU" --cluster-prefix k31-conv \
  --rl-image "$RL_IMAGE" \
  --model "$MODEL" --model-revision "$MODEL_REVISION" \
  --data "$DATA" --data-revision "$DATA_REVISION" \
  --rl-prompt-column prompt --rl-label-column answer \
  --reward-function yeto.rl.numeric_answer_reward:score \
  --trust-remote-code --tuning full --rl-model-recipe kimi_k31 --rl-expert-selection "$EXPERT_SELECTION" \
  --expert-parallel 8 --tensor-parallel 1 --pipeline-parallel 1 \
  --rl-sample-role trainer --rl-single-island-no-sync --controller local \
  --rl-checkpoint-store "$STORE_URI" --rl-object-store-endpoint "$STORE_ENDPOINT" \
  --rl-round-cuts off \
  --modal-store-secret "$STORE_SECRET" \
  --inner-lr "${INNER_LR:-1e-6}" --rl-max-policy-age 2 --rl-algorithm-spec "$HERE/algorithm-spec.async-age2.json" \
  --rl-loss score_centering --rl-score-centering-is tis --rl-rollout-top-logprobs-num 128 \
  --rl-allow-unverified-mechanism losses:score_centering \
  --rl-allow-unverified-mechanism features:rollout_logprobs_as_old \
  --rl-allow-unverified-mechanism features:no_grpo_std_normalization \
  --rl-allow-unverified-mechanism loss_aggregations:token \
  --rollout-batch-size 32 --n-samples-per-prompt 8 --rollout-max-response-len 8192 \
  --seq-len "${SEQ_LEN:-8192}" --total-steps "${TOTAL_STEPS:-8}" \
  --rl-release-delta-full-every 8 \
  "$@" > "$OUT/trainer-dry-run.txt"

# 2. The convertible spec: the sampler spec plus the role_switch section. The trainer command is
#    the learner line of the dry run (the line that runs yeto.rl.adapters.miles.island_entry).
python - "$SAMPLER_SPEC" "$OUT/trainer-dry-run.txt" "$OUT/convertible-spec.json" <<'PY'
import json, shlex, sys
spec = json.load(open(sys.argv[1]))
learner = next(l for l in open(sys.argv[2]) if "yeto.rl.adapters.miles.island_entry" in l)
spec["role_switch"] = {
    "node": __import__("os").environ.get("NODE", "n-island-0"),
    "after_seconds": (int(__import__("os").environ.get("SAMPLE_MINUTES", "30")) * 60) or None,
    "after_groups": int(__import__("os").environ.get("SAMPLE_GROUPS", "0")) or None,
    "marker": True, "store_marker": True,
    "trainer": {"command": shlex.split(learner.strip())},
    "then_sampler": __import__("os").environ.get("THEN_SAMPLER", "0") == "1",
    "max_cycles": 1,
    "device_idle": {"used_mib": 1024, "timeout_s": 300},
}
json.dump(spec, open(sys.argv[3], "w"), indent=2)
PY

# 3. Run the island in the k31 image on one node: sampler (SGLang server on the 8 GPUs, buffer shard
#    NODE) -> trainer (the learner command above) -> optional sampler. The operator can force the
#    switch earlier with `yeto.rl.engine.role_switch.write_switch_marker` on the store
#    (convertible/<node>/switch.json) or by touching $YETO_ROLE_SWITCH_MARKER in the container.
echo "convertible spec: $OUT/convertible-spec.json"
echo "run in the island container: python -m yeto.rl.engine.role_switch --spec $OUT/convertible-spec.json"

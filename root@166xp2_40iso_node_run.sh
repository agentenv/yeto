#!/bin/bash
# EXP2.40 Iso-C + SGD bake-off (capture ON, 12-sigdigit raw eval loss).
# Product-direction priority-1 per docs/NEXT_OPTIMIZER_PLAN.md.
#
# Question: does spectral preconditioning of the CURRENT-ROUND merged delta
# (IsoLoCo Iso-C, --matrix-merge iso) beat memoryless SGD-0.28 WITHOUT
# reintroducing outer first-moment momentum?
#
# Every iso cell is paired with a same-node/same-toolchain/same-eta plain
# SGD-0.28 (nesterov mu0 eta0.28) anchor so the comparison is apples-to-apples
# at eta0.28 (removes the eta/toolchain confound vs historical eta0.175 refs).
#
# Arms (seed223 shuffle / 223223 training, mu0 eta0.28, lora rank2 alpha4
# unless noted):
#   core:        {iso,sgd028}-h16 / -h64 / -h256
#   sensitivity: {iso,sgd028}-innerlr-hi-h64 (inner-lr 0.002)
#                {iso,sgd028}-rank16-h16 (lora-r 16 alpha 32)
# iso arms capture the syncer probe (--syncer-probe-capture every 1).
set -euxo pipefail
cd ~/yeto
. "$HOME/.cargo/env" || true
. "$HOME/exp2_29_env.sh"   # TORCH_SPEC / TORCH_INDEX / GPU_SLOTS [/ CC]
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

E="experiment-results/EXP2/exp2-40-isobakeoff"
S="s3://yeto-exp-artifacts-533462777468-us-west-2/probecommit-resume-20260710/exp2-40-isobakeoff"
# Clobber-safe exclusive prefix for captures + durable results (the parallel
# mediation node runs `aws s3 sync ... exp2-40-isobakeoff --delete`, which wipes
# the shared prefix; nothing else writes to -capture).
SAFE="s3://yeto-exp-artifacts-533462777468-us-west-2/probecommit-resume-20260710/exp2-40-isobakeoff-capture"
export AWS_DEFAULT_REGION=us-west-2
mkdir -p "$E"
git rev-parse HEAD > "$E/git_commit.txt" 2>/dev/null || cat ~/yeto/GIT_COMMIT > "$E/git_commit.txt" || true
nvidia-smi --query-gpu=index,name,memory.total --format=csv > "$E/gpus.txt"

# run_arm label mu h steps wtokens seed eta ilr mm dnr probe lora_r lora_alpha
run_arm () {
  local label=$1 mu=$2 h=$3 steps=$4 wtokens=$5 seed=$6 eta=$7 ilr=$8
  local mm=$9 dnr=${10} probe=${11} lr=${12:-2} la=${13:-4}
  local tseed="${seed}${seed}"
  if [ -f "$E/$label/report/results.jsonl" ]; then
    echo "[exp2-40c] arm $label already complete, skipping"; return 0
  fi
  local extra=()
  [ "$mm" != "rda" ] && extra+=(--matrix-merge "$mm")
  if python3 -c "import sys;sys.exit(0 if float('$dnr')>0 else 1)"; then
    extra+=(--delta-norm-ref "$dnr")
  fi
  [ "$probe" = "1" ] && extra+=(--syncer-probe-capture --syncer-probe-capture-every 1)
  python scripts/compare_diloco.py \
    --model qwen35-9b --data trl-lib/Capybara --settings m4 \
    --baseline-loss 0.0 --delta-correction none \
    --outer-optimizer nesterov --outer-lr "$eta" --outer-momentum "$mu" \
    --token-budget 700000 --seq-len 128 --micro-batch-size 1 \
    --inner-lr "$ilr" --lora-r "$lr" --lora-alpha "$la" \
    --eval-rows 64 --max-rows 5000 \
    --shuffle-rows-seed "$seed" --training-seed "$tseed" \
    --device cuda --gpu-slots "$GPU_SLOTS" \
    --fixed-window-tokens "$wtokens" --fixed-window-microsteps "$h" \
    --pad-to-fixed-window-tokens --freeze-delta-before-delay \
    --learner-push-delay-ms 0,0,0,0 --learner-delay-jitter-ms 0 \
    --syncer-total-steps "$steps" --learner-max-steps 2500 --strict-quorum \
    "${extra[@]}" \
    --work-dir "$E/$label/work" --report-dir "$E/$label/report"
}

# capture arm: persist syncer_probe to S3 right after, free locally.
arm_cap () {  # same args as run_arm but probe forced 1
  local label=$1
  local cap="$E/$label/work/m4/syncer_probe"
  local s3cap="$SAFE/$label/work/m4/syncer_probe"
  # results.jsonl is the sole completion signal; never delete/rerun a finished
  # arm because a capture is missing. run_arm self-skips if results.jsonl exists.
  run_arm "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" 1 "${12:-2}" "${13:-4}"
  # persist capture to the SAFE prefix, non-fatal (never abort the run on an S3
  # hiccup / clobber race), free local disk on a successful upload.
  if [ -f "$cap/index.jsonl" ]; then
    if aws s3 sync "$cap" "$s3cap/" --quiet --region us-west-2; then
      rm -rf "$cap"
      echo "[exp2-40c] capture for $label persisted to $s3cap and freed"
    else
      echo "[exp2-40c] capture upload for $label failed; keeping local copy"
    fi
  fi
}

# pairs interleaved so each cell finishes iso+anchor together (resume-friendly).
#         label              mu h   steps wtok  seed eta   ilr   mm  dnr [r a]
# --- core ---
arm_cap  iso-h64             0  64  80    8192  223 0.28  0.001 iso 0
run_arm  sgd028-h64          0  64  80    8192  223 0.28  0.001 rda 0 0
arm_cap  iso-h16             0  16  320   2048  223 0.28  0.001 iso 0
run_arm  sgd028-h16          0  16  320   2048  223 0.28  0.001 rda 0 0
arm_cap  iso-h256            0  256 20    32768 223 0.28  0.001 iso 0
run_arm  sgd028-h256         0  256 20    32768 223 0.28  0.001 rda 0 0
aws s3 sync "$E" "$SAFE/" --quiet --region us-west-2 --exclude "*/syncer_probe/*" || true
echo "[exp2-40] CORE-COMPLETE"

# --- sensitivity ---
arm_cap  iso-innerlr-hi-h64  0  64  80    8192  223 0.28  0.002 iso 0
run_arm  sgd028-innerlr-hi-h64 0 64 80    8192  223 0.28  0.002 rda 0 0
arm_cap  iso-rank16-h16      0  16  320   2048  223 0.28  0.001 iso 0 16 32
run_arm  sgd028-rank16-h16   0  16  320   2048  223 0.28  0.001 rda 0 0 16 32
aws s3 sync "$E" "$SAFE/" --quiet --region us-west-2 --exclude "*/syncer_probe/*" || true

# final loss summary straight into the run log (full 17-sigdigit precision).
python3 - "$E" <<'PY'
import json,sys,glob,os
root=sys.argv[1]
print("=== EXP2.40 ISO BAKEOFF LOSSES (17 sigdigit) ===")
for d in sorted(glob.glob(os.path.join(root,"*","report","results.jsonl"))):
    label=d.split("/")[-3]
    rows=[json.loads(l) for l in open(d) if l.strip()]
    # trained arm = highest-m non-base/baseline line
    tr=[r for r in rows if r.get("arm") not in ("base (untrained)","baseline (sync, injected)")]
    r=max(tr,key=lambda x:x.get("m",0)) if tr else max(rows,key=lambda x:x.get("m",0))
    print(f"{label}\t{r['eval_loss']!r}\tarm={r.get('arm')}\tm={r.get('m')}")
PY
aws s3 sync "$E" "$SAFE/" --quiet --region us-west-2 --exclude "*/syncer_probe/*" || true
echo "[exp2-40] COMPLETE"

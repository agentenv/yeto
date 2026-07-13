#!/bin/bash
# EXP2.41 memoryless outer-optimizer bake-off (capture ON, .17g raw eval loss).
# Product-direction candidates per docs/NEXT_OPTIMIZER_PLAN.md, all memoryless
# (no outer first-moment / directional memory), all mu0 eta0.28:
#   worker-snr : same-round cross-worker consensus MERGE + plain SGD-0.28
#   block-rms  : per-tensor beta1=0 second-moment outer opt, global norm-match
#   block-yogi : robust per-block beta1=0 second-moment outer opt
# Each candidate at H=16/64/256, each paired with a same-node/same-eta plain
# SGD-0.28 (nesterov mu0 eta0.28 rda) anchor (removes eta/toolchain confound
# vs the historical eta0.175 refs 1.3519/1.3578/1.3805). Then the two
# highest-sensitivity extra cells (inner-lr 0.002 H64, LoRA rank16 H16) for the
# LEADING candidate (auto-picked by mean core loss). lora rank2 alpha4, seed223
# shuffle / 223223 training, delta-correction none.
set -euxo pipefail
cd ~/yeto
. "$HOME/.cargo/env" || true
. "$HOME/exp2_29_env.sh"   # TORCH_SPEC / TORCH_INDEX / GPU_SLOTS [/ CC]
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

E="experiment-results/EXP2/exp2-41-bakeoff"
S="s3://yeto-exp-artifacts-533462777468-us-west-2/probecommit-resume-20260710/exp2-41-bakeoff"
mkdir -p "$E"
git rev-parse HEAD > "$E/git_commit.txt" 2>/dev/null || cat ~/yeto/GIT_COMMIT > "$E/git_commit.txt" || true
nvidia-smi --query-gpu=index,name,memory.total --format=csv > "$E/gpus.txt"

# run_arm label opt mm h steps wtokens seed eta ilr probe [lora_r lora_alpha]
run_arm () {
  local label=$1 opt=$2 mm=$3 h=$4 steps=$5 wtokens=$6 seed=$7 eta=$8 ilr=$9
  local probe=${10} lr=${11:-2} la=${12:-4}
  local tseed="${seed}${seed}"
  if [ -f "$E/$label/report/results.jsonl" ]; then
    echo "[exp2-41] arm $label already complete, skipping"; return 0
  fi
  local extra=()
  [ "$mm" != "rda" ] && extra+=(--matrix-merge "$mm")
  [ "$probe" = "1" ] && extra+=(--syncer-probe-capture --syncer-probe-capture-every 1)
  python scripts/compare_diloco.py \
    --model qwen35-9b --data trl-lib/Capybara --settings m4 \
    --baseline-loss 0.0 --delta-correction none \
    --outer-optimizer "$opt" --outer-lr "$eta" --outer-momentum 0 \
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
arm_cap () {  # label opt mm h steps wtokens seed eta ilr [lora_r lora_alpha]
  local label=$1
  local cap="$E/$label/work/m4/syncer_probe"
  local s3cap="$S/$label/work/m4/syncer_probe"
  if [ -f "$E/$label/report/results.jsonl" ]; then
    if aws s3 ls "$s3cap/index.jsonl" >/dev/null 2>&1; then
      echo "[exp2-41] $label complete w/ capture in S3, skipping"; return 0
    fi
    if [ ! -f "$cap/index.jsonl" ]; then
      echo "[exp2-41] capture for $label lost; re-running"; rm -rf "$E/$label"
      aws s3 rm "$S/$label/" --recursive --quiet || true
    fi
  fi
  run_arm "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" 1 "${10:-2}" "${11:-4}"
  if [ -f "$cap/index.jsonl" ]; then
    aws s3 sync "$cap" "$s3cap/" --quiet && aws s3 ls "$s3cap/index.jsonl" >/dev/null
    rm -rf "$cap"
    echo "[exp2-41] capture for $label persisted to S3 and freed"
  fi
}

sync_up () { aws s3 sync "$E" "$S/" --delete --quiet --exclude "*/syncer_probe/*"; }


# ==== NODE 1 (sharded): worker-snr + block-rms across H{16,64,256} ====
#        label       opt        mm         h   steps wtok  seed eta   ilr
arm_cap  wsnr-h64    nesterov   worker-snr 64  80    8192  223  0.28  0.001
arm_cap  brms-h64    block-rms  rda        64  80    8192  223  0.28  0.001
arm_cap  wsnr-h16    nesterov   worker-snr 16  320   2048  223  0.28  0.001
arm_cap  brms-h16    block-rms  rda        16  320   2048  223  0.28  0.001
arm_cap  wsnr-h256   nesterov   worker-snr 256 20    32768 223  0.28  0.001
arm_cap  brms-h256   block-rms  rda        256 20    32768 223  0.28  0.001
sync_up
python3 - "$E" <<'PY'
import json,glob,os,sys
root=sys.argv[1]
print("=== NODE1 (wsnr/brms) LOSSES (.17g) ===")
for d in sorted(glob.glob(os.path.join(root,"*","report","results.jsonl"))):
    label=d.split("/")[-3]
    if not (label.startswith("wsnr") or label.startswith("brms")): continue
    rows=[json.loads(l) for l in open(d) if l.strip()]
    tr=[r for r in rows if r.get("arm") not in ("base (untrained)","baseline (sync, injected)")]
    r=max(tr,key=lambda x:x.get("m",0)) if tr else max(rows,key=lambda x:x.get("m",0))
    print(f"{label}\t{r['eval_loss']!r}")
PY
sync_up
echo "[exp2-41] NODE1-COMPLETE"

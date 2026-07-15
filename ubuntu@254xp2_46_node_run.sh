#!/bin/bash
# EXP2.46 3-arm current-anchor causal control (capture ON, .17g raw eval loss).
# docs/ANCHOR_DRIFT_CONTROL.md. Isolates the current-anchor confounder in our
# strict-quorum, non-barrier, current-anchor streaming DiLoCo variant:
#   A. barrier + version-matched   (--barrier-sync --version-matched-anchor)
#   B. non-barrier + version-matched (--version-matched-anchor)
#   C. non-barrier + current-anchor  (--anchor-drift-log only; current impl)
# across the crossover corners {H16-mu0, H16-mu09, H256-mu0, H256-mu05}.
# All else identical (seed 223/223223, m4 strict-quorum, eta0.28 nesterov, rda,
# delta-correction none, lora r2 a4, inner-lr 0.001) -- mirrors exp2-41's
# known-good invocation, adding only the 3-arm flags + H/mu/steps/wtok.
# Anchor-drift instrumentation lands in each arm's syncer event tape
# (tape.jsonl: per-responder anchor_drift_norm / local_delta_norm /
# anchor_drift_ratio / anchor_drift_momentum_cos / anchor_base_resolved).
set -euxo pipefail
cd ~/yeto
. "$HOME/.cargo/env" || true
. "$HOME/exp2_29_env.sh"   # TORCH_SPEC / TORCH_INDEX / GPU_SLOTS [/ CC]
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Build the syncer with the version-matched-anchor code (this commit).
( cd ~/yeto/syncer && cargo build --release -q )

E="experiment-results/EXP2/exp2-46-anchorctl"
S="s3://yeto-exp-artifacts-533462777468-us-west-2/probecommit-resume-20260710/exp2-46-anchorctl"
mkdir -p "$E"
git rev-parse HEAD > "$E/git_commit.txt" 2>/dev/null || true
nvidia-smi --query-gpu=index,name,memory.total --format=csv > "$E/gpus.txt" || true

# run_arm: label arm cell  where
#   arm  in {A,B,C}
#   cell in {h16mu0,h16mu09,h256mu0,h256mu05}
run_arm () {
  local label=$1 arm=$2 cell=$3
  if [ -f "$E/$label/report/results.jsonl" ]; then
    echo "[exp2-46] arm $label already complete, skipping"; return 0
  fi
  # cell -> H, mu, syncer-total-steps, fixed-window-tokens (equal token budget)
  local h mu steps wtok
  case "$cell" in
    h16mu0)   h=16;  mu=0;   steps=320; wtok=2048  ;;
    h16mu09)  h=16;  mu=0.9; steps=320; wtok=2048  ;;
    h256mu0)  h=256; mu=0;   steps=20;  wtok=32768 ;;
    h256mu05) h=256; mu=0.5; steps=20;  wtok=32768 ;;
    *) echo "[exp2-46] unknown cell $cell"; return 1 ;;
  esac
  # arm -> anchoring flags
  local aflags=()
  case "$arm" in
    A) aflags=(--barrier-sync --version-matched-anchor) ;;
    B) aflags=(--version-matched-anchor) ;;
    C) aflags=(--anchor-drift-log) ;;
    *) echo "[exp2-46] unknown arm $arm"; return 1 ;;
  esac
  python scripts/compare_diloco.py \
    --model qwen35-9b --data trl-lib/Capybara --settings m4 \
    --baseline-loss 0.0 --delta-correction none \
    --outer-optimizer nesterov --outer-lr 0.28 --outer-momentum "$mu" \
    --token-budget 700000 --seq-len 128 --micro-batch-size 1 \
    --inner-lr 0.001 --lora-r 2 --lora-alpha 4 \
    --eval-rows 64 --max-rows 5000 \
    --shuffle-rows-seed 223 --training-seed 223223 \
    --device cuda --gpu-slots "$GPU_SLOTS" \
    --fixed-window-tokens "$wtok" --fixed-window-microsteps "$h" \
    --pad-to-fixed-window-tokens --freeze-delta-before-delay \
    --learner-push-delay-ms 0,0,0,0 --learner-delay-jitter-ms 0 \
    --syncer-total-steps "$steps" --learner-max-steps 2500 --strict-quorum \
    --syncer-probe-capture --syncer-probe-capture-every 1 \
    "${aflags[@]}" \
    --work-dir "$E/$label/work" --report-dir "$E/$label/report"
}

# capture arm: persist syncer_probe + event tape to S3, free capture locally.
arm_cap () {  # arm cell
  local arm=$1 cell=$2
  local label="${arm}-${cell}"
  local cap="$E/$label/work/m4/syncer_probe"
  local s3cap="$S/$label/work/m4/syncer_probe"
  if [ -f "$E/$label/report/results.jsonl" ] && aws s3 ls "$s3cap/index.jsonl" >/dev/null 2>&1; then
    echo "[exp2-46] $label complete w/ capture in S3, skipping"; return 0
  fi
  run_arm "$label" "$arm" "$cell"
  if [ -f "$cap/index.jsonl" ]; then
    aws s3 sync "$cap" "$s3cap/" --quiet && aws s3 ls "$s3cap/index.jsonl" >/dev/null
    rm -rf "$cap"
    echo "[exp2-46] capture for $label persisted to S3 and freed"
  fi
}

sync_up () { aws s3 sync "$E" "$S/" --delete --quiet --exclude "*/syncer_probe/*"; }

# ==== 3 arms x 4 crossover corners = 12 configs ====
# Ordering for value-under-preemption: the primary result is B-vs-C (does
# current-anchor amplify the poison?), answerable WITHOUT the barrier arm. So
# run ALL non-barrier arms C+B across every corner FIRST (poison corners before
# their mu0 baselines), then the slower/novel barrier arm A. If A (m4 +
# --barrier-sync + fixed-window, a novel combination) ever stalls, complete B/C
# for all four corners are already banked.
CELLS="h16mu09 h16mu0 h256mu05 h256mu0"
for cell in $CELLS; do
  arm_cap C "$cell"   # current-anchor (drift logged, merge unchanged)
  arm_cap B "$cell"   # version-matched, non-barrier
  sync_up
done
for cell in $CELLS; do
  arm_cap A "$cell"   # version-matched + barrier (confirmatory)
  sync_up
done

# Interpretation dump: per (arm,cell) final eval loss (.17g) + median anchor
# drift ratio from the event tape.
python3 - "$E" <<'PY'
import json,glob,os,sys,statistics
root=sys.argv[1]
print("=== EXP2.46 anchor-control losses (.17g) + tape drift ===")
for d in sorted(glob.glob(os.path.join(root,"*","report","results.jsonl"))):
    label=d.split("/")[-3]
    rows=[json.loads(l) for l in open(d) if l.strip()]
    tr=[r for r in rows if r.get("arm") not in ("base (untrained)","baseline (sync, injected)")]
    r=max(tr,key=lambda x:x.get("m",0)) if tr else max(rows,key=lambda x:x.get("m",0))
    tape=os.path.join(root,label,"work","m4","tape.jsonl")
    ratios=[]
    if os.path.exists(tape):
        for l in open(tape):
            try: rec=json.loads(l)
            except: continue
            for resp in rec.get("responders",[]):
                v=resp.get("anchor_drift_ratio")
                if isinstance(v,(int,float)) and resp.get("anchor_base_resolved"):
                    ratios.append(v)
    med=round(statistics.median(ratios),5) if ratios else None
    print(f"{label}\tloss={r['eval_loss']!r}\tdrift_ratio_med={med}\tn_drift={len(ratios)}")
PY
sync_up
echo "[exp2-46] COMPLETE"

#!/bin/bash
# S27 G-L-F L9: resume from a full-parameter round cut written with the Adam state on the
# host (--rl-optimizer-offload-fraction 1.0). Qwen3-0.6B, 1 train + 1 dedicated engine
# H100!, a cut every round. Phase a: rounds 0-1 then stop (--rl-stop-after-rounds 2);
# phase b: same store, resumes and trains rounds 2-3. Review: SESSION27/gpu-review/GL-F.md.
# usage: l9.sh RUN_ID (s27-glf-l9-*)   (phases run as $RUN_ID-pa, $RUN_ID-pb)
set -u
B=${1:?run id}; case $B in s27-glf-l9-*) ;; *) echo "abort: run id must start with s27-glf-l9-"; exit 65;; esac
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../.." && pwd)
STOREVOL=yeto-eval-store
base="launch --training-mode rl --rl-engine ports --preflight-threads off --modal-gpu-exact --modal-retries 0 \
--modal-cpu 8 --modal-memory 64 --modal-tape-volume yeto-event-tapes --controller local --rl-observe-timeline \
--model Qwen/Qwen3-0.6B --data zhuzilin/gsm8k --data-revision 0cbd9f31d91ac21a7613dcbc7fef992adac459ae \
--rl-checkpoint-store modal-volume://$STOREVOL/$B --reward-function gsm8k_reward:score --fragments 1 --pipeline 1 \
--rollout-batch-size 8 --n-samples-per-prompt 4 --rollout-max-response-len 256 --seq-len 512 --inner-lr 1e-6 \
--seed 17 --total-steps 4 --trust-remote-code --gpu modal:2xh100 --rl-placement fixed-partition --rl-rollout-gpus 1 \
--rl-single-island-no-sync --tuning full --rl-optimizer-offload-fraction 1.0 --rl-cut-every 1"
export REPO STOREVOL STORE_P=$B ISLANDS=1 HARD=${HARD:-1500} STALL_S=600 PY=/home/michael/work/gpu-head/venv/bin/python TOKIO_WORKER_THREADS=4
for ph in pa pb; do
  P=$B-$ph; R=/home/michael/work/s1-runs/s27-glf/$P; mkdir -p "$(dirname $R)"
  extra=""; [ $ph = pa ] && extra=" --rl-stop-after-rounds 2"
  ARGS="$base --cluster-prefix $P$extra" P=$P R=$R bash "$HERE/miles_local.sh"
  echo "PHASE $P rc=$(cat $R/rc 2>/dev/null)"
  [ -n "${DRY:-}" ] || [ "$(cat $R/rc 2>/dev/null)" = 0 ] || { echo "abort: phase $ph rc"; exit 1; }
done

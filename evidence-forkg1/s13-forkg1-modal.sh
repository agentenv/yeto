#!/bin/bash
# S13 critic-family fork G1 on Modal 1xH100! (exact type, container asserts H100 via --modal-gpu-exact), one island,
# --rl-single-island-no-sync. Image MILES_NEXT_IMAGE (Miles c35702e) + critic overlay (fork 6e7365b60 as patch
# yeto/rl/overlays/miles-critic-c357.patch sha256 7cbc0a42...), applied by the ports setup (REFUSING on mismatch).
# Derived from s13-g1-modal.sh. CASE selects the test:
#   gae-la  task 6.4  PPO + length_adaptive lambda (alpha 1.5), 2 rounds, Qwen3-0.6B LoRA, gsm8k
#   gae-cs  task 6.4  CompactionRL spec (cross_segment_per_sample, warm-up 0) on SYNTHETIC two-segment labels
#                     (reward yeto.rl.synthetic_segments:score = gsm8k score + tokens_after 64/0 by index parity), 2 rounds
#   vapo    task 7.3  VAPO spec (warm-up W 50 steps + 3 rounds), reward yeto.rl.gsm8k_reward:score (success field)
#   sao     task 8.4  SAO reasoning recipe as spec (sao_dis 0.3/5.0, HL-Gauss 51, 2 critic epochs), Qwen3.5-0.8B, 3 rounds
# Review: /home/michael/work/infra-drafts/CRITIC-FORK-G1-PRELAUNCH-REVIEW.md
# Cost: 1 x H100 $3.95/h + 4 cores x $0.0472 + 32 GiB x $0.0080 = $4.39/h (modal.com/pricing 2026-10-07).
# usage: CASE=<case> s13-forkg1-modal.sh <run_id>   env: PLAN_ONLY=1, REPO, SHA, HARD, THREAD_MAX
set -u
CASE=${CASE:?CASE=gae-la|gae-cs|vapo|sao}; P=${1:?run id}; HARD=${HARD:-5400}; WD=$(( HARD + 300 )); MODAL_TIMEOUT=$HARD
REPO=${REPO:-/home/michael/work/s13-forkg1}; B=/home/michael/work/s1-runs; R=$B/$P
PY=/home/michael/work/gpu-head/venv/bin/python
APP=yeto-$P; TAPEVOL=yeto-event-tapes; EXPECT_GPU=H100; PRICE=4.39
IMAGE=$(cd "$REPO" && PYTHONPATH=. /usr/bin/python3 -c "import yeto.rl as r; print(r.MILES_NEXT_IMAGE if r.MILES_NEXT_IMAGE.startswith('docker:') else 'docker:'+r.MILES_NEXT_IMAGE)" 2>/dev/null)
[ -n "$IMAGE" ] || { echo "abort: could not resolve MILES_NEXT_IMAGE from $REPO"; exit 65; }
T=$(ps -u michael -L -o pid= | wc -l); if [ "$T" -ge ${THREAD_MAX:-3000} ]; then echo "abort: $T user threads (max ${THREAD_MAX:-3000})"; exit 3; fi
HAS_TAPE=0; grep -q -- '"--modal-tape-volume"' $REPO/yeto/cli.py && HAS_TAPE=1
case "${TAPE:-auto}" in
  auto) [ $HAS_TAPE = 1 ] || { echo "abort: $REPO has no --modal-tape-volume (merge s13-modalmn 4a85c122 first, or TAPE=0 to run without the baseline tape)"; [ "${PLAN_ONLY:-0}" = 1 ] || exit 64; }; USE_TAPE=$HAS_TAPE ;;
  0) USE_TAPE=0 ;;
  *) [ $HAS_TAPE = 1 ] || { echo "abort: TAPE=1 but $REPO has no --modal-tape-volume"; exit 64; }; USE_TAPE=1 ;;
esac
TAPEX=""; [ $USE_TAPE = 1 ] && TAPEX="--modal-tape-volume $TAPEVOL"
SPECDIR=${PLAN_ONLY:+/tmp/$P-plan}; SPECDIR=${SPECDIR:-$R}; mkdir -p $SPECDIR
cp $REPO/evidence-forkg1/spec-$CASE.json $SPECDIR/spec.json || exit 65
QWEN3="--model Qwen/Qwen3-0.6B --model-revision c1899de289a04d12100db370d81485cdf75e47ca"
QWEN35="--model Qwen/Qwen3.5-0.8B --model-revision 2fc06364715b967f1860aea9cf38778875588b17"
REWARD=yeto.rl.gsm8k_reward:score; MODEL=$QWEN3; STEPS=2; ALLOW="--rl-allow-unverified-mechanism execution:critic --rl-allow-unverified-mechanism advantage_estimators:ppo"
case $CASE in
  gae-la) MECH="--rl-allow-unverified-mechanism features:gae_length_adaptive" ;;
  gae-cs) MECH="--rl-allow-unverified-mechanism features:critic_multi_update --rl-allow-unverified-mechanism features:gae_cross_segment --rl-allow-unverified-mechanism features:gae_length_adaptive"; REWARD=yeto.rl.synthetic_segments:score ;;
  vapo) STEPS=3; MECH="--rl-allow-unverified-mechanism features:gae_decoupled --rl-allow-unverified-mechanism features:gae_length_adaptive --rl-allow-unverified-mechanism features:positive_example_lm_loss" ;;
  vapo-w0) STEPS=3; MECH="--rl-allow-unverified-mechanism features:gae_decoupled --rl-allow-unverified-mechanism features:gae_length_adaptive --rl-allow-unverified-mechanism features:positive_example_lm_loss" ;;
  sao) STEPS=${SAO_STEPS:-12}; MECH="--rl-allow-unverified-mechanism features:critic_multi_update --rl-allow-unverified-mechanism features:gae_decoupled --rl-allow-unverified-mechanism features:gae_length_adaptive --rl-allow-unverified-mechanism features:sao_dis --rl-allow-unverified-mechanism features:value_hl_gauss"; MODEL=$QWEN35; LORA_OVERRIDE="--tuning lora --lora-r 16 --lora-targets attention" ;;
  *) echo "abort: unknown CASE $CASE"; exit 65 ;;
esac
ALLOW="$ALLOW $MECH ${EXTRA_ALLOW:-}"
LORA=${LORA_OVERRIDE:-"--tuning lora --lora-r 16 --lora-targets all-linear"}
CRITIC="--rl-algorithm-spec $SPECDIR/spec.json $ALLOW --rl-miles-overlay ${OVERLAY:-auto}"
MODALX="--modal-gpu-exact --modal-retries 0 --modal-timeout-s $MODAL_TIMEOUT $TAPEX"
ARGS="launch --controller local --training-mode rl --rl-engine ports --rl-single-island-no-sync --on-demand --gpu modal:1xh100 --cluster-prefix $P --no-island-relaunch $MODALX --rl-image $IMAGE $MODEL --data zhuzilin/gsm8k --data-revision 0cbd9f31d91ac21a7613dcbc7fef992adac459ae --reward-function $REWARD $LORA --fragments 1 --pipeline 1 --rollout-batch-size 4 --n-samples-per-prompt 8 --rollout-max-response-len 384 --seq-len 1024 --inner-lr 1e-5 --seed 17 --apply-chat-template-kwargs '{\"enable_thinking\": false}' --trust-remote-code --total-steps $STEPS $CRITIC"

if [ "${PLAN_ONLY:-0}" = 1 ]; then
  echo "PLAN_ONLY run=$P app=$APP tape=$([ $USE_TAPE = 1 ] && echo $TAPEVOL || echo OFF) image=$IMAGE hard=${HARD}s modal_timeout=${MODAL_TIMEOUT}s watchdog=${WD}s code=$REPO@$(git -C $REPO rev-parse --short ${SHA:-HEAD}) threads=$T"
  echo "case=$CASE est: ~0.6-1.2 h x \$$PRICE/h = ~\$4.4-5.7; worst (Modal timeout ${MODAL_TIMEOUT}s + stop) \$$(python3 -c "print(round($PRICE*$WD/3600,2))"); cap \$8"
  echo "$ARGS"
  (cd $REPO && HOME=/home/michael PYTHONPATH=$REPO:$REPO/tests:/tmp/pystub $PY - "$ARGS" <<'PY'
import shlex, sys
from yeto import launcher
from yeto.cli import parse_args
from yeto.gpu_spec import parse_gpu_spec
args = parse_args(shlex.split(sys.argv[1])[1:])
args.source_sha256 = "c" * 64; args.reward_sha256 = "d" * 64  # placeholders (the real launch hashes the snapshot)
launcher._prepare_rl_args(args)
specs = parse_gpu_spec(args.gpu)
launcher.check_cloud_prerequisites(specs, args=args)  # Modal token, image digest, shape (no resources)
print("recover_timeout:", launcher.effective_recover_timeout(args), "islands:", launcher.learner_cluster_names(args.cluster_prefix, specs))
task = launcher.make_miles_island_task(args, specs[0], 0, 1, "none")
cfg = launcher.build_modal_island_config(args, specs[0], 0, task, "none")
cfg.validate()
keys = ("app_name", "function_name", "gpu_request", "num_nodes", "gpu_exact", "retries", "timeout_s",
        "tape_volume_name", "tape_subdir", "cpu_request", "memory_request_mib", "image_ref")
print("modal cfg:", {k: getattr(cfg, k, "<absent in this code>") for k in keys})
learner = next(l for l in task.run.splitlines() if "-m yeto.rl.learner" in l)
print("learner flags:", learner.split("-m yeto.rl.learner", 1)[1].strip())
print("algorithm sha:", args.rl_expected_algorithm_sha256)
print("overlay:", getattr(args, "rl_miles_overlay", None)); import re as _re
print("setup overlay lines:", [l.strip()[:160] for l in (task.setup or "").splitlines() if _re.search(r"REFUSING|sha256sum|git apply|overlay", l)][:12])
PY
  ) || exit 66
  exit 0
fi

[ -e $R/start_utc.txt ] && { echo "abort: $R already used"; exit 67; }
mkdir -p $R/home $R/runs $R/yeto
for d in .modal.toml .sky .cache; do ln -sfn /home/michael/$d $R/home/$d; done
[ -e /home/michael/.huggingface ] && ln -sfn /home/michael/.huggingface $R/home/.huggingface
git -C $REPO archive ${SHA:-HEAD} | tar x -C $R/yeto
touch $R/yeto/yeto-rl-echo-events
git -C $REPO rev-parse ${SHA:-HEAD} > $R/yeto_sha.txt; echo "$ARGS" > $R/args.txt; echo $APP > $R/app.txt; echo $USE_TAPE > $R/tape.txt; echo $CASE > $R/case.txt
# independent watchdog: stop THIS app only, then record `modal app list`
setsid nohup bash -c "sleep $WD; cd /tmp; HOME=/home/michael $PY -m modal app stop --yes $APP > $R/watchdog.out 2>&1; HOME=/home/michael $PY -m modal app list --json > $R/watchdog-applist.json 2>&1; touch $R/WATCHDOG_FIRED" >/dev/null 2>&1 &
echo $! > $R/watchdog.pid
(
export HOME=$R/home YETO_RUNS_DIR=$R/runs PYTHONPATH=$R/yeto
eval "$(/usr/bin/python3 - <<'PY'
import base64, json, shlex
a = json.load(open("/home/michael/.docker/config.json"))["auths"]["ghcr.io"]["auth"]
u, t = base64.b64decode(a).decode().split(":", 1)
print(f"export SKYPILOT_DOCKER_USERNAME={shlex.quote(u)} SKYPILOT_DOCKER_PASSWORD={shlex.quote(t)} SKYPILOT_DOCKER_SERVER=ghcr.io")
PY
)"
cd $R/yeto; date -u +%FT%TZ > $R/start_utc.txt
eval "timeout $HARD $PY -m yeto.cli $(cat $R/args.txt)" 2>&1 | tee $R/launch.log | awk '{ print strftime("%FT%TZ", systime(), 1) " " $0; fflush() }' > $R/launch.ts.log
echo "rc=${PIPESTATUS[0]}" > $R/rc.txt; date -u +%FT%TZ > $R/end_utc.txt
)
cd /tmp; export HOME=/home/michael
# belt and braces: stop this app (no-op if the launcher already did), verify, pull the tape Volume again
timeout 300 $PY -m modal app stop --yes $APP > $R/final-stop.out 2>&1
sleep 15; timeout 120 $PY -m modal app list --json > $R/final-applist.json 2>&1
if [ $USE_TAPE = 1 ]; then mkdir -p $R/tape-direct; timeout 600 $PY -m modal volume get --force $TAPEVOL $APP $R/tape-direct > $R/tape-direct.out 2>&1; fi
kill $(cat $R/watchdog.pid) 2>/dev/null
/usr/bin/python3 $B/s13-forkg1-judge.py $R $CASE $EXPECT_GPU > $R/judgment.json 2>&1
echo "done $P $(cat $R/rc.txt) judgment=$(head -c 300 $R/judgment.json)"

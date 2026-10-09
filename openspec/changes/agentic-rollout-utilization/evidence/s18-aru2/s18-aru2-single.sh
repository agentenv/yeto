#!/bin/bash
# S18 ARU-2 (copied from s17-g1-single.sh; repo, batch/length/thinking, EXTRA). S17 G1 (N3) single-island Modal run, local controller, no syncer. Derived from
# rl-algo-mismatch-correction/evidence/2026-09-29-trigger/harness/run_one.sh (puller + watchdog + app stop),
# plus --modal-tape-volume. usage: s17-g1-single.sh <run_dir_name> <steps> [spec.json|-] [allow...]
set -u; P=$1; STEPS=$2; SPEC=${3:--}; EXTRA=${EXTRA:-}; shift 3 || shift $#
B=/home/michael/work/s1-runs; R=$B/$P; REPO=/home/michael/work/s18-aru-stage2; PY=/home/michael/work/gpu-head/venv/bin/python
MODAL="$PY -m modal"; PFX=$P; APP=yeto-$PFX; TAPEVOL=yeto-event-tapes; HARD=${HARD:-2700}
T=$(ps -u michael -L --no-headers | wc -l); [ "$T" -lt 3000 ] || { echo "abort: $T threads"; exit 3; }
pgrep -x raylet >/dev/null && { echo "abort: local Ray"; exit 3; }
mkdir -p $R/yeto; [ -e $R/start_utc.txt ] && { echo "abort: $R used"; exit 67; }
git -C $REPO archive HEAD | tar x -C $R/yeto; git -C $REPO rev-parse HEAD > $R/yeto_sha.txt
cp /home/michael/work/gpu-default-modal/yeto/gsm8k_reward.py $R/yeto/
SPECARG=""; [ "$SPEC" != "-" ] && { cp $SPEC $R/spec.json; SPECARG="--rl-algorithm-spec $R/spec.json"; }
ALLOW=(); for a in "$@"; do ALLOW+=(--rl-allow-unverified-mechanism "$a"); done
eval "$(/usr/bin/python3 - <<'PYC'
import shlex, tomllib
t = tomllib.load(open("/home/michael/.modal.toml", "rb"))
prof = next((v for v in t.values() if isinstance(v, dict) and v.get("active")), None) or next(v for v in t.values() if isinstance(v, dict))
print(f"export MODAL_TOKEN_ID={shlex.quote(prof['token_id'])} MODAL_TOKEN_SECRET={shlex.quote(prof['token_secret'])}")
PYC
)"
cleanup(){ (cd /tmp && timeout 120 $MODAL app stop -y $APP > $R/final-stop.out 2>&1); }
trap cleanup EXIT
setsid nohup bash -c "sleep $(( HARD + 300 )); cd /tmp; $MODAL app stop -y $APP; touch $R/WATCHDOG_FIRED" >/dev/null 2>&1 < /dev/null &
echo $! > $R/watchdog.pid
( while sleep 10; do c=$(cd /tmp && timeout 60 $MODAL container list --json 2>/dev/null | python3 -c "import json,sys; print(' '.join(x['container_id'] for x in json.load(sys.stdin) if x.get('app_name')=='$APP'))" 2>/dev/null)
    for x in $c; do echo $x > $R/container_id.txt
      (cd /tmp && timeout 60 $MODAL container exec $x -- sh -c "cat /root/yeto-output/rl-island-0.jsonl 2>/dev/null") > $R/.tape 2>/dev/null && [ -s $R/.tape ] && grep -q '"event"' $R/.tape && mv $R/.tape $R/tape.jsonl
      (cd /tmp && timeout 30 $MODAL container exec $x -- sh -c "nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv,noheader") >> $R/nvml.csv 2>/dev/null; done; done ) &
PULLER=$!
ARGS="launch --training-mode rl --rl-engine ports --gpu modal:1xh100 --modal-gpu-exact --modal-retries 0 --modal-tape-volume $TAPEVOL --controller local --rl-single-island-no-sync $SPECARG ${ALLOW[*]} --rl-observe-timeline --cluster-prefix $PFX --model Qwen/Qwen3-0.6B --model-revision c1899de289a04d12100db370d81485cdf75e47ca --data zhuzilin/gsm8k --data-revision 0cbd9f31d91ac21a7613dcbc7fef992adac459ae --reward-function gsm8k_reward:score --tuning lora --lora-r 16 --lora-targets all-linear --total-steps $STEPS --fragments 1 --pipeline 1 --rollout-batch-size 8 --n-samples-per-prompt 8 --rollout-max-response-len 2048 --seq-len 2560 $EXTRA --inner-lr 1e-5 --seed 17 --apply-chat-template-kwargs '{\"enable_thinking\": true}' --trust-remote-code"
echo "$ARGS" > $R/args.txt
date -u +%FT%TZ > $R/start_utc.txt
cd $R/yeto && eval "PYTHONPATH=$R/yeto YETO_RUNS_DIR=$R/runs timeout $HARD $PY -m yeto.cli $ARGS" 2>&1 | awk '{ print strftime("%FT%TZ", systime(), 1) " " $0; fflush() }' > $R/launch.ts.log
echo ${PIPESTATUS[0]} > $R/rc; date -u +%FT%TZ > $R/end_utc.txt
sed 's/^[^ ]* //' $R/launch.ts.log > $R/launch.log
kill $PULLER 2>/dev/null; kill $(cat $R/watchdog.pid) 2>/dev/null
cleanup; sleep 10
(cd /tmp && timeout 120 $MODAL app list --json) > $R/final-applist.json 2>&1
mkdir -p $R/tape-direct; (cd /tmp && timeout 600 $MODAL volume get --force $TAPEVOL $APP $R/tape-direct) > $R/tape-direct.out 2>&1
AID=$(/usr/bin/python3 -c "import json,sys;print(next((a['app_id'] for a in json.load(open(sys.argv[1])) if a.get('description')==sys.argv[2]),''))" $R/final-applist.json $APP 2>/dev/null); echo "$AID" > $R/app_id.txt
[ -n "$AID" ] && (cd /tmp && timeout 180 $MODAL app logs $AID > $R/modal-app-logs.txt 2>&1)
echo "$P rc=$(cat $R/rc)"

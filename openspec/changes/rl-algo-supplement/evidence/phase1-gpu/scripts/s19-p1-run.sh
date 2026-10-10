#!/bin/bash
# S19 rl-algo-supplement phase-1 G1 single-island Modal run (derived from s17-g1-single.sh).
# usage: CASE=<case> [STEPS=3] [OPT=2] [GRP=4] [LR=1e-5] [GPU=modal:1xa10g] [HARD=2400] s19-p1-run.sh <run-id>
# spec/allow from $SPECDIR/<case>.json|.allow; launch holds GPU-LAUNCH.lock until the Modal app exists.
set -u; P=$1; CASE=${CASE:?}; STEPS=${STEPS:-3}; OPT=${OPT:-2}; GRP=${GRP:-4}; LR=${LR:-1e-5}
GPU=${GPU:-modal:1xa10g}; HARD=${HARD:-2400}; EXTRA=${EXTRA:-}
B=/home/michael/work/s1-runs; R=$B/$P; REPO=${REPO:-/home/michael/work/algosup-run}; PY=/home/michael/work/gpu-head/venv/bin/python
SPECDIR=${SPECDIR:-$B/s19-p1-specs}; LOCK=/home/michael/work/infra-drafts/GPU-LAUNCH.lock
MODAL="$PY -m modal"; APP=yeto-$P; TAPEVOL=yeto-event-tapes
pgrep -x raylet >/dev/null && { echo "abort: local Ray"; exit 3; }
mkdir -p $R/yeto; [ -e $R/start_utc.txt ] && { echo "abort: $R used"; exit 67; }
git -C $REPO archive HEAD | tar x -C $R/yeto; git -C $REPO rev-parse HEAD > $R/yeto_sha.txt
cp /home/michael/work/gpu-default-modal/yeto/gsm8k_reward.py $R/yeto/
SPECARG=""; [ "${CASE%old}" != base ] && { cp $SPECDIR/$CASE.json $R/spec.json; SPECARG="--rl-algorithm-spec $R/spec.json"; }
ALLOW=(); while read -r a; do [ -n "$a" ] && ALLOW+=(--rl-allow-unverified-mechanism "$a"); done < $SPECDIR/$CASE.allow
eval "$(/usr/bin/python3 - <<'PYC'
import shlex, tomllib
t = tomllib.load(open("/home/michael/.modal.toml", "rb"))
prof = next((v for v in t.values() if isinstance(v, dict) and v.get("active")), None) or next(v for v in t.values() if isinstance(v, dict))
print(f"export MODAL_TOKEN_ID={shlex.quote(prof['token_id'])} MODAL_TOKEN_SECRET={shlex.quote(prof['token_secret'])}")
PYC
)"
cleanup(){ (cd /tmp && timeout 120 $MODAL app stop -y $APP > $R/final-stop.out 2>&1); }
trap cleanup EXIT
# independent watchdog (survives this shell): stop the app after HARD+300 s
systemd-run --user --unit=$P-wd --collect /bin/bash -c "sleep $(( HARD + 300 )); cd /tmp; export HOME=/home/michael; env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET $MODAL app stop -y $APP; touch $R/WATCHDOG_FIRED" > $R/watchdog.out 2>&1 \
  || { setsid nohup env -u MODAL_TOKEN_ID -u MODAL_TOKEN_SECRET HOME=/home/michael bash -c "sleep $(( HARD + 300 )); cd /tmp; $MODAL app stop -y $APP; touch $R/WATCHDOG_FIRED" >/dev/null 2>&1 < /dev/null & echo $! > $R/watchdog.pid; }
( while sleep 15; do c=$(cd /tmp && timeout 60 $MODAL container list --json 2>/dev/null | python3 -c "import json,sys; print(' '.join(x['container_id'] for x in json.load(sys.stdin) if x.get('app_name')=='$APP'))" 2>/dev/null)
    for x in $c; do echo $x >> $R/container_ids.txt
      (cd /tmp && timeout 60 $MODAL container exec $x -- sh -c "cat /root/yeto-output/rl-island-0.jsonl 2>/dev/null") > $R/.tape 2>/dev/null && [ -s $R/.tape ] && grep -q '"event"' $R/.tape && mv $R/.tape $R/tape.jsonl
      (cd /tmp && timeout 30 $MODAL container exec $x -- sh -c "nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,power.draw --format=csv,noheader") >> $R/nvml.csv 2>/dev/null; done; done ) &
PULLER=$!
trap 'kill $PULLER 2>/dev/null; cleanup' EXIT
ARGS="launch --training-mode rl --rl-engine ports --gpu $GPU --modal-gpu-exact --modal-retries 0 --no-island-relaunch --modal-timeout-s $HARD --modal-tape-volume $TAPEVOL --controller local --rl-single-island-no-sync $SPECARG ${ALLOW[*]} --rl-optimizer-steps $OPT --rl-observe-timeline --preflight-threads error --cluster-prefix $P --model Qwen/Qwen3-0.6B --model-revision c1899de289a04d12100db370d81485cdf75e47ca --data zhuzilin/gsm8k --data-revision 0cbd9f31d91ac21a7613dcbc7fef992adac459ae --reward-function gsm8k_reward:score --tuning lora --lora-r 16 --lora-targets all-linear --total-steps $STEPS --fragments 1 --pipeline 1 --rollout-batch-size $GRP --n-samples-per-prompt 8 --rollout-max-response-len 384 --seq-len 1024 --inner-lr $LR --seed 17 --apply-chat-template-kwargs '{\"enable_thinking\": false}' --trust-remote-code $EXTRA"
echo "$ARGS" > $R/args.txt
if [ -n "${PLAN_ONLY:-}" ]; then cd $R/yeto && eval "PYTHONPATH=$R/yeto YETO_PLAN_ONLY=1 $PY -m yeto.cli $ARGS --dry-run" > $R/plan.txt 2>&1; echo "plan rc=$?"; kill $PULLER 2>/dev/null; trap - EXIT; systemctl --user stop $P-wd 2>/dev/null; rm -f $R/start_utc.txt; exit 0; fi
for i in $(seq 1 180); do [ $(ps -eLo user= | grep -c '^michael') -lt 2760 ] && break; sleep 10; done
exec 9>$LOCK; flock 9
T=$(ps -eLo user= | grep -c '^michael'); echo "threads $T" > $R/threads.txt; [ "$T" -lt 2800 ] || { echo "abort: $T threads"; systemctl --user stop $P-wd 2>/dev/null; exit 3; }
date -u +%FT%TZ > $R/start_utc.txt
{ ( exec 9>&-; cd $R/yeto && eval "PYTHONPATH=$R/yeto YETO_RUNS_DIR=$R/runs timeout $HARD $PY -m yeto.cli $ARGS" 2>&1; echo $? > $R/rc ) | awk '{ print strftime("%FT%TZ", systime(), 1) " " $0; fflush() }' > $R/launch.ts.log; } 9>&- &
LP=$!
for i in $(seq 1 60); do { grep -q "modal-island 0\]" $R/launch.ts.log 2>/dev/null || [ -e $R/rc ]; } && break; sleep 5; done
exec 9>&-   # release launch lock
wait $LP; date -u +%FT%TZ > $R/end_utc.txt
sed 's/^[^ ]* //' $R/launch.ts.log > $R/launch.log
kill $PULLER 2>/dev/null; systemctl --user stop $P-wd 2>/dev/null; [ -f $R/watchdog.pid ] && kill $(cat $R/watchdog.pid) 2>/dev/null
cleanup; sleep 10
(cd /tmp && timeout 120 $MODAL app list --json) > $R/final-applist.json 2>&1
mkdir -p $R/tape-direct; (cd /tmp && timeout 600 $MODAL volume get --force $TAPEVOL $APP $R/tape-direct) > $R/tape-direct.out 2>&1
AID=$(/usr/bin/python3 -c "import json,sys;print(next((a['app_id'] for a in json.load(open(sys.argv[1])) if a.get('description')==sys.argv[2]),''))" $R/final-applist.json $APP 2>/dev/null); echo "$AID" > $R/app_id.txt
[ -n "$AID" ] && (cd /tmp && timeout 180 $MODAL app logs $AID > $R/modal-app-logs.txt 2>&1)
echo "$P rc=$(cat $R/rc)"

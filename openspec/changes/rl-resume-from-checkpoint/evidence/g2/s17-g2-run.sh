#!/bin/bash
# S17 G2 (N8) single-island resume runs on Modal H100!:1. Derived from s17-g1-single.sh.
# usage: s17-g2-run.sh <run_dir_name> <store_prefix> [extra launch args...]
# env: PREEMPT_RID=<rid> -> `modal container stop` the island once the tape shows that round
#      started (phase event with rollout_id=<rid>) and it is not trained yet (once).
set -u; P=$1; PREFIX=$2; shift 2
B=/home/michael/work/s1-runs; R=$B/$P; REPO=/home/michael/work/s17-resume-impl; PY=/home/michael/work/gpu-head/venv/bin/python
MODAL="$PY -m modal"; PFX=$P; APP=yeto-$PFX; TAPEVOL=yeto-event-tapes; CKVOL=yeto-ckpt-s17; HARD=${HARD:-1800}
T=$(ps -u michael -L --no-headers | wc -l); [ "$T" -lt 3000 ] || { echo "abort: $T threads"; exit 3; }
pgrep -x raylet >/dev/null && { echo "abort: local Ray"; exit 3; }
mkdir -p $R/yeto; [ -e $R/start_utc.txt ] && { echo "abort: $R used"; exit 67; }
git -C $REPO archive HEAD | tar x -C $R/yeto; git -C $REPO rev-parse HEAD > $R/yeto_sha.txt
cp /home/michael/work/gpu-default-modal/yeto/gsm8k_reward.py $R/yeto/
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
PRE=${PREEMPT_RID:-}
( while sleep 10; do c=$(cd /tmp && timeout 60 $MODAL container list --json 2>/dev/null | python3 -c "import json,sys; print(' '.join(x['container_id'] for x in json.load(sys.stdin) if x.get('app_name')=='$APP'))" 2>/dev/null)
    for x in $c; do grep -qx "$x" $R/container_ids.txt 2>/dev/null || { echo $x >> $R/container_ids.txt; echo "$(date -u +%FT%TZ) $x" >> $R/container_seen.txt; }
      (cd /tmp && timeout 60 $MODAL container exec $x -- sh -c "cat /root/yeto-output/rl-island-0.jsonl 2>/dev/null") > $R/.tape.$x 2>/dev/null && [ -s $R/.tape.$x ] && grep -q '"event"' $R/.tape.$x && mv $R/.tape.$x $R/tape-$x.jsonl
      (cd /tmp && timeout 30 $MODAL container exec $x -- sh -c "nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,power.draw --format=csv,noheader") >> $R/nvml-$x.csv 2>/dev/null
      done; done ) &
PULLER=$!
if [ -n "$PRE" ]; then
  # fast trigger (G2 C2: the 10 s puller + 25 s wait overshot a 24 s round): follow the
  # launcher log; once rollout PRE-1 is trained, wait PREEMPT_DELAY s and stop the container
  ( touch $R/launch.ts.log; tail -F -n +1 $R/launch.ts.log 2>/dev/null | python3 -u -c "
import json, sys
rid = int(sys.argv[1])
for line in sys.stdin:
    if 'YETO_RL_EVENT' in line and '\"rl_round_trained\"' in line:
        try: r = json.loads(line[line.index('{'):])
        except ValueError: continue
        if r.get('rollout_id') == rid - 1: print('go', flush=True); break
" $PRE | head -1 > /dev/null
    sleep ${PREEMPT_DELAY:-5}; x=$(tail -1 $R/container_ids.txt)
    echo "$(date -u +%FT%TZ) stop $x (rid $PRE mid-round: rid $((PRE-1)) trained + ${PREEMPT_DELAY:-5} s)" > $R/PREEMPTED
    (cd /tmp && timeout 120 $MODAL container stop -y $x) >> $R/PREEMPTED 2>&1 ) &
fi
ARGS="launch --training-mode rl --rl-engine ports --gpu modal:1xh100 --modal-gpu-exact --modal-retries 0 --modal-tape-volume $TAPEVOL --controller local --rl-single-island-no-sync --rl-observe-timeline --cluster-prefix $PFX --model Qwen/Qwen3-0.6B --model-revision c1899de289a04d12100db370d81485cdf75e47ca --data zhuzilin/gsm8k --data-revision 0cbd9f31d91ac21a7613dcbc7fef992adac459ae --reward-function gsm8k_reward:score --tuning lora --lora-r 16 --lora-targets all-linear --total-steps 6 --fragments 1 --pipeline 1 --rollout-batch-size 4 --n-samples-per-prompt 8 --rollout-max-response-len 384 --seq-len 1024 --inner-lr 1e-5 --seed 17 --apply-chat-template-kwargs '{\"enable_thinking\": false}' --trust-remote-code --rl-checkpoint-store modal-volume://$CKVOL/$PREFIX --rl-cut-every 2 --rl-cut-keep 2 $*"
echo "$ARGS" > $R/args.txt
date -u +%FT%TZ > $R/start_utc.txt
cd $R/yeto && eval "PYTHONPATH=$R/yeto YETO_RUNS_DIR=$R/runs timeout $HARD $PY -m yeto.cli $ARGS" 2>&1 | awk '{ print strftime("%FT%TZ", systime(), 1) " " $0; fflush() }' > $R/launch.ts.log
echo ${PIPESTATUS[0]} > $R/rc; date -u +%FT%TZ > $R/end_utc.txt
sed 's/^[^ ]* //' $R/launch.ts.log > $R/launch.log
kill $PULLER 2>/dev/null; kill $(cat $R/watchdog.pid) 2>/dev/null
cleanup; sleep 10
(cd /tmp && timeout 120 $MODAL app list --json) > $R/final-applist.json 2>&1
mkdir -p $R/tape-direct; (cd /tmp && timeout 600 $MODAL volume get --force $TAPEVOL $APP $R/tape-direct) > $R/tape-direct.out 2>&1
mkdir -p $R/store; (cd /tmp && timeout 900 $MODAL volume get --force $CKVOL $PREFIX $R/store) > $R/store-get.out 2>&1
AID=$(/usr/bin/python3 -c "import json,sys;print(next((a['app_id'] for a in json.load(open(sys.argv[1])) if a.get('description')==sys.argv[2]),''))" $R/final-applist.json $APP 2>/dev/null); echo "$AID" > $R/app_id.txt
[ -n "$AID" ] && (cd /tmp && timeout 180 $MODAL app logs $AID > $R/modal-app-logs.txt 2>&1)
echo "$P rc=$(cat $R/rc)"

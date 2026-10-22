#!/bin/bash
# S26 E6 copy of tools/eval/miles_local.sh (S26 E0.1, s26-e01-effeval 5a01732c): one yeto
# launcher run on Modal (`ARGS`), optionally a second sampler-only launch (`SARGS`).
# Derived from s1-runs/s25-c1/run.sh (S25 C1) and s1-runs/s24-l5.4/run.sh (S24 L5.4).
#
# env: P (run id; app yeto-$P, store prefix $P), R (raw dir), REPO (yeto checkout),
#      ARGS (launcher args; the store is modal-volume://$STOREVOL/$P), SARGS (optional),
#      ISLANDS (containers before the GPU lock is released), HARD (s), STOREVOL, TAPEVOL,
#      RECLAIM_LABEL (e.g. "l0") + RECLAIM_ROUND (rollout id written before the stop),
#      PY (python with modal + the launcher deps), DRY (non-empty: --dry-run only)
set -u
: "${P:?}" "${R:?}" "${REPO:?}" "${ARGS:?}" "${STOREVOL:?}" "${PY:?}"
HARD=${HARD:-3000}; ISLANDS=${ISLANDS:-1}; TAPEVOL=${TAPEVOL:-yeto-event-tapes}
MODAL="$PY -m modal"; APP=yeto-$P; SAPP=yeto-$P-s; LOCK=/home/michael/work/infra-drafts/GPU-LAUNCH.lock
CLONE=$R/yeto
T=$(ps -u michael -L --no-headers | wc -l); [ -n "${DRY:-}" ] || [ "$T" -lt 3000 ] || { echo "abort: $T threads"; exit 3; }
pgrep -x raylet >/dev/null && { echo "abort: local Ray"; exit 3; }
[ -e $R/start_utc.txt ] && { echo "abort: $R used"; exit 67; }
mkdir -p $R/tapes $R/store
# a real clone: the code pack needs a git checkout
git clone -q --no-local $REPO $CLONE && git -C $REPO rev-parse HEAD > $R/yeto_sha.txt
cp $REPO/yeto/rl/gsm8k_reward.py $CLONE/gsm8k_reward.py
(cd $CLONE && git add -f gsm8k_reward.py && git -c user.email=eval@yeto -c user.name=t-eval commit -qm 'run inputs')
eval "$(/usr/bin/python3 - <<'PYC'
import shlex, tomllib
t = tomllib.load(open("/home/michael/.modal.toml", "rb"))
prof = next((v for v in t.values() if isinstance(v, dict) and v.get("active")), None) or next(v for v in t.values() if isinstance(v, dict))
print(f"export MODAL_TOKEN_ID={shlex.quote(prof['token_id'])} MODAL_TOKEN_SECRET={shlex.quote(prof['token_secret'])}")
PYC
)"
echo "$ARGS" > $R/args.txt; echo "${SARGS:-}" > $R/sargs.txt
if [ -n "${DRY:-}" ]; then
  (cd $CLONE && eval "PYTHONPATH=$CLONE YETO_RUNS_DIR=$R/runs $PY -m yeto.cli $ARGS --dry-run") > $R/plan.json 2> $R/plan.err
  echo "dry-run rc=$?"; tail -3 $R/plan.err
  if [ -n "${SARGS:-}" ]; then
    (cd $CLONE && eval "PYTHONPATH=$CLONE YETO_RUNS_DIR=$R/sruns $PY -m yeto.cli $SARGS --dry-run") > $R/splan.json 2> $R/splan.err
    echo "sampler dry-run rc=$?"; tail -3 $R/splan.err
  fi
  exit 0
fi
stop_apps(){ for a in $APP $SAPP; do case $a in yeto-$P|yeto-$P-s) (cd /tmp && timeout 120 $MODAL app stop -y $a);; esac; done > $R/final-stop.out 2>&1; }
trap stop_apps EXIT
setsid nohup bash -c "sleep $(( HARD + 300 )); cd /tmp; $MODAL app stop -y $APP; $MODAL app stop -y $SAPP; touch $R/WATCHDOG_FIRED" >/dev/null 2>&1 < /dev/null &
echo $! > $R/watchdog.pid
# container tape + NVML puller (both apps)
( while sleep 20; do c=$(cd /tmp && timeout 60 $MODAL container list --json 2>/dev/null | python3 -c "import json,sys; print(' '.join(x['container_id'] for x in json.load(sys.stdin) if x.get('app_name') in ('$APP','$SAPP')))" 2>/dev/null)
    echo "$(date -u +%FT%TZ) $c" >> $R/containers.log
    for x in $c; do
      (cd /tmp && timeout 60 $MODAL container exec $x -- sh -c "cat /root/yeto-output/rl-island-*.jsonl 2>/dev/null") > $R/tapes/.$x 2>/dev/null && [ -s $R/tapes/.$x ] && grep -q '"event"' $R/tapes/.$x && mv $R/tapes/.$x $R/tapes/$x.jsonl
      (cd /tmp && timeout 30 $MODAL container exec $x -- sh -c "nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,power.draw --format=csv,noheader") 2>/dev/null | sed "s/^/$x,/" >> $R/nvml.csv; done
  done ) 9>&- &
PULLER=$!
exec 9>$LOCK; flock 9
date -u +%FT%TZ > $R/start_utc.txt
( cd $CLONE && eval "PYTHONPATH=$CLONE YETO_RUNS_DIR=$R/runs timeout $HARD $PY -m yeto.cli $ARGS" 2>&1 | awk '{ print strftime("%FT%TZ", systime(), 1) " " $0; fflush() }' > $R/launch.ts.log; echo ${PIPESTATUS[0]} > $R/rc ) 9>&- &
SUB=$!
SSUB=""
if [ -n "${SARGS:-}" ]; then
  ( cd $CLONE && eval "PYTHONPATH=$CLONE YETO_RUNS_DIR=$R/sruns timeout $HARD $PY -m yeto.cli $SARGS" 2>&1 | awk '{ print strftime("%FT%TZ", systime(), 1) " " $0; fflush() }' > $R/slaunch.ts.log; echo ${PIPESTATUS[0]} > $R/src ) 9>&- &
  SSUB=$!
fi
# stall stop: no new trained round (trainer) or sampler round for STALL_S s -> stop both apps
( last=$(date +%s); prev=-1
  while kill -0 $SUB 2>/dev/null; do sleep 30
    n=$(cat $R/launch.ts.log $R/slaunch.ts.log 2>/dev/null | grep -aEc '"event":"(rl_round_trained|rl_sampler_round)"')
    [ "$n" != "$prev" ] && { prev=$n; last=$(date +%s); }
    if [ $(( $(date +%s) - last )) -ge ${STALL_S:-900} ]; then
      echo "$(date -u +%FT%TZ) no new step for ${STALL_S:-900} s (events=$n): stopping" >> $R/STALL_STOP
      (cd /tmp && timeout 120 $MODAL app stop -y $APP; timeout 120 $MODAL app stop -y $SAPP) >> $R/STALL_STOP 2>&1; exit 0
    fi
  done ) 9>&- &
# release the launch lock once every island has a container (or the launch ended)
for i in $(seq 1 120); do
  n=$(cat $R/launch.ts.log $R/slaunch.ts.log 2>/dev/null | grep -ac 'rank 0 container')
  [ "$n" -ge "$ISLANDS" ] && break
  kill -0 $SUB 2>/dev/null || break; sleep 10
done
exec 9>&-
# one reclaim: island $RECLAIM_LABEL wrote rollout $RECLAIM_ROUND -> stop its container
if [ -n "${RECLAIM_LABEL:-}" ]; then
( cd /tmp
  for i in $(seq 1 600); do
    kill -0 $SUB 2>/dev/null || { echo "$(date -u +%FT%TZ) launch ended before the reclaim" >> $R/reclaim.log; exit 0; }
    if grep -a -- "-$RECLAIM_LABEL-modal" $R/launch.ts.log | grep -a '"event":"rl_sampler_round"' | grep -aq "\"rollout_id\":$RECLAIM_ROUND[,}]"; then
      CID=$(grep -a -oP "\[$P-$RECLAIM_LABEL-modal\] \[modal-island \d+\] rank 0 container \K\S+" $R/launch.ts.log | tail -1)
      echo "$(date -u +%FT%TZ) trigger container=$CID" >> $R/reclaim.log
      [ -n "$CID" ] && timeout 120 $MODAL container stop --yes $CID >> $R/reclaim.log 2>&1; echo "stop rc=$? $(date -u +%FT%TZ) $(date +%s.%N)" >> $R/reclaim.log
      exit 0
    fi
    sleep 5
  done ) 9>&- &
fi
wait $SUB
# a sampler-only launch ends at its backlog cap once the trainer is gone: stop it
if [ -n "$SSUB" ]; then sleep 30; (cd /tmp && timeout 120 $MODAL app stop -y $SAPP) >> $R/final-stop.out 2>&1; wait $SSUB; fi
date -u +%FT%TZ > $R/end_utc.txt
sed 's/^[^ ]* //' $R/launch.ts.log > $R/launch.log
kill $PULLER 2>/dev/null; kill $(cat $R/watchdog.pid) 2>/dev/null
stop_apps; trap - EXIT; sleep 10
(cd /tmp && timeout 120 $MODAL app list --json) > $R/final-applist.json 2>&1
mkdir -p $R/tape-direct
for a in $APP $SAPP; do
  (cd /tmp && timeout 600 $MODAL volume get --force $TAPEVOL $a $R/tape-direct) >> $R/tape-direct.out 2>&1
  AID=$(/usr/bin/python3 -c "import json,sys;print(next((a['app_id'] for a in json.load(open(sys.argv[1])) if a.get('description')==sys.argv[2]),''))" $R/final-applist.json $a 2>/dev/null)
  [ -n "$AID" ] && (cd /tmp && timeout 180 $MODAL app logs --timestamps $AID > $R/modal-app-logs-$a.txt 2>&1)
done
(cd /tmp && for d in sample-buffer policy-releases round-cuts trainer-control; do timeout 900 $MODAL volume get --force $STOREVOL $P/$d $R/store; done
 timeout 300 $MODAL volume ls --json $STOREVOL $P) > $R/store.out 2>&1
echo "$P rc=$(cat $R/rc) src=$(cat $R/src 2>/dev/null)"

#!/bin/bash
# S26 E0.3: one multi-island run with the head controller (Nebius CPU VM) and Modal islands.
# Copied from S26 E3 (SESSION26/logs/E3/run.sh); the launcher args come from run.sh (env ARGS).
# Review: infra-drafts/SESSION26/gpu-review/E0.3.md.
# env: P (run id, s26-e03-*), ARGS (launcher args), REPO, ISLANDS, HARD, PLAN_ONLY=1
set -u
P=${P:?run id}; case $P in s26-e03-*) ;; *) echo "abort: run id must start with s26-e03-"; exit 65;; esac
ARGS=${ARGS:?ARGS}; HARD=${HARD:-2400}; WD=$(( HARD + 300 )); CASE=multi
REPO=${REPO:?REPO}; SHA=${SHA:-HEAD}; B=/home/michael/work/s1-runs/s26-e03; R=$B/$P
SKY=/home/michael/work/gpu-head/venv/bin/sky; PY=/home/michael/work/gpu-head/venv/bin/python
APP=yeto-$P; TAPEVOL=yeto-event-tapes
export CARGO_HOME=/home/michael/work/gpu-default-modal/cargo RUSTUP_HOME=/home/michael/work/gpu-default-modal/rustup
export PATH=$CARGO_HOME/bin:$PATH
for i in $(seq 1 180); do T=$(ps -u michael -L --no-headers | wc -l); [ "$T" -lt 2950 ] && break; sleep 5; done
[ "${PLAN_ONLY:-0}" = 1 ] || [ "$T" -lt 3000 ] || { echo "abort: $T user threads (>= 3000)"; exit 3; }
pgrep -x raylet >/dev/null && { echo "abort: local Ray running"; exit 3; }
HEAD_CL=$(cd $REPO && PYTHONPATH=$REPO /usr/bin/python3 -c "from yeto.launch.sky_ops import sky_cluster_name; print(sky_cluster_name('$P-head'))")
if [ "${PLAN_ONLY:-0}" = 1 ]; then R=/tmp/$P-plan; rm -rf $R; fi
[ -e $R/start_utc.txt ] && { echo "abort: $R already used"; exit 67; }
mkdir -p $R/home $R/runs $R/yeto $R/head
for d in .sky .ssh .nebius .cache .huggingface .config; do [ -e /home/michael/$d ] && ln -sfn /home/michael/$d $R/home/$d; done
git -C $REPO archive $SHA | tar x -C $R/yeto
touch $R/yeto/yeto-rl-echo-events

git -C $REPO rev-parse ${SHA:-HEAD} > $R/yeto_sha.txt; echo "$ARGS" > $R/args.txt; echo "$APP $HEAD_CL" > $R/resources.txt
# secrets: process env only (never echoed, never written)
[ -n "${MODAL_TOKEN_ID:-}" ] || eval "$(/usr/bin/python3 - <<'PYC'
import json, shlex, tomllib
t = tomllib.load(open("/home/michael/.modal.toml", "rb"))
prof = next((v for v in t.values() if isinstance(v, dict) and v.get("active")), None) or next(v for v in t.values() if isinstance(v, dict))
print(f"export MODAL_TOKEN_ID={shlex.quote(prof['token_id'])} MODAL_TOKEN_SECRET={shlex.quote(prof['token_secret'])}")
PYC
)"
# island HMAC key (0.20): process env only -> shipped as sky/Modal secret; never written or echoed
export YETO_ISLAND_HMAC_KEY=${YETO_ISLAND_HMAC_KEY:-$(openssl rand -hex 32)}
[ -n "${MODAL_TOKEN_ID:-}" ] || { echo "abort: no modal token"; exit 65; }
export YETO_RUNS_DIR=$R/runs PYTHONPATH=$R/yeto
if [ "${PLAN_ONLY:-0}" = 1 ]; then
  echo "PLAN_ONLY run=$P case=$CASE app=$APP head=$HEAD_CL steps=$STEPS hard=${HARD}s wd=${WD}s code=$REPO@$(cat $R/yeto_sha.txt|cut -c1-8) threads=$T"
  (cd $R/yeto && eval "$PY -m yeto.cli $(cat $R/args.txt) --dry-run") > $R/plan.json 2> $R/plan.err; echo "dry-run rc=$?"; tail -3 $R/plan.err
  /usr/bin/python3 -c "import json,sys;p=json.load(open(sys.argv[1]));[print(i['learner_id'],i.get('island_overrides'),i['memory_estimate'].get('peak_gib'),i['memory_estimate'].get('fits'),[t for t in i['learner_command'].split(' --') if t.startswith(('rl-lr','rl-max-policy','rl-island-sched','rl-algorithm','rl-stall','modal-launcher','no-island'))]) for i in p['island_requests']]" $R/plan.json
  exit 0
fi
grep -q "E0.3 偏差叠加消融" /home/michael/work/infra-drafts/gpu-spend.md || { echo "abort: E0.3 not pre-registered in gpu-spend.md"; exit 68; }
date -u +%FT%TZ > $R/start_utc.txt
# teardown of THIS run only (Modal app + head cluster; never `sky down -a`)
cat > $R/teardown.sh <<EOF
cd /tmp; export HOME=/home/michael
S='ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20'
JOB=\$(grep -oP 'head_job=\K\d+' $R/head_job.txt 2>/dev/null)
# 1) pull head evidence BEFORE anything is stopped (launch.log, syncer log, output, queue)
if timeout 60 $SKY status $HEAD_CL 2>/dev/null | grep -q ' UP '; then
  timeout 600 $SKY logs $HEAD_CL \$JOB --no-follow > $R/\${1:-final}-launch.log 2>&1
  timeout 300 $SKY queue $HEAD_CL > $R/head/queue.txt 2>&1
  timeout 600 \$S $HEAD_CL 'cd ~ && tar czf - yeto-output yeto-syncer.log sky_logs 2>/dev/null' > $R/head/head-\${1:-final}.tgz 2>/dev/null
  (cd $R/head && tar xzf head-\${1:-final}.tgz 2>/dev/null)
  timeout 60 \$S $HEAD_CL 'hostname; nproc; free -g; ss -ltnp 2>/dev/null|grep -E ":29400"' > $R/head/host.txt 2>&1
  [ -s $R/launch.log ] || cp $R/\${1:-final}-launch.log $R/launch.log
fi
echo "pulled \$(date -u +%FT%TZ) syncer_log_bytes=\$(stat -c %s $R/head/yeto-syncer.log 2>/dev/null || echo 0)" > $R/\${1:-final}-pull.out
# 2) stop Modal app, 3) down THIS head only (never -a)
case $APP in yeto-s26-e03-*) timeout 300 $PY -m modal app stop --yes $APP;; *) echo REFUSE-STOP $APP;; esac > $R/\${1:-final}-stop.out 2>&1
timeout 900 $SKY down -y $HEAD_CL > $R/\${1:-final}-down.out 2>&1; echo "down rc=\$?" >> $R/\${1:-final}-down.out
sleep 15; timeout 120 $PY -m modal app list --json > $R/\${1:-final}-applist.json 2>&1
timeout 120 $SKY status $HEAD_CL > $R/\${1:-final}-skystatus.txt 2>&1
EOF
if [ "${E3_REMOTE:-0}" = 1 ]; then setsid nohup /bin/bash -c "sleep $WD; [ -e $R/rc.txt ] || { bash $R/teardown.sh watchdog; touch $R/WATCHDOG_FIRED; }" > $R/watchdog.out 2>&1 < /dev/null & else systemd-run --user --unit=$P-wd --on-active=${WD}s /bin/bash -c "bash $R/teardown.sh watchdog; touch $R/WATCHDOG_FIRED" > $R/watchdog.out 2>&1; fi
echo "$P-wd.timer" > $R/watchdog.pid
echo "ps_threads_at_start=$T RELAXED preflight-thread-start=2950 (main agent 10-20, GOAL §6); hard 3000" > $R/threads-ps.txt; echo "RELAXED: --preflight-thread-start 2950 (hard limit 3000 unchanged)"
cd $R/yeto
exec 9>${LAUNCH_LOCK:-/home/michael/work/infra-drafts/GPU-LAUNCH.lock}; flock 9
# other sessions' thread count swings by a few hundred: wait (bounded, 15 min) instead of aborting
for i in $(seq 1 180); do T2=$(ps -u michael -L --no-headers | wc -l); [ "$T2" -lt 2950 ] && break; sleep 5; done
echo "threads_at_lock=$T2" >> $R/threads-ps.txt
[ "$T2" -lt 3000 ] || { echo "abort: $T2 threads"; exec 9>&-; systemctl --user stop $P-wd.timer; exit 3; }
( eval "timeout $HARD $PY -m yeto.cli $(cat $R/args.txt)" 2>&1 | tee $R/launch.stream.log | awk '{ print strftime("%FT%TZ", systime(), 1) " " $0; fflush() }' > $R/launch.ts.log; echo "rc=${PIPESTATUS[0]}" > $R/submit_rc.txt ) 9>&- &
SUB=$!
# wait for the controller job id
JOB=""
for i in $(seq 1 360); do
  JOB=$(grep -oP "submitted: job \K\d+" $R/launch.stream.log 2>/dev/null | head -1); [ -n "$JOB" ] && break
  kill -0 $SUB 2>/dev/null || break; sleep 10
done
echo "head_job=$JOB" > $R/head_job.txt
for i in $(seq 1 90); do [ "$(grep -c 'modal-island [01]\] requested' $R/launch.stream.log 2>/dev/null)" -ge $( [ $CASE = fpprobe ] && echo 1 || echo 2) ] && break; kill -0 $SUB 2>/dev/null || break; sleep 10; done
exec 9>&-   # release launch lock
# head host memory sampler (reader only)
( cd /tmp; export HOME=/home/michael; until timeout 120 $SKY status $HEAD_CL 2>/dev/null | grep -q " UP "; do sleep 20; done; for i in $(seq 1 $(( HARD / 15 ))); do echo "$(date -u +%FT%TZ) $(timeout 20 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 $HEAD_CL "free -m | awk '/Mem:/{print \$3, \$2}'; ps -C yeto-syncer -o rss= | head -1" 2>/dev/null | tr '\n' ' ')"; sleep 15; done > $R/head-mem.txt ) 9>&- &
HMON=$!
# per-connection send queue on the syncer port (C32 stall evidence; reader only)
[ "${SSQ:-0}" = 1 ] && ( cd /tmp; export HOME=/home/michael; until timeout 120 $SKY status $HEAD_CL 2>/dev/null | grep -q " UP "; do sleep 20; done; for i in $(seq 1 $(( HARD / 20 ))); do echo "== $(date -u +%FT%TZ)"; timeout 20 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 $HEAD_CL "ss -tn state established '( sport = :29400 )'" 2>/dev/null | awk 'NR>1{print $2, $4}'; sleep 20; done > $R/head-sendq.txt ) 9>&- &
# cold-start timing: live Modal log stream with server timestamps (reader only; stopping it never affects the app)
( cd /tmp; HOME=/home/michael timeout $HARD $PY -m modal app logs --timestamps $APP > $R/modal-live-logs.txt 2>&1 ) 9>&- &
wait $SUB
# the stream may drop early: wait (bounded) for the head job to reach a terminal state
if [ -n "$JOB" ]; then
  END=$(( $(date +%s) + HARD ))
  while [ $(date +%s) -lt $END ]; do
    st=$(cd /tmp && timeout 120 $SKY queue $HEAD_CL 2>/dev/null | awk -v j=$JOB '$1==j{print}')
    echo "$st" | grep -qE "RUNNING|PENDING|SETTING_UP|INIT" || break; sleep 30
  done
fi
date -u +%FT%TZ > $R/end_utc.txt
export HOME=/home/michael
bash $R/teardown.sh final
[ -s $R/launch.log ] || cp $R/launch.stream.log $R/launch.log
# rc: the head job's terminal status (SUCCEEDED -> 0)
grep -E "^ *$JOB " $R/head/queue.txt 2>/dev/null | grep -q SUCCEEDED && echo "rc=0" > $R/rc.txt || echo "rc=1 ($(grep -E "^ *$JOB " $R/head/queue.txt 2>/dev/null | tr -s ' ' | cut -c1-120))" > $R/rc.txt
AID=$(/usr/bin/python3 -c "import json,sys;print(next((a['app_id'] for a in json.load(open(sys.argv[1])) if a.get('description')==sys.argv[2]),''))" $R/final-applist.json $APP 2>/dev/null); [ -n "$AID" ] && (cd /tmp && timeout 180 $PY -m modal app logs --timestamps --show-function-id $AID > $R/modal-app-logs.txt 2>&1)
mkdir -p $R/tape-direct; timeout 600 $PY -m modal volume get --force $TAPEVOL $APP $R/tape-direct > $R/tape-direct.out 2>&1
systemctl --user stop $P-wd.timer 2>/dev/null
echo "judge: see SESSION26/gpu-review/E0.3.md" > $R/judge.out
kill $HMON 2>/dev/null
echo "done $P $(cat $R/rc.txt) judge=$(tail -c 400 $R/judge.out)"

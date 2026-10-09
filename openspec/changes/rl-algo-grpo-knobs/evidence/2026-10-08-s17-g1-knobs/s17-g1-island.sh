#!/bin/bash
# S17 G1 (N3): derived from s15-island1b-pause-remote.sh. Code = s17-g1 worktree (agentenv/main 80e944b6 + nothing).
#   PAUSE=0 drops --rl-elastic-debug-pause; SPEC=<file> copies it to yeto/g1-spec.json and adds --rl-algorithm-spec g1-spec.json;
#   SCHED=legacy for strict-avg (grpo-knobs 8.4). Elastic runs also get transfer records (0.28) on the head tape.
# S16 1b-pause (user-approved): derived from s15-island1b-rejoin-remote.sh. No kill, no SIGSTOP stopper (STOP=0): island 1 uses
#   the learner test switch --rl-elastic-debug-pause 1:AFTER_V:S (link silent S = LEASE+30 s after v1; process keeps
#   running). Judge: s15-island1b-pause-judge.py.
#   delay by default; freeze ONLY island 1's learner main process for LEASE+30 s right after v1 (stopper runs the
#   STOP/sleep/CONT inside the container). Judge: s15-island1b-rejoin-judge.py.
# S15 rl-inter-island-scheduling stage 1b: ELASTIC-mode (same shape as 1a, adds --rl-island-scheduling $SCHED).
#   NOT TO BE LAUNCHED until the Rust syncer elastic mode passes cargo test and is merged into s15-interisland
#   AND the user approves. Aborts (exit 64) while the CLI lacks --rl-island-scheduling.
# (1a header follows) stage 1a: LEGACY-mode two-island baseline (reference for the later elastic run 1b).
#   Derived from s13-g3-remote.sh (+ s14-dlr-remote.sh fixes): ONE Nebius eu-north1 CPU VM (no GPU) hosts the fleet
#   controller + actor syncer (:29400); two Modal islands (1x H100! each) dial its public IP. Shape = s13-g3 minus
#   critic (no spec, no critic syncer: only LoRA deltas cross the WAN), no --keep, --rl-stall-timeout 2400,
#   --total-steps $STEPS. Kill/relaunch: a killer job on the VM cancels island 1's Modal call once island 1 applied
#   v$KILL_AT_V; --modal-launcher-relaunch (+ --recover-timeout 1800) lets the launcher relaunch it. We record what
#   the legacy path does with that membership change (exit code, syncer stall, resume).
#   Code = worktree s15-interisland via git archive; island scheduling mode defaults to legacy (no flag).
#   Public image (no ghcr creds). This machine only submits; no controller / syncer / Ray locally.
# usage: s15-island1a-remote.sh [run_id]   env: PLAN_ONLY=1, REPO, SHA, KILL=0, KILL_DELAY, KILL_AT_V, THREAD_MAX, HARD, STEPS
set -u
SCHED=${SCHED:-elastic}; KILL_ISLAND=${KILL_ISLAND:-0}; DELAY=${DELAY:-}; LEASE=${LEASE:-90}; STOP_S=${STOP_S:-$(( LEASE + 30 ))}; P=${1:-s15-island1b-20261008a}; HARD=${HARD:-3600}; STEPS=${STEPS:-6}; KILL_AT_V=${KILL_AT_V:-2}; WD=$(( HARD + 600 )); MODAL_TIMEOUT=$HARD; KILL_DELAY=${KILL_DELAY:-20}
REPO=${REPO:-/home/michael/work/s17-g1}; B=/home/michael/work/s1-runs; R=$B/$P
SKY=/home/michael/work/gpu-head/venv/bin/sky; PY=/home/michael/work/gpu-head/venv/bin/python
APP=yeto-$P; TAPEVOL=yeto-event-tapes; EXPECT_GPU=H100; HEAD_REGION=nebius/eu-north1
export CARGO_HOME=/home/michael/work/gpu-default-modal/cargo RUSTUP_HOME=/home/michael/work/gpu-default-modal/rustup
export PATH=$CARGO_HOME/bin:$PATH
IMAGE=$(cd "$REPO" && PYTHONPATH=. /usr/bin/python3 -c "import yeto.rl as r; print(r.MILES_NEXT_IMAGE if r.MILES_NEXT_IMAGE.startswith('docker:') else 'docker:'+r.MILES_NEXT_IMAGE)" 2>/dev/null)
[ -n "$IMAGE" ] || { echo "abort: no MILES_NEXT_IMAGE"; exit 65; }
T=$(ps -u michael -L -o pid= | wc -l); [ "$T" -lt ${THREAD_MAX:-3000} ] || { echo "abort: $T user threads"; exit 3; }
pgrep -x raylet >/dev/null && { echo "abort: local Ray running"; exit 3; }
grep -q -- "--rl-elastic-debug-pause" $REPO/yeto/cli.py || { echo "abort: $REPO cli lacks --rl-elastic-debug-pause"; exit 64; }
grep -q 'secrets=secrets or None' $REPO/yeto/cli.py || { echo "abort: $REPO lacks head secrets (s13-g3remote)"; exit 64; }
HEAD_CL=$(cd $REPO && PYTHONPATH=$REPO /usr/bin/python3 -c "from yeto.launcher import sky_cluster_name; print(sky_cluster_name('$P-head'))")
L1=$(cd $REPO && PYTHONPATH=$REPO /usr/bin/python3 -c "
from yeto.launcher import learner_cluster_names; from yeto.gpu_spec import parse_gpu_spec
print(learner_cluster_names('$P', parse_gpu_spec('modal:1xh100,modal:1xh100'))[$KILL_ISLAND])")
if [ "${PLAN_ONLY:-0}" = 1 ]; then R=/tmp/$P-plan; rm -rf $R; fi
[ -e $R/start_utc.txt ] && { echo "abort: $R already used"; exit 67; }
mkdir -p $R/home $R/runs $R/yeto $R/head
# client HOME: everything sky/yeto need EXCEPT ~/.modal.toml (Modal goes by env -> sky secret)
for d in .sky .ssh .nebius .cache .huggingface .config; do [ -e /home/michael/$d ] && ln -sfn /home/michael/$d $R/home/$d; done
git -C $REPO archive ${SHA:-HEAD} | tar x -C $R/yeto
cp /home/michael/work/gpu-default-modal/yeto/gsm8k_reward.py $R/yeto/; touch $R/yeto/yeto-rl-echo-events; [ -n "${SPEC:-}" ] && cp $SPEC $R/yeto/g1-spec.json
MODEL="--model Qwen/Qwen3-0.6B --model-revision c1899de289a04d12100db370d81485cdf75e47ca"; LORA="--tuning lora --lora-r 16 --lora-targets all-linear"
MODALX="--modal-gpu-exact --modal-retries 0 --modal-launcher-relaunch --recover-timeout 1800 --modal-timeout-s $MODAL_TIMEOUT --modal-tape-volume $TAPEVOL"
ARGS="launch --controller head --syncer-region $HEAD_REGION --training-mode rl --rl-engine ports --rl-island-scheduling $SCHED $( [ "$SCHED" = elastic ] && echo "--rl-soft-deadline-s ${SOFT:-180} ")${DELAY:+--rl-elastic-debug-delay-s $DELAY }$( [ "${PAUSE:-1}" = 1 ] && echo "--rl-elastic-debug-pause ${PAUSE_ISLAND:-1}:${PAUSE_AT_V:-1}:$STOP_S ")${SPEC:+--rl-algorithm-spec g1-spec.json }$( [ "$SCHED" = elastic ] && echo "--rl-island-lease-s $LEASE ")--rl-stall-timeout 2400 --rl-observe-timeline --on-demand --gpu modal:1xh100,modal:1xh100 --cluster-prefix $P $MODALX --rl-image $IMAGE $MODEL --data zhuzilin/gsm8k --data-revision 0cbd9f31d91ac21a7613dcbc7fef992adac459ae --reward-function gsm8k_reward:score $LORA --fragments 1 --pipeline 1 --rollout-batch-size 4 --n-samples-per-prompt 8 --rollout-max-response-len 384 --seq-len 1024 --inner-lr 1e-5 --seed 17 --apply-chat-template-kwargs '{\"enable_thinking\": false}' --trust-remote-code --total-steps $STEPS"
git -C $REPO rev-parse ${SHA:-HEAD} > $R/yeto_sha.txt; echo "$ARGS" > $R/args.txt; echo "$APP $HEAD_CL" > $R/resources.txt
# secrets: process env only (never echoed, never written)
eval "$(/usr/bin/python3 - <<'PYC'
import json, shlex, tomllib
t = tomllib.load(open("/home/michael/.modal.toml", "rb"))
prof = next((v for v in t.values() if isinstance(v, dict) and v.get("active")), None) or next(v for v in t.values() if isinstance(v, dict))
print(f"export MODAL_TOKEN_ID={shlex.quote(prof['token_id'])} MODAL_TOKEN_SECRET={shlex.quote(prof['token_secret'])}")
PYC
)"
# island HMAC key (0.20): process env only -> shipped as sky/Modal secret; never written or echoed
export YETO_ISLAND_HMAC_KEY=${YETO_ISLAND_HMAC_KEY:-$(openssl rand -hex 32)}
[ -n "${MODAL_TOKEN_ID:-}" ] || { echo "abort: no modal token"; exit 65; }
export HOME=$R/home YETO_RUNS_DIR=$R/runs PYTHONPATH=$R/yeto
if [ "${PLAN_ONLY:-0}" = 1 ]; then
  echo "PLAN_ONLY run=$P app=$APP head=$HEAD_CL ($HEAD_REGION) l1=$L1 steps=$STEPS kill_at_v=$KILL_AT_V hard=${HARD}s wd=${WD}s code=$REPO@$(cat $R/yeto_sha.txt|cut -c1-8) threads=$T"
  (cd $R/yeto && $PY - "$ARGS" <<'PY'
import shlex, sys, yaml
import sky
from yeto import cli, launcher
args = cli.parse_args(shlex.split(sys.argv[1])[1:])
launcher.prepare_launch_args(args)
print("controller:", args.controller, "syncer ports:", launcher.syncer_ports(args), "needs critic:", launcher.rl_needs_critic(args),
      "sched:", getattr(args, "rl_island_scheduling", None), "delay:", getattr(args, "rl_elastic_debug_delay_s", None), "pause:", getattr(args, "rl_elastic_debug_pause", None), "lease:", getattr(args, "rl_island_lease_s", None), "stall:", args.rl_stall_timeout, "keep:", getattr(args, "keep", None), "steps:", args.total_steps,
      "relaunch:", getattr(args, "modal_launcher_relaunch", None), "no_island_relaunch:", getattr(args, "no_island_relaunch", None))
print("syncer cmd (head LocalSyncer):", launcher.syncer_command(args, 2))
mounts, envs = launcher.head_cloud_credentials(launcher.fleet_clouds(args))
print("head cred file mounts:", sorted(mounts), "| cred env NAMES (sent as secrets):", sorted(envs))
assert "~/.modal.toml" not in mounts and {"MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"} <= set(envs)
import os
from yeto.gpu_spec import parse_gpu_spec
key = os.environ["YETO_ISLAND_HMAC_KEY"]
scmd = launcher.syncer_command(args, 2)
assert (getattr(args,"rl_island_scheduling","legacy")!="elastic") or key not in scmd and "YETO_ISLAND_HMAC_KEY:?" in scmd, "syncer cmd must not carry the key value"
spec = parse_gpu_spec(args.gpu)[0]
itask = launcher.make_miles_island_task(args, spec, 0, 1, "127.0.0.1:29400")
mcfg = launcher.build_modal_island_config(args, spec, 0, itask, "1.2.3.4:29400")
assert (getattr(args,"rl_island_scheduling","legacy")!="elastic") or (mcfg.envs.get("YETO_ISLAND_HMAC_KEY") == key and key not in mcfg.run_script), "modal cfg.envs lacks key / script leaks it"
print("HMAC check: modal cfg.envs has key, run_script/syncer cmd clean; soft deadline:", args.rl_soft_deadline_s)
spec1 = parse_gpu_spec(args.gpu)[1]
t1 = launcher.make_miles_island_task(args, spec1, 1, 2, "127.0.0.1:29400")
c1 = launcher.build_modal_island_config(args, spec1, 1, t1, "1.2.3.4:29400")
print("pause env island1 modal cfg:", c1.envs.get("YETO_RL_ELASTIC_DEBUG_PAUSE"), "| island0:", mcfg.envs.get("YETO_RL_ELASTIC_DEBUG_PAUSE"))
assert c1.envs.get("YETO_RL_ELASTIC_DEBUG_PAUSE") == args.rl_elastic_debug_pause
print("algorithm spec:", getattr(args, "rl_algorithm_spec", None), "| sync preset:", getattr(args, "rl_sync_preset", None))
task = cli._make_head_task(args, {})
cfg = task.to_yaml_config()
import inspect
src = inspect.getsource(cli.cmd_launch_head)
assert "secrets.update(launcher.island_hmac_secret(args))" in src and "secrets=secrets or None" in src
assert (getattr(args,"rl_island_scheduling","legacy")!="elastic") or (launcher.island_hmac_secret(args) == {"YETO_ISLAND_HMAC_KEY": key} and args.training_mode == "rl")
stask = launcher.make_syncer_task(args, 2)
_sec = getattr(stask, "secrets", None) or {}
print("syncer task secret NAMES:", sorted(_sec), "| key in run:", key in (stask.run or ""), "| key in setup:", key in (stask.setup or ""),
      "| key in envs:", key in str(getattr(stask, "envs", {})))
assert (getattr(args,"rl_island_scheduling","legacy")!="elastic") or ("YETO_ISLAND_HMAC_KEY" in _sec and key not in (stask.run or "") and key not in (stask.setup or ""))
print("HMAC check: make_syncer_task secrets has key, run/setup clean")
print("HMAC check: head job secrets include YETO_ISLAND_HMAC_KEY (cmd_launch_head path), value not printed")
print("head resources:", cfg.get("resources"), "| secret NAMES:", sorted(cfg.get("secrets") or {}), "| env NAMES:", sorted(cfg.get("envs") or {}))
print("head setup tail:", task.setup.splitlines()[-1])
print("head job argv _head payload keys:", len(cli._serializable_args(args)))
req = sky.launch(task, cluster_name="dryrun-island1b", dryrun=True)
sky.stream_and_get(req)
PY
  ) 2>&1 | grep -v "^\s*$" | tail -40
  exit 0
fi
date -u +%FT%TZ > $R/start_utc.txt
# teardown of THIS run only (Modal app + head cluster; never `sky down -a`)
cat > $R/teardown.sh <<EOF
cd /tmp; export HOME=/home/michael
timeout 300 $PY -m modal app stop --yes $APP > $R/\${1:-final}-stop.out 2>&1
timeout 900 $SKY down -y $HEAD_CL > $R/\${1:-final}-down.out 2>&1; echo "down rc=\$?" >> $R/\${1:-final}-down.out
sleep 15; timeout 120 $PY -m modal app list --json > $R/\${1:-final}-applist.json 2>&1
timeout 120 $SKY status $HEAD_CL > $R/\${1:-final}-skystatus.txt 2>&1
EOF
setsid nohup bash -c "sleep $WD; bash $R/teardown.sh watchdog; touch $R/WATCHDOG_FIRED" >/dev/null 2>&1 &
echo $! > $R/watchdog.pid
# killer (runs ON the head VM as its own sky job; Modal token as sky secret)
cat > $R/yeto/g3_killer.py <<'PY'
import glob, json, os, re, sys, time
import modal  # fail at start, not at kill time (1a-b)
name, delay, at_v, kill_island = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]); out = os.path.expanduser("~/g3-kill"); os.makedirs(out, exist_ok=True)
def head_log():
    best = ""
    for p in glob.glob(os.path.expanduser("~/sky_logs/*/run.log")):
        t = open(p, errors="replace").read()
        if "[modal]" in t and "YETO_RL_EVENT" in t and len(t) > len(best): best = t  # 1a-a: head log has no "[yeto]" tag
    return best
deadline = time.time() + 6000
while time.time() < deadline:
    text = head_log()
    calls = re.findall(rf"\[modal\] {re.escape(name)}: function call (\S+)", text)
    applied = False; rounds1 = 0
    for line in text.splitlines():
        at = line.find("YETO_RL_EVENT ")
        if at < 0: continue
        try: e = json.loads(line[at + 14:])
        except ValueError: continue
        if e.get("event") == "rl_policy_apply" and e.get("island_id") == kill_island and e.get("policy_version") == at_v: applied = True
        # 1b fallback: elastic islands may version differently -> also fire after island 1 finished at_v+1 local rounds
        if e.get("event") == "rl_local_round" and e.get("island_id") == kill_island: rounds1 = rounds1 + 1
    if rounds1 >= at_v + 1: applied = True
    if "finished:" in text or "rl_learner_finalized" in text:
        print("run finished before the kill point; no kill", flush=True); sys.exit(2)
    if calls and applied: break
    time.sleep(5)
else:
    print("killer deadline", flush=True); sys.exit(3)
seen = time.time(); time.sleep(delay)
cid = calls[0]
modal.FunctionCall.from_id(cid).cancel(terminate_containers=True)
rec = {"island": name, "call_id": cid, "vN_seen_unix": seen, "kill_unix": time.time()}
open(f"{out}/kill.json", "w").write(json.dumps(rec) + "\n"); print(rec, flush=True)
PY
cp $B/s15-island1b-rejoin-stopper.py $R/yeto/stopper.py
cd $R/yeto
( eval "timeout $HARD $PY -m yeto.cli $(cat $R/args.txt)" 2>&1 | tee $R/launch.stream.log | awk '{ print strftime("%FT%TZ", systime(), 1) " " $0; fflush() }' > $R/launch.ts.log; echo "rc=${PIPESTATUS[0]}" > $R/submit_rc.txt ) &
SUB=$!
# once the controller job is submitted: start the killer job on the VM
JOB=""
for i in $(seq 1 360); do
  JOB=$(grep -oP "submitted: job \K\d+" $R/launch.stream.log 2>/dev/null | head -1); [ -n "$JOB" ] && break
  kill -0 $SUB 2>/dev/null || break; sleep 10
done
echo "head_job=$JOB" > $R/head_job.txt
if [ -n "$JOB" ]; then
  cat > $R/killer.yaml <<EOF
name: island1b-killer
run: |
  for i in \$(seq 1 180); do [ -f ~/.yeto_head_ready ] && break; sleep 5; done
  if [ -x ~/yeto-head-py/bin/python3 ]; then export PATH=~/yeto-head-py/bin:\$PATH; fi  # 1a-b: modal lives in the head venv
  python3 -c "import modal; print('killer modal', modal.__version__)" || exit 5
  cd ~/sky_workdir && python3 g3_killer.py $L1 $KILL_DELAY $KILL_AT_V $KILL_ISLAND
EOF
  cat > $R/stopper.yaml <<EOF
name: island1b-stopper
run: |
  for i in \$(seq 1 180); do [ -f ~/.yeto_head_ready ] && break; sleep 5; done
  if [ -x ~/yeto-head-py/bin/python3 ]; then export PATH=~/yeto-head-py/bin:\$PATH; fi
  cd ~/sky_workdir && python3 stopper.py ${STOP_ISLAND:-1} $STOP_S ${STOP_AT_V:-1} ${STOP_AFTER:-10}
EOF
  [ "${STOP:-0}" = 1 ] && (cd /tmp && timeout 600 $SKY exec -d $HEAD_CL $R/stopper.yaml --secret MODAL_TOKEN_ID --secret MODAL_TOKEN_SECRET) > $R/stopper-submit.log 2>&1
  [ "${KILL:-0}" = 1 ] && (cd /tmp && timeout 600 $SKY exec -d $HEAD_CL $R/killer.yaml --secret MODAL_TOKEN_ID --secret MODAL_TOKEN_SECRET) > $R/killer-submit.log 2>&1
fi
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
S='ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20'
if timeout 60 $SKY status $HEAD_CL 2>/dev/null | grep -q ' UP '; then
  (cd /tmp && timeout 600 $SKY logs $HEAD_CL $JOB --no-follow) > $R/launch.log 2>&1
  (cd /tmp && timeout 300 $SKY queue $HEAD_CL) > $R/head/queue.txt 2>&1
  timeout 600 $S $HEAD_CL 'cd ~ && tar czf - yeto-output yeto-syncer.log g3-kill sky_logs 2>/dev/null' > $R/head/head.tgz 2>/dev/null
  (cd $R/head && tar xzf head.tgz 2>/dev/null)
  cp $R/head/g3-kill/kill.json $R/kill.json 2>/dev/null; cp $R/head/g3-kill/stop.json $R/stop.json 2>/dev/null
  timeout 60 $S $HEAD_CL 'hostname; nproc; free -g; nvidia-smi -L 2>&1|head -2; ss -ltnp 2>/dev/null|grep -E ":29400"' > $R/head/host.txt 2>&1
fi
[ -s $R/launch.log ] || cp $R/launch.stream.log $R/launch.log
# rc: the head job's terminal status (SUCCEEDED -> 0)
grep -E "^ *$JOB " $R/head/queue.txt 2>/dev/null | grep -q SUCCEEDED && echo "rc=0" > $R/rc.txt || echo "rc=1 ($(grep -E "^ *$JOB " $R/head/queue.txt 2>/dev/null | tr -s ' ' | cut -c1-120))" > $R/rc.txt
bash $R/teardown.sh final
AID=$(/usr/bin/python3 -c "import json,sys;print(next((a['app_id'] for a in json.load(open(sys.argv[1])) if a.get('description')==sys.argv[2]),''))" $R/final-applist.json $APP 2>/dev/null); [ -n "$AID" ] && (cd /tmp && timeout 180 $PY -m modal app logs $AID > $R/modal-app-logs.txt 2>&1)
mkdir -p $R/tape-direct; timeout 600 $PY -m modal volume get --force $TAPEVOL $APP $R/tape-direct > $R/tape-direct.out 2>&1
kill $(cat $R/watchdog.pid) 2>/dev/null
/usr/bin/python3 $B/s15-island1a-judge.py $R $EXPECT_GPU $STEPS > $R/judgment.json; /usr/bin/python3 $B/s15-island1b-judge.py $R $B/s15-island1a-20261007c > $R/judgment-1b.json 2>&1; /usr/bin/python3 $B/s15-island1b-rejoin-judge.py $R > $R/judgment-rejoin.json; /usr/bin/python3 $B/s15-island1b-pause-judge.py $R 1 $LEASE $STOP_S > $R/judgment-pause.json 2>&1
echo "done $P $(cat $R/rc.txt) judgment=$(head -c 300 $R/judgment.json)"

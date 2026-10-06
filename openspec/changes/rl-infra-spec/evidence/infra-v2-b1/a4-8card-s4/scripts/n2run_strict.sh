#!/bin/bash
# usage: n2run_strict.sh <prefix> <ngpu> <hard_s> <watchdog_s> <launch args...>
# Strict-avg variant of n2run.sh (recovery-design.md §10 / E1D-RECOVERY-PROGRESS "方案 A"): ONE Nebius island, NO --rl-single-island-no-sync,
# head mode = this machine (run_local_head.py -> cli.cmd_head -> launcher.LocalSyncer: the yeto-syncer binary runs HERE at $SYNCER_PUBLIC_IP:29400,
# HOME=$R/home so its checkpoint/log live under the run dir).  The syncer's current version is the in-place restart point of the learner
# (StrictAvgSync.start -> wait_for_initial_policy -> ledger.rebase(v)); a no-sync island has none (LedgerError at rebase(0)).
# Differences from n2run.sh: no `--controller local`, no `--rl-single-island-no-sync`; syncer binary + run_local_head.py copied into the run dir;
# the host syncer port must be free before the start (else "abort: ..." -> chain8 stops); after the launcher ends the host syncer tree is killed (idempotent).
# Precedent for a Nebius island reaching the head syncer on this host: infra-v2-b1-nsmoke-20260930-2 (rc=0, "learner connected" in launch.log).
set -u
T=$(ps -u michael -L -o pid= | wc -l); if [ "$T" -ge ${THREAD_MAX:-3000} ]; then echo "abort: $T user threads"; exit 3; fi   # THREAD_MAX: chain8.sh guard value / test hook
P=$1; NG=$2; HARD=$3; WD=$4; shift 4; EXTRA="$*"
B=/home/michael/work/gpu-b1-runs; R=${RUN_ROOT:-$B}/$P; SKY=/home/michael/work/gpu-head/venv/bin/sky
SYNCER_PORT=${SYNCER_PORT:-29400}; SYNCER_PUBLIC_IP=${SYNCER_PUBLIC_IP:-185.189.44.160}; SYNCER_BIN=${SYNCER_BIN:-/home/michael/work/gpu-default-modal/home/yeto-syncer}
CP=${CLUSTER_PREFIX:-$P}
# one head syncer per host: a busy port means another strict run (or an orphan) -- never start a second island against it
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":${SYNCER_PORT}\$"; then echo "abort: syncer port $SYNCER_PORT busy on this host ($(pgrep -af 'yeto-syncer --port' | head -2 | cut -c1-160 | tr '\n' ';'))"; exit 3; fi
[ -x $SYNCER_BIN ] || { echo "abort: syncer binary $SYNCER_BIN missing"; exit 3; }
mkdir -p $R/home $R/runs $R/pulled $R/yeto
for d in .sky .nebius .ssh; do ln -sfn /home/michael/$d $R/home/$d; done
cp $SYNCER_BIN $R/home/yeto-syncer; cp $B/run_local_head.py $R/
git -C /home/michael/work/gpu-b1 archive ${SHA:-47efd25} | tar x -C $R/yeto
cp /home/michael/work/gpu-default-modal/yeto/gsm8k_reward.py $R/yeto/
touch $R/yeto/yeto-rl-echo-events; echo ${SHA:-47efd25} > $R/yeto_sha.txt
echo "$SYNCER_PUBLIC_IP:$SYNCER_PORT" > $R/syncer_host.txt   # marker: this run has a host-side syncer (nstop_item_strict.sh / reset_island_strict.sh clean it)
echo "launch --training-mode rl --on-demand --gpu ${GPU_SPEC:-nebius:${NG}xh100@eu-north1} --cluster-prefix $CP ${KEEP:+--keep} --no-island-relaunch --modal-retries 0 --model Qwen/Qwen3-0.6B --model-revision c1899de289a04d12100db370d81485cdf75e47ca --data zhuzilin/gsm8k --data-revision 0cbd9f31d91ac21a7613dcbc7fef992adac459ae --reward-function gsm8k_reward:score --tuning lora --lora-r 16 --lora-targets all-linear --fragments 1 --pipeline 1 --rollout-batch-size 4 --n-samples-per-prompt 8 --rollout-max-response-len 384 --seq-len 1024 --inner-lr 1e-5 --seed 17 --apply-chat-template-kwargs '{\"enable_thinking\": false}' --trust-remote-code $EXTRA" > $R/args.txt
CL=$CP-l0-eu-north1; echo $CL > $R/cluster.txt
if [ "${SHARED:-0}" != 1 ]; then
setsid nohup bash -c "sleep $WD; HOME=/home/michael $SKY down -y $CL > $R/watchdog.out 2>&1" >/dev/null 2>&1 &
echo $! > $R/watchdog.pid
fi
setsid nohup bash -c "
export HOME=/home/michael
as=0; lastsz=x; lastt=\$(date +%s)
while [ ! -f $R/rc.txt ]; do
  if timeout 60 $SKY status $CL 2>/dev/null | grep -q ' UP \| INIT '; then
    [ \$as = 1 ] || [ \"${SHARED:-0}\" = 1 ] || { timeout 120 $SKY autostop -y -i 10 --down $CL > $R/autostop.out 2>&1 && as=1; }   # chain mode: the chain-level watchdog/cleanup owns the cluster
    S='ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 $CL'
    timeout 60 \$S 'cat ~/yeto-output/rl-island-0.jsonl 2>/dev/null' > $R/pulled/.tmp 2>/dev/null && [ -s $R/pulled/.tmp ] && mv $R/pulled/.tmp $R/pulled/rl-island-0.jsonl
    [ -s $R/pulled/gpu.txt ] || timeout 60 \$S 'nvidia-smi --query-gpu=index,uuid,name,driver_version --format=csv,noheader' > $R/pulled/gpu.txt 2>/dev/null
    timeout 60 \$S 'date -u +%FT%TZ; nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader' >> $R/pulled/compute-apps.txt 2>/dev/null
    timeout 60 \$S 'date -u +%FT%TZ; ps -eo pid,args --no-headers | grep -E \"ray::|sglang|scheduler|yeto.rl.learner\" | grep -v grep | cut -c1-160' >> $R/pulled/ps.txt 2>/dev/null
    timeout 60 \$S 'cat ~/yeto-rl/inwatch.log 2>/dev/null' > $R/pulled/inwatch.log 2>/dev/null
    timeout 60 \$S 'cd ~/yeto-rl 2>/dev/null && tar czf - --exclude=cuts elastic-state | base64 -w0' > $R/pulled/.st 2>/dev/null && [ -s $R/pulled/.st ] && mv $R/pulled/.st $R/pulled/elastic-state.tgz.b64
    timeout 60 \$S 'cd ~/yeto-rl 2>/dev/null && tar czf - router_samples.jsonl gpu_samples.jsonl sampler.out inwatch.log inwatch.out dkill.log dctl.log term_probe.log 2>/dev/null | base64 -w0' > $R/pulled/.sm 2>/dev/null && [ -s $R/pulled/.sm ] && mv $R/pulled/.sm $R/pulled/samplers.tgz.b64
  fi
  sz=\$(stat -c %s $R/pulled/rl-island-0.jsonl 2>/dev/null || echo 0); now=\$(date +%s)
  if [ \"\$sz\" != \"\$lastsz\" ]; then lastsz=\$sz; lastt=\$now; fi
  if [ \$(( now - lastt )) -ge 1200 ] && [ ! -f $R/progress_stop.txt ]; then
    echo \"no new events for 20 min at \$(date -u +%FT%TZ); stopping\" > $R/progress_stop.txt
    cp $R/pulled/rl-island-0.jsonl $R/pulled/at-stop.jsonl 2>/dev/null
    timeout 300 $SKY down -y $CL >> $R/progress_stop.txt 2>&1
  fi
  sleep 10
done" > $R/puller.log 2>&1 &
echo $! > $R/puller.pid
(
export HOME=$R/home YETO_RUNS_DIR=$R/runs PYTHONPATH=$R/yeto SYNCER_PUBLIC_IP=$SYNCER_PUBLIC_IP
eval "$(/usr/bin/python3 - <<'PY'
import base64, json, shlex
a = json.load(open("/home/michael/.docker/config.json"))["auths"]["ghcr.io"]["auth"]
u, t = base64.b64decode(a).decode().split(":", 1)
print(f"export SKYPILOT_DOCKER_USERNAME={shlex.quote(u)} SKYPILOT_DOCKER_PASSWORD={shlex.quote(t)} SKYPILOT_DOCKER_SERVER=ghcr.io")
PY
)"
cd $R/yeto
date -u +%FT%TZ > $R/start_utc.txt
# head controller on this host: cmd_head starts the LocalSyncer subprocess (own process group), then launcher.run(local_syncer=...) launches the island
eval "timeout $HARD /home/michael/work/gpu-head/venv/bin/python $R/run_local_head.py $R/args.txt" 2>&1 | tee $R/launch.log | awk '{ print strftime("%FT%TZ", systime(), 1) " " $0; fflush() }' > $R/launch.ts.log
echo "rc=${PIPESTATUS[0]}" > $R/rc.txt.tmp
date -u +%FT%TZ > $R/end_utc.txt
# the hard timeout SIGTERMs python before cmd_head's finally (syncer.stop()) runs: kill the syncer tree ourselves (scoped to this run dir; idempotent)
$B/syncer_host_clean.sh $R >> $R/syncer_clean.txt 2>&1
sleep 15; mv $R/rc.txt.tmp $R/rc.txt
)

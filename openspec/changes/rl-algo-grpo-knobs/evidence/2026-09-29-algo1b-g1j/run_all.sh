#!/bin/bash
# algo1b G1 (plan.md). Run from this directory. Hard limits: sandbox timeout 10800 s,
# independent watchdog kills the sandbox at 11100 s, each mechanism exec 1800 s.
set -u
P=/tmp/modal-venv/bin/python; W=/tmp/algo1b-g1j; mkdir -p $W out
SHA=$(git -C /home/michael/work/algo-1b rev-parse HEAD)
git -C /home/michael/work/algo-1b archive --format=tar.gz -o $W/yeto.tgz HEAD
echo $SHA > harness/YETO_SHA; tar czf $W/harness.tgz harness
date -u +%FT%TZ > out/t_start
SID=$($P sbx.py create 'H100!' 10800 2>/dev/null | tail -1); echo $SID > out/sandbox_id; echo "sandbox $SID"
setsid nohup bash -c "sleep 11100; $P $(pwd)/sbx.py kill $SID > $(pwd)/out/watchdog.log 2>&1" >/dev/null 2>&1 &
echo $! > out/watchdog_pid
trap '$P sbx.py kill $SID >> out/teardown.log 2>&1; date -u +%FT%TZ > out/t_end' EXIT
$P sbx.py put $SID $W/yeto.tgz /work/yeto && $P sbx.py put $SID $W/harness.tgz /work || exit 1
T=2400 $P sbx.py exec $SID "bash /work/harness/setup.sh" > out/setup.log 2>&1
grep -q SETUP_OK out/setup.log || { echo "setup failed (see out/setup.log)"; exit 2; }
for m in ${MECHS:-clip_sym clip_hi}; do
  T=1800 $P sbx.py exec $SID "export PATH=\$HOME/.cargo/bin:\$PATH; mkdir -p /work/out && cd /work/yeto && export PYTHONPATH=/opt/miles-next:/work/yeto:/work/harness:\$PYTHONPATH HF_HUB_DISABLE_TELEMETRY=1; SEED=17 OPT_STEPS=3 INNER_LR=1e-4 python /work/harness/g1.py $m /work/out/$m > /work/out/$m.log 2>&1; echo rc=\$?; tail -3 /work/out/$m.log; tar czf /tmp/$m.tgz --exclude='*.pt' --exclude='*.safetensors' --exclude='*.ckpt*' -C /work/out $m $m.log" > out/$m.exec.log 2>&1
  $P sbx.py get $SID /tmp/$m.tgz $W/$m.tgz >/dev/null 2>&1 && mkdir -p out/$m && tar xzf $W/$m.tgz -C out/ || echo "fetch failed $m" >> out/errors.txt
  tail -2 out/$m.exec.log
done

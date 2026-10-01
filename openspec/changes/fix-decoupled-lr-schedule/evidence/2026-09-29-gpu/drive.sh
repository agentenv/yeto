#!/bin/bash
# drive.sh <tag> <legacy|ports> <gpu> <steps...>   steps: parse strict dec
TAG=$1; ENG=$2; GPU=$3; shift 3; C=/tmp/lrfix/ctl; cd $C; P=/tmp/modal-venv/bin/python
EV=/tmp/lrfix/ev; mkdir -p $EV
teardown() { S=$(cat $TAG.sid 2>/dev/null); [ -n "$S" ] && { T=300 $P sbx.py exec $S "nvidia-smi; cd /opt/miles-legacy 2>/dev/null || cd /opt/miles-next; git rev-parse HEAD; cd /opt/sglang-legacy 2>/dev/null || cd /opt/sglang-next; git rev-parse HEAD; pip freeze 2>/dev/null" > $EV/env-$TAG.txt 2>&1; $P sbx.py kill $S; }; date -u +%s > $TAG.t_down; echo DOWN; }
trap 'echo TRAP; teardown' EXIT
date -u +%s > $TAG.t_create
timeout 3600 $P sbx.py create $ENG $GPU 7200 > $TAG.create.log 2>&1 || { echo CREATE_FAIL; tail $TAG.create.log; exit 1; }
tail -1 $TAG.create.log > $TAG.sid; S=$(cat $TAG.sid); date -u +%s > $TAG.t_sbx; echo SBX $S
X() { T=${T:-600} $P sbx.py exec $S "$1" 2>&1 | grep -v -i deprec; }
X "nvidia-smi -L" | tee $EV/gpu-$TAG.txt
case $GPU in L40S*) X "nvidia-smi -L | grep -q L40S" | true; X "nvidia-smi --query-gpu=name --format=csv,noheader | grep -v L40S | grep -q . && echo GPU_MISMATCH || echo GPU_OK" | tee -a $EV/gpu-$TAG.txt; grep -q GPU_OK $EV/gpu-$TAG.txt || exit 2;; esac
X "mkdir -p /work/out"
$P sbx.py put $S yeto.tgz /work >/dev/null; $P sbx.py put $S harness.tgz /work >/dev/null; $P sbx.py put $S data.tgz /work >/dev/null
cp YETO_SHA $EV/YETO_SHA
if [ $ENG = legacy ]; then MR=/opt/miles-legacy; $P sbx.py put $S shim.tgz /usr/local/lib/python3.12/dist-packages >/dev/null
  X "echo 'import yeto_legacy_router_timeout' > /usr/local/lib/python3.12/dist-packages/zz_yeto_legacy_router_timeout.pth"; else MR=/opt/miles-next; fi
T=1500 X "export PATH=\$HOME/.cargo/bin:\$PATH; cd /work/yeto/syncer && cargo build --release --locked --quiet 2>&1 | grep -i '^error'; ls target/release/yeto-syncer"
COMMON="--model Qwen/Qwen3-0.6B --model-revision c1899de289a04d12100db370d81485cdf75e47ca --data /work/data/gsm8k.jsonl --reward-function gsm8k_reward:score --islands 2 --gpus-per-island 1 --groups-per-island 4 --samples-per-group 8 --rollout-max-response-len 384 --seq-len 1024 --lora-r 16 --lora-targets all-linear --eval-prompts 8 --eval-samples-per-prompt 1 --pass-k 1 --eval-device cuda --miles-root $MR --apply-chat-template-kwargs '{\"enable_thinking\": false}' --trust-remote-code --seeds 17 --rl-engine $ENG"
runstep() { N=$1; LA=$2
  echo "$LA" > $EV/launch-args-$N.txt
  T=5400 X "export PATH=\$HOME/.cargo/bin:\$PATH; cd /work/yeto && export PYTHONPATH=$MR:/work/yeto:/work/harness:\$PYTHONPATH && rm -rf /work/out/$N && mkdir -p /work/out/$N && date -u +%s > /work/out/$N/t_start; python scripts/benchmark_rl.py $LA --work-dir /work/out/$N/work --report-dir /work/out/$N/report > /work/out/$N/launch.log 2>&1; echo \$? > /work/out/$N/rc; date -u +%s > /work/out/$N/t_end; echo $N rc=\$(cat /work/out/$N/rc); tail -3 /work/out/$N/launch.log; cd /work/out && tar czf /tmp/$N.tgz --exclude='*.pt' --exclude='*.safetensors' --exclude='*.f32' --exclude='state.ckpt*' $N"
  $P sbx.py get $S /tmp/$N.tgz $C/$TAG-$N.tgz; tar xzf $C/$TAG-$N.tgz -C $EV; echo "copied $N rc=$(cat $EV/$N/rc)"
}
for step in "$@"; do case $step in
 parse) T=1200 X "cd /work/yeto && export PYTHONPATH=$MR:/work/yeto:\$PYTHONPATH && (python -c 'import pytest' 2>/dev/null || pip install -q pytest) && python -c 'import miles.utils.arguments, megatron.training; print(\"imports ok\")' && python -m pytest -q -rs -p no:cacheprovider tests/test_rl_miles_adapter_config.py tests/test_rl_applied_lr.py tests/test_rl_argv_snapshot.py; echo pytest_rc=\$?" > $EV/parse-args-$TAG.log 2>&1; tail -15 $EV/parse-args-$TAG.log;;
 strict) runstep strict2-$ENG "$COMMON --arms federated --global-rounds 3 --inner-lr 0.0001 --fragments 8 --pipeline 2 --local-horizon 4 --arm-timeout-min 45";;
 dec) runstep dec-$ENG "$COMMON --arms decoupled --global-rounds 4 --fragments 4 --pipeline 2 --local-horizon 2 --inner-lr 1e-5 --arm-timeout-min 60";;
esac; done
echo ALLDONE

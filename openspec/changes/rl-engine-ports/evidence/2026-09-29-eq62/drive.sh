#!/bin/bash
# drive.sh <tag> <engine legacy|ports> <preset strict|dec>  -- one sandbox, all seeds + TF, guaranteed teardown
TAG=$1; ENG=$2; PRE=$3; C=/home/michael/work/gpu-eq/ctl; cd $C; P=/tmp/modal-venv/bin/python
if [ $PRE = strict ]; then EV=/home/michael/work/gpu-eq/evidence/2026-09-29-eq62; else EV=/home/michael/work/gpu-eq/evidence/2026-09-29-eq63; fi
mkdir -p $EV/env-$ENG; L=$C/$TAG.log
teardown() { S=$(cat $TAG.sid 2>/dev/null); [ -n "$S" ] && { T=300 $P sbx.py exec $S "nvidia-smi; cd /opt/miles-legacy 2>/dev/null || cd /opt/miles-next; git rev-parse HEAD; cd /opt/sglang-legacy 2>/dev/null || cd /opt/sglang-next; git rev-parse HEAD; pip freeze 2>/dev/null" > $EV/env-$ENG/sandbox-env.txt 2>&1; $P sbx.py kill $S; }; date -u +%s > $TAG.t_down; echo DOWN; }
trap 'echo TRAP; teardown' EXIT
date -u +%s > $TAG.t_create
timeout 3600 $P sbx.py create $ENG H100:2 10800 > $TAG.create.log 2>&1 || { echo CREATE_FAIL; tail $TAG.create.log; exit 1; }
tail -1 $TAG.create.log > $TAG.sid; S=$(cat $TAG.sid); date -u +%s > $TAG.t_sbx; echo SBX $S
X() { T=${T:-600} $P sbx.py exec $S "$1" 2>&1 | grep -v -i deprec; }
X "mkdir -p /work/out"
$P sbx.py put $S yeto.tgz /work >/dev/null; $P sbx.py put $S harness.tgz /work >/dev/null; $P sbx.py put $S data.tgz /work >/dev/null
if [ $ENG = legacy ]; then MR=/opt/miles-legacy; $P sbx.py put $S shim.tgz /usr/local/lib/python3.12/dist-packages >/dev/null
  X "echo 'import yeto_legacy_router_timeout' > /usr/local/lib/python3.12/dist-packages/zz_yeto_legacy_router_timeout.pth"; else MR=/opt/miles-next; fi
T=1500 X "export PATH=\$HOME/.cargo/bin:\$PATH; cd /work/yeto/syncer && cargo build --release --locked --quiet 2>&1 | grep -i '^error'; ls target/release/yeto-syncer; wc -l /work/data/gsm8k.jsonl"
COMMON="--model Qwen/Qwen3-0.6B --model-revision c1899de289a04d12100db370d81485cdf75e47ca --data /work/data/gsm8k.jsonl --reward-function gsm8k_reward:score --islands 2 --gpus-per-island 1 --groups-per-island 4 --samples-per-group 8 --rollout-max-response-len 384 --seq-len 1024 --lora-r 16 --lora-targets all-linear --eval-prompts 8 --eval-samples-per-prompt 1 --pass-k 1 --eval-device cuda --miles-root $MR --apply-chat-template-kwargs '{\"enable_thinking\": false}' --trust-remote-code"
if [ $PRE = strict ]; then LA="$COMMON --arms federated --global-rounds 3 --arm-timeout-min 45"; else LA="$COMMON --arms decoupled --global-rounds 8 --fragments 8 --pipeline 2 --local-horizon 4 --arm-timeout-min 60"; fi
echo "$LA" > $EV/launch-args-$ENG.txt
runstep() { # name seed extra envprefix
  N=$1; SD=$2; EXTRA=$3; ENVP=$4
  T=5400 X "export PATH=\$HOME/.cargo/bin:\$PATH; cd /work/yeto && export PYTHONPATH=$MR:/work/yeto:/work/harness:\$PYTHONPATH && rm -rf /work/out/$N && mkdir -p /work/out/$N && date -u +%s > /work/out/$N/t_start; $ENVP python scripts/benchmark_rl.py $LA $EXTRA --seeds $SD --rl-engine $ENG --work-dir /work/out/$N/work --report-dir /work/out/$N/report > /work/out/$N/launch.log 2>&1; echo \$? > /work/out/$N/rc; date -u +%s > /work/out/$N/t_end; echo $N rc=\$(cat /work/out/$N/rc); tail -3 /work/out/$N/launch.log; cd /work/out && tar czf /tmp/$N.tgz --exclude='*.pt' --exclude='*.safetensors' --exclude='*.f32' --exclude='state.ckpt*' $N; find $N \\( -path '*/audit/round-00000001.*' -a -path 'tf-*' \\) -o -path '*/adapter/adapter_model.safetensors' | tar czf /tmp/$N.x.tgz -T -"
  $P sbx.py get $S /tmp/$N.tgz $C/$TAG-$N.tgz; $P sbx.py get $S /tmp/$N.x.tgz $C/$TAG-$N.x.tgz
  tar xzf $C/$TAG-$N.tgz -C $EV && tar xzf $C/$TAG-$N.x.tgz -C $EV; echo "copied $N rc=$(cat $EV/$N/rc)"
}
TFX="--global-rounds 1 --custom-generate-function-path yeto.rl.teacher_forcing.replay_generate"
REPLAY="YETO_RL_REPLAY_ROLLOUTS='/work/out/legacy-s17/work/seed-17/*/rollouts/island-*/0.pt'"
if [ $ENG = legacy ]; then
  runstep legacy-s17 17
  T=600 X "cd /work/out && tar czf /tmp/pt.tgz legacy-s17/work/seed-17/*/rollouts/island-*/0.pt && ls -la legacy-s17/work/seed-17/*/rollouts/island-*/"
  $P sbx.py get $S /tmp/pt.tgz $C/pt-$PRE.tgz && touch $C/pt-$PRE.READY
  runstep tf-legacy 17 "$TFX" "$REPLAY"
  for s in 18 19 20 21; do runstep legacy-s$s $s; done
else
  for s in 17 18 19 20 21; do runstep ports-s$s $s; done
  for i in $(seq 180); do [ -f $C/pt-$PRE.READY ] && break; sleep 20; done
  $P sbx.py put $S $C/pt-$PRE.tgz /work/out >/dev/null && runstep tf-ports 17 "$TFX" "$REPLAY"
fi
echo ALLDONE

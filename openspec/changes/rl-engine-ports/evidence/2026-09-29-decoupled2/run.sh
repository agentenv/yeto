N=decoupled2
export PATH=$HOME/.cargo/bin:$PATH; cd /work/yeto
export PYTHONPATH=/opt/miles-next:/work/yeto:/work/harness:$PYTHONPATH
rm -rf /work/out/$N && mkdir -p /work/out/$N
python scripts/benchmark_rl.py --model Qwen/Qwen3-0.6B --model-revision c1899de289a04d12100db370d81485cdf75e47ca --data /work/data/gsm8k.jsonl --reward-function gsm8k_reward:score --arms decoupled --islands 2 --gpus-per-island 1 --global-rounds ${ROUNDS:-8} --groups-per-island 4 --samples-per-group 8 --fragments 8 --pipeline 2 --local-horizon 4 --seeds 17 --rollout-max-response-len 384 --seq-len 1024 --lora-r 16 --lora-targets all-linear --eval-prompts 8 --eval-samples-per-prompt 1 --pass-k 1 --eval-device cuda --miles-root /opt/miles-next --apply-chat-template-kwargs '{"enable_thinking": false}' --trust-remote-code --rl-engine ports --arm-timeout-min 60 --work-dir /work/out/$N/work --report-dir /work/out/$N/report > /work/out/$N/benchmark.log 2>&1
echo rc=$? > /work/out/$N/rc

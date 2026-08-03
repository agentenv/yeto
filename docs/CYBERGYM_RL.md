# CyberGym RL integration

Yeto includes an experimental local RL loop that generates candidate
proof-of-concept (PoC) bytes, submits them to CyberGym's `/submit-vul`
endpoint, and uses the vulnerable runner's exit code as the reward for a PPO
update.

This is an integration and training smoke path. It is not yet a full
CyberGym agent: the model currently receives a task ID rather than the task
repository and description, and a crash on the vulnerable runner is not a
verified benchmark solve until the PoC is also checked against the fixed
runner.

## Local setup

CyberGym executes uploaded data against vulnerable Docker images. Keep the
server local; never expose its port to the public internet.

Install Yeto in its repository:

```bash
python -m venv yeto_rl_env
source yeto_rl_env/bin/activate
pip install -e .
```

In a separate CyberGym checkout, use the same environment or another Python
environment, install its server dependencies, and download the ten runner
images used by Yeto's default task list:

```bash
pip install -e '.[dev,server]'
python scripts/server_data/download_subset.py --max-workers 4
```

Start the server on the loopback interface. The current Yeto adapter uses
the raw task IDs, so omit `--mask_map_path`:

```bash
POC_SAVE_DIR=./server_poc
python -m cybergym.server \
  --host 127.0.0.1 \
  --port 8666 \
  --log_dir "$POC_SAVE_DIR" \
  --db_path "$POC_SAVE_DIR/poc.db"
```

If the server returns an error such as `No such image:
n132/arvo:47101-vul`, the subset download is missing or incomplete. Finish
that download before training. Yeto aborts on HTTP and connectivity errors
so infrastructure failures are not recorded as negative training rewards.

## Run a smoke update

From the Yeto checkout, with the CyberGym server running:

```bash
yeto rl \
  --env cybergym \
  --model Qwen/Qwen2.5-0.5B \
  --server-host 127.0.0.1 \
  --server-port 8666 \
  --iterations 1 \
  --steps 16 \
  --epochs 1 \
  --output ./integration_test
```

Exit codes `0` and `300` mean that the candidate did not crash the vulnerable
runner and receive reward `-1`; other exit codes receive reward `+1`. The
command saves the model and tokenizer plus `policy_state_dict.pt`, which also
contains the value-head parameters.

The `$10` budget displayed by this command is monitoring metadata only. This
path runs locally and does not launch a Yeto SkyPilot fleet.

## Run the direct-Miles comparison

The branch also includes a real Miles reward adapter at
`yeto_miles_cybergym.reward.score`. It submits the generated response as the
CyberGym PoC multipart file, computes the documented checksum, and accepts
both Miles' single-sample and batched async reward contracts. Generate the
same task roster used by the local smoke loop:

```bash
python -m yeto_miles_cybergym.prompts \
  --output ./cybergym_prompts.jsonl
```

Do not repair the old `miles` conda environment by installing an arbitrary
SGLang wheel. Miles depends on a patched SGLang/Megatron/CUDA stack. Start a
clean checkout at the pinned commit inside the immutable Miles image instead:

```bash
git clone https://github.com/radixark/miles ~/miles-clean
git -C ~/miles-clean fetch --depth 1 origin \
  dfc66ff38752bfa2c5d325e0037ebc4b537c06de
git -C ~/miles-clean checkout --detach \
  dfc66ff38752bfa2c5d325e0037ebc4b537c06de

docker run --rm -it --gpus all --network host --ipc host --shm-size=64g \
  -e HF_TOKEN="$HF_TOKEN" \
  -v "$PWD":/workspace/yeto:ro \
  -v "$HOME/miles-clean":/workspace/miles:ro \
  -w /workspace/yeto \
  radixark/miles@sha256:95b3afa9ee4313f5633e6ed3779c8276353cc8e24a2462e4f54ec0d5978fbae7 \
  bash
```

Inside the container, with the CyberGym server still listening on the host's
`127.0.0.1:8666`, run the three-iteration, four-sample direct-Miles smoke:

```bash
python -m yeto_miles_cybergym.prompts \
  --output /tmp/cybergym_prompts.jsonl
python -m yeto_miles_cybergym.launcher \
  --miles-root /workspace/miles \
  --prompt-data /tmp/cybergym_prompts.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --iterations 3 \
  --samples-per-iteration 4 \
  --samples-per-prompt 2
```

The launcher verifies the Miles commit, checks that `miles` and `sglang` can
be imported, starts Ray with one training GPU and one rollout GPU, and prints
the exact `ray job submit` command. Use `--dry-run` to inspect that command
without starting Ray or contacting CyberGym. `samples-per-prompt` is at least
two because one sample per GRPO group has no variance signal.

This direct baseline uses Miles' CI-gated experimental FSDP path; the pinned
image rejects non-CI FSDP runs because that backend is not actively maintained.
It is a protocol/throughput comparison against Yeto's strict distributed LoRA
path, not an identical optimizer recipe. Keep the model revision, task JSONL,
response length, temperature, learning rate, number of rollouts, and K fixed
when comparing the reward traces.

## Run the strict Yeto comparison

For the Yeto side, use the same JSONL and reward source. The two one-GPU
islands must be able to reach the CyberGym URL; `localhost` only works when the
server is on each island. Replace the GPU and model revision placeholders with
your pinned SkyPilot resources:

```bash
yeto launch --training-mode rl \
  --gpu '<cloud>:1xh100@<region-a>,<cloud>:1xh100@<region-b>' \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --model-revision <40-char-model-commit> \
  --data ./cybergym_prompts.jsonl \
  --data-format openai \
  --tuning lora --lora-r 8 --lora-alpha 8 \
  --seq-len 1024 --inner-lr 1e-5 \
  --rl-global-rounds 3 \
  --rl-groups-per-island-round 1 \
  --rl-samples-per-group 2 \
  --rl-local-optimizer-steps 1 \
  --reward-function yeto_miles_cybergym.reward:score \
  --cybergym-url http://<reachable-cybergym-host>:8666 \
  --learner-image \
    radixark/miles@sha256:95b3afa9ee4313f5633e6ed3779c8276353cc8e24a2462e4f54ec0d5978fbae7 \
  --trust-remote-code \
  --confirm
```

Yeto attests the reward source and model/data revisions before launch. A
CyberGym HTTP error or missing runner image aborts the run; it is not counted
as a negative reward. With two islands, `G=1, K=2` gives four total
CyberGym submissions per global round, matching the direct and local-Yeto
smokes above.

## Checks performed

The branch was exercised with:

```bash
pytest tests/test_cybergym_checksum.py -v
```

The unit/integration checks cover checksum construction, reward semantics,
server-error handling, and optional connectivity to a local CyberGym server.

An end-to-end run was also completed with Qwen2.5-0.5B, one iteration, 16
environment steps, and one PPO epoch. All 16 submissions reached real
CyberGym task containers over HTTP 200, the update completed with
`loss=38.8535` and mean `reward=-0.88`, and the model artifact was saved.
Fifteen vulnerable-runner submissions returned exit code `0`; one returned
exit code `1`. The latter is a crash signal used for training, not a claimed
verified CyberGym solve.

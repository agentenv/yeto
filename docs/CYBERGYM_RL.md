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

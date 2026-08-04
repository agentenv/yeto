# CyberGym RL

Yeto provides an experimental single-process PPO command for exercising a
causal language model against a local CyberGym server. It is independent of
the distributed Miles RL path used by `yeto launch --training-mode rl`.

Install the optional environment dependencies:

```bash
pip install -e '.[local-rl]'
```

CyberGym executes uploaded proof-of-concept bytes inside vulnerable runner
images. Bind its server to loopback and do not expose the service publicly.
After the required runner images are available, run one local update with:

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

Exit codes `0`, `300`, or a missing code receive reward `-1`; other exit
codes receive reward `+1`. HTTP, connectivity, and runner-image failures abort
the update instead of becoming negative rewards. The output contains the
Hugging Face model and tokenizer plus `policy_state_dict.pt`, which also
contains the scalar value head.

This path currently prompts with the task ID rather than the task repository
and description. A vulnerable-runner crash is a training signal, not a
verified benchmark solve against the fixed runner.

## Miles reward adapter

`yeto_miles_cybergym.reward:score` implements the same CyberGym submission
contract for distributed Miles RL. Generate a deterministic JSONL prompt set
with task IDs preserved in each row's metadata:

```bash
python -m yeto_miles_cybergym.prompts \
  --output ./cybergym_prompts.jsonl
```

Pass that file and reward callable to `yeto launch --training-mode rl`.
`--cybergym-url`, `--cybergym-agent-id`, and `--cybergym-timeout` are forwarded
to every Miles island. Set `CYBERGYM_API_KEY` in the submitting environment
when authentication is required; Yeto forwards it through the job environment
without recording it in the launch arguments.

The LM benchmark accepts this prompt file and reward callable unchanged. Use
`--arms native` to run only the direct Miles reference, or select any
comma-separated combination of `native`, `single`, and `federated`.

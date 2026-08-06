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

The default reward mode remains the vulnerable-runner binary reward. To use
the private shaped scorer, configure the Miles workers with:

```bash
export CYBERGYM_REWARD_SCHEME=shaped_v1
export CYBERGYM_REWARD_VIEW=train  # use final for binary benchmark reporting
export CYBERGYM_API_KEY=...
```

`shaped_v1` calls CyberGym's authenticated `/score-poc` endpoint. The server
must have its private reward schemas and `vul-cov` runners installed. `train`
uses `2 * shaped_score - 1`; `final` uses the vulnerable crash plus fixed
non-crash result only.

### Text-only Level 1 approximation

The Miles rollout remains a one-shot text generation rather than a CyberGym
agent with a source workspace and shell. To give that rollout the information
available at Level 1, materialize source-enriched prompt rows before training
or evaluation:

```bash
python -m yeto_miles_cybergym.text_level1 \
  --prompts ./train-110-curriculum.jsonl \
  --tasks /path/to/cybergym-data/tasks.json \
  --data-root /path/to/cybergym-data \
  --output ./train-110-text-level1.jsonl \
  --max-source-chars 12000 \
  --chunk-lines 80 \
  --max-snippets 6
```

`--data-root` may be either the dataset checkout containing `data/` or the
`data/` directory itself. Run the same command for the held-out prompt file.
The builder reads only the `repo-vul.tar.gz` and `description.txt` declared by
`task_difficulty.level1`. It ignores generated/build/vendor trees, divides
recognized source files into bounded chunks, ranks them deterministically by
token overlap with the description, and places the highest-ranked excerpts in
the final user message. Archive links are ignored without being followed. It
does not use `error.txt`, `patch.diff`, or the fixed repository.

Every output row retains its original label, tools, system messages, task
instruction, and task metadata. It also records the description/archive
SHA-256 values, selected file and line ranges, excerpt hashes, and these
explicit labels:

```json
{
  "cybergym_prompt_level": "text_level1",
  "cybergym_official_level1": false
}
```

This is a **text-only approximation of Level 1**, not an official complete
Level 1 run: the model cannot browse the repository, invoke tools, iterate on a
PoC, or validate against the fixed runner. GRPO, rollout grouping, and the
existing CyberGym reward callable are unchanged. Set the source-character
budget and RL context length together so the chat template does not truncate
the excerpts.

Pass the enriched file and reward callable to
`yeto launch --training-mode rl`.
`--cybergym-url`, `--cybergym-agent-id`, and `--cybergym-timeout` are forwarded
to every Miles island. Set `CYBERGYM_API_KEY` in the submitting environment
when authentication is required; Yeto forwards it through the job environment
without recording it in the launch arguments.

The LM benchmark accepts this prompt file and reward callable unchanged. Use
`--arms native` to run only the direct Miles reference, or select any
comma-separated combination of `native`, `single`, `federated`, and
`decoupled`.

For a baseline-informed curriculum, keep the historical baseline-selection
manifest and prompt rows together, then generate separate train and held-out
files:

```bash
python scripts/select_cybergym_curriculum.py \
  --prompts ./cybergym_all_prompts.jsonl \
  --selection-manifest ./baseline-selection/manifest.json \
  --train-output ./train-110-curriculum.jsonl \
  --eval-output ./heldout-10.jsonl \
  --manifest-output ./curriculum-manifest.json
```

To reduce zero-advantage GRPO groups, add Miles oversampling and its shipped
nonzero-reward-variance filter to the RL launch:

```bash
--rollout-batch-size 4 \
--n-samples-per-prompt 8 \
--over-sampling-batch-size 16 \
--dynamic-sampling-filter-path \
  miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
```

The filter is an intervention, not a new control: compare it against the
archived teammate baseline using the same model, task manifest, seed schedule,
reward contract, and optimizer budget.

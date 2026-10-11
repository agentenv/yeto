# RLF → Yeto Codex RL runbook

The online RL harness is Codex only. rlenvfactory/AgentFlow may produce or review offline task packages, but its node output is never used as task identity, grader evidence, reward, or rollout input. SWE and the legacy `rl-agentic-reward-env` change remain deferred.

## Apply order

1. Generate or install an rlenvfactory CTF package and retain `public/`, `task.yaml`, `MANIFEST.json`, source commit, package/image digests, and runtime quality-gate evidence.
2. Admit the package with `yeto.rl.rlf.admit_task_package`; reject packages without a trusted runtime gate.
3. Acquire one Modal VM lease per `(task_id, episode_id, trajectory_id)`. Keep secrets on the trusted side and tear down the lease after grading.
4. Run the existing Codex subprocess harness. Copy RLF identities and policy identity into every chain/compaction segment; preserve tokens, logprobs, loss masks, and segment metadata.
5. Finish and grade on the trusted side, then pass the resulting metadata through the Miles or generic verl preflight. Bare float rewards are rejected.
6. Run the CPU gate before any paid GPU run. A healthy RL gate must show nonzero reward variance, success count, absolute advantage sum, and policy update.

The baseline profile is Qwen3-30B-A3B thinking mode with a 32K window, 1024 generated tokens per turn, at most 48 turns, and 8192-byte tool output.

## Qwen3 cyber baseline profile (5.6)

The registered backend profile is `qwen3_30b_a3b` (`yeto.rl.codex_backend`,
allowlisted in `yeto/rl/harness/codex/pins.py`). Pass it per launch with
`--codex-backend-profile qwen3_30b_a3b`. Its budgets are declared on the
profile; three of them are launch-time knobs that must be set explicitly
because their global defaults deliberately stay unchanged:

| Budget | Profile value | How it takes effect |
| --- | --- | --- |
| context window | 32768 tokens | `--agent-max-seq-len 32768` (run config `agent_max_seq_len`) |
| generated tokens per turn | 1024 | `YETO_CODEX_BACKEND_MAX_TOKENS=1024` |
| turns | 48 | `SECRLENV_MAX_TURNS=48` (worker-passthrough env; global default stays 40) |
| tool output | 8192 bytes | read from the profile by `codex_harness_agent` (`MAX_TOOL_OUTPUT_BYTES`); profiles without the field keep 32768 |

Keep the 20-30 minute episode deadline (`OPENENV_MAX_ROLLOUT_TIME_SECONDS`,
default 1800s) and record `max_seq_len_hit`, `end_reason`, turns, model
tokens, tool calls, and tool-output bytes per episode for the baseline
analysis. If a flag is reached only after truncation, enable the registered compaction plan: 24K active window, 4K–8K reserve, and at most 2–3 compactions while preserving segment identity.

## Baseline runner (5.6): RLF 形态的 Qwen3-30B-A3B

The runner is `yeto.rl.eval.rlf_baseline` (RLF counterpart of
`yeto/rl/eval/codex_infer.py`, which covers the TB2 attempt shape). Per run:
admit the task packages into the RLF catalog, render prompt rows, bring up
serving, then per row acquire one RLF CTF lease (W4 `RlfModalProvider`),
drive the training Codex subprocess worker, finish on the trusted side, and
write the trace. Codex-only; AgentFlow artifacts are refused at admission.

### Serving shape (inside the GPU container)

- Modal `A100-80GB` x1, image = the pinned Miles-ports image
  (`MILES_NEXT_IMAGE`), model read from the `yeto-fn-models` Volume:
  `/mnt/yeto-models/hf/Qwen3-30B-A3B/ad44e777` (bf16).
- `python -m sglang.launch_server --model-path /mnt/yeto-models/hf/Qwen3-30B-A3B/ad44e777 \
   --trust-remote-code --context-length 32768 --mem-fraction-static 0.9`
  plus the Miles session server (`session_server_config`, LoRA-free). The
  profile's parsers/chat template come from Miles' `resolve_serving("qwen3")`.
- Serving that is not healthy within 15 min aborts the whole run and the
  summary is written with `aborted.classification = "infrastructure"`
  (never scored). HARD per-task deadline: 30 min
  (`--episode-seconds`, default 1800; a tighter lease deadline wins).

### Run

Driver side (needs a local modal client):

    export YETO_CODEX_BUNDLE_DIR=/path/to/codex-bundle        # scripts/fetch_codex_bundle.py
    export YETO_RLF_CTF_TASKS_DIR=/path/to/tasks              # admitted packages (task_id subdirs)
    export YETO_RLF_BASELINE_ENVS_JSON=/path/to/envs.json     # see below
    modal run yeto.cloud.modal_rlf_baseline --config-json run.json

`envs.json` holds the container env, same inputs as the eval island smoke:
the Codex contract env (`yeto.launcher.codex_container_env(
codex_bundle_contract(...))`), `TBENCH_REWARD_HMAC_KEY`, the sandbox-only
`MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` (the RLF provider opens one VM per
episode from inside the container), and `YETO_RLF_CTF_FLAGS_JSON`
(`{"task_id": "flag{...}"}` — trusted; lives only in the Modal Secret,
never in the image, the traces, or the logs). `run.json`:

    {"run_name": "qwen3-30b-baseline-01", "repeat": 1,
     "evidence": {"runtime_quality_gate": {"healthy": true}},
     "baseline": {"episode_seconds": 1800}}

Evidence lands in the `yeto-rlf-baseline-runs` Volume under the run name.
In-container / CPU-smoke entry without Modal:

    python -m yeto.rl.eval.rlf_baseline --tasks-dir /root/rlf-tasks \
        --flags-json @flags.json --evidence '{"runtime_quality_gate": {"healthy": true}}' \
        --out /tmp/rlf-baseline-out [--repeat N] [--task ID] [--serve external --session-url URL]

### Evidence checklist (per run)

- `traces.jsonl`: one record per episode — RLF identity quadruple,
  `policy_id` (`Qwen/Qwen3-30B-A3B@ad44e777...`), the task prompt, per-turn
  completion/context tokens and tool-output bytes, per-tool events
  (scrubbed command, bounded scrubbed output preview, `flag_seen` marker),
  submission markers (`after_tool_calls`, `has_flag`, `flag_correct`; the
  submitted flag value is never recorded), `submit_calls`,
  `max_model_total_tokens`, `max_seq_len_hit`, compaction count,
  `end_reason`/`end_kind`, `grader_evidence`, reward/success and the
  `classification` (`success` / `failed` / `infrastructure` / `rejected`).
- `summary.json`: success rate over evaluated episodes (infrastructure and
  rejected episodes are excluded, never scored 0), reward distribution,
  end-reason counts, `max_seq_len_hit` rate, total compactions,
  turns/tool-calls/token means and maxima, per-task breakdown.

Stop conditions: serving start timeout (infrastructure abort), the 30-min
per-task deadline (`timeout` episodes keep reward 0 via the signed outcome
envelope and are never graded), or every prompt row processed. A
`max_seq_len_hit` rate above zero is the evidence for enabling the
registered 24K/4K compaction plan (5.7).

Known unverified (first GPU run must confirm): Miles `resolve_serving`
coverage of the `qwen3` tito family, LoRA-free `SessionServerConfig`
(`lora_rank=None`), and A100-80GB scheduling latency on Modal.


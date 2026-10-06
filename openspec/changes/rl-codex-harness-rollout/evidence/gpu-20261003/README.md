# GPU evidence 2026-10-03 (T3: 9.1 / 9.2 smoke, Modal)

Runs `codex-smoke-20261003-{2..12}`; local run dirs `/home/michael/work/gpu-b1-runs/codex-smoke-20261003-N/` (launch.log, pulled/rl-island-0.jsonl).

- `codex-smoke-5-lora-layout-mismatch.txt`: A-T3-3 forensics (Megatron CanonicalLoRA export vs PEFT prediction; fixture `tests/fixtures/qwen35_08b_lora_export_smoke5.json`).
- `codex-smoke-12-local-rounds.jsonl`: the two `rl_local_round` records (step 1 = 9.1, step 2 = 9.2) from the passing run (1xH100, code 56c34191).
- `codex-smoke-12-rl-island-0.jsonl`: island tape (24 records, `rl_learner_finalized`).
- `codex-smoke-12-launcher-events.jsonl`: launcher event tape.

Root causes fixed on the way (all yeto-side, Miles pin e3a11ab3 and image 12fcd9e5 unchanged):
A-T3-3 0581bb95 (gated q/k/v LoRA), A-T3-4 bb5bd832 (rollout-worker configure), A-T3-5 d74665cb/1b97b570/b94c4ddb (Modal client in island + provider-outage fail-fast),
A-T3-6 56c34191 (tito_session_mismatch list), A-T3-7 9cc2b621 (ordered tool-wait board calls; landed after -12, CPU-tested only).
A-T3-8: `--attention-backend flash` (qwen3_5 recipe) uses flash_attn cute (FA4) kernels that fail on L40S/Ada -> training needs H100 (hardware, not code).

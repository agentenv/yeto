# Corrected source-turn SFT

Codex can record an announcement and its tool call as separate events in one
assistant turn. Rendering each event as a finished Qwen message taught an end
token before the action. This package groups continuing assistant events before
the pinned Qwen renderer emits the assistant message and its final end token.

The boundary adapter preserves explicit final-channel, source-turn and role
boundaries. Where source turn identifiers are absent, the named contiguous
assistant fallback is recorded in the audit. It never reorders text around tool
calls. Native formats that cannot represent the original order are excluded
with a reason. Empty assistant messages do not become end-token-only targets.

## Active training paths

- No-CoT baseline: `training.qwen38_turn_boundary_v2.prepare_data`, `.data`,
  `.recipe`, and `.train`. It uses non-thinking mode.
- Masked-CoT: `training.qwen38_native_gap_v3.prepare_masked_full`, `.masked_data`,
  `.masked_recipe`, and `.masked_train`. It uses native xhigh formatting, with
  generated reasoning as input and loss zero. Original source/generation
  receipts remain unchanged.

The masked path keeps a compatible leading CoT in its original position. If a
later CoT would have to move ahead of earlier visible content, only that CoT is
omitted; the trace and its assistant targets are retained. A trace can remain
in this arm with no retained CoT. Per-gap omissions are accounted separately
from whole-source format/quality exclusions and the token cutoff. Multiple
inline reasoning blocks are not used.

The assistant/tool serialization follows the pinned Qwen template. The existing
chronological system-message adaptation remains in the renderer. Original
source reasoning is removed, unsupported capability declarations are omitted,
and reserved control-token spellings are escaped as data. Secrets are not
redacted. The template, tokenizer and adapters are hash-bound in each export.

## Training contract

Both runs start from the pinned original Qwen3.8-27B base with a fresh optimizer.
They retain the qualified one-host eight-GPU CP8/FSDP2 setup, learning rate
`1e-5`, full text-parameter SFT, whole-block activation checkpointing, fused CE,
disabled prefetch, and expandable CUDA allocations. The first 262,144 tokens
are retained; later tokens are discarded without appending a synthetic end.

System/user/tool observations, assistant headers, and all thinking wrappers and
CoT have loss zero. Visible assistant text, tool calls and the true end token
have loss one. Stored labels are unshifted; the exact loader shifts labels once
before context parallelism, because the pinned fused loss does not shift them.

Full-corpus qualification checks every row, native mask, shard/index hash,
retained-array duplicate, and original session split. The actual training image
then checks the loader and collator. Checkpoints include model, optimizer,
loader, RNG, scheduler and a bound resume contract; old converted-data
checkpoints cannot silently resume these runs.

## Validation and quality measurements

Run the CPU regression suites from the repository root with the pinned tokenizer
assets available:

```sh
python -m pytest training/qwen38_turn_boundary_v2 training/qwen38_native_gap_v3
python -m pytest evaluation/qwen38_training_quality
```

The training runtime uses the pinned NeMo image documented by the run receipts;
the general repository Python environment is not a substitute for it. CPU
tokenizer tests and actual-image runtime checks are distinct qualifications.

`evaluation/qwen38_training_quality` supplies a fixed 16-task action suite and
checkpoint-specific export/parity/outcome tooling. Initial checkpoints are 25,
50 and 100 completed updates. Compare action success, premature stops,
infrastructure failures, held-out loss and supervised-token budgets. Training
loss alone does not establish behavioral improvement. The earlier corrected
step-299 evaluation scored 5/89; the turn-boundary defect is a strong process
finding, not a proven causal explanation of every failure.

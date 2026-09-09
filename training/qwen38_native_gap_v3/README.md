# Official-native Qwen CoT input with per-gap omission

This is a separate training contract, `qwen38-xhigh-native-gap-cot-loss-zero/v3`. It uses the pinned official Qwen3.8 template with `enable_thinking=true`, `preserve_thinking=true`, and `reasoning_effort=xhigh`.

Consecutive assistant continuation events become one native assistant message, retaining the original visible text and tool-call order and one true end-of-turn target. Generated reasoning stays in its original native leading position. If a later reasoning block would have to move ahead of an earlier visible action or combine with another reasoning block, only that generated block is omitted. Its source trace remains eligible, including when no generated reasoning remains. Native text-after-tool ordering that cannot be represented faithfully remains a separate whole-source exclusion.

Each row binds the original generation receipts and partitions them into retained and omitted entries. Omitted entries name their original event and message index and the native placement conflict. Neither the original source nor the generation journal is changed. These existing generations are unreviewed: the contract does not claim semantic approval or a future-information check.

The full export reconciles all frozen generations as retained before truncation, individually omitted, or belonging to a separately excluded source. `token_counts.retained_filled_gaps` separately counts reasoning blocks that survive the first-262144-token cutoff. All native reasoning wrappers and bodies have loss 0, as do headers, system/user text and tool observations. Assistant text, serialized tool calls and true end-of-turn tokens have loss 1. The exact loader applies the causal label shift once and retains the complete input prefix.

The selected source quality/confidence filters, exact baseline replay representatives, native source guards and deduplication are preserved. Known session groups use the original baseline split. Captures with unresolved session lineage are train-only, without a complete heldout-isolation claim. A fixed bounded subset of known heldout rows supplies validation loss.

`prepare_masked_full` produces immutable shards and a full row/mask index. `masked_train` delegates the unchanged qualified CP8/FSDP2 full-parameter training lifecycle under a distinct identity, fresh base and output directory. This package does not use the canceled custom inline-reasoning renderer.

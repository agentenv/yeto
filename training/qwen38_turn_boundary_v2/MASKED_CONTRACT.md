# Corrected masked-CoT experiment

The immutable generation snapshot remains the source of each candidate and receipt. Generated reasoning stays at its original gap and is context only (loss zero). No source is relabelled reviewed or checked for future information.

Contiguous assistant text followed by tool calls is coalesced before native Qwen rendering, respecting explicit source-turn, final, user and observation barriers. Empty assistant events preserve barriers but do not contribute an EOT-only target. If a later filled gap would move reasoning before earlier visible content, the whole source is excluded with its exact generated-gap count. Multiple reasoning blocks or text after a tool call in one continuation are also explicit whole-source exclusions.

Every saved row uses pinned Qwen3.8 native xhigh rendering, first262144-token cutoff, and unshifted labels. The loader independently reconstructs body/action/EOT supervision from token IDs; headers, think wrappers, generated reasoning and all nonassistant messages are masked. It then applies the existing one-position label shift exactly once, retaining all input IDs.

The builder reconciles every frozen generation as included or explicitly excluded, preserves original replay representatives, and removes duplicate normalized sequences or retained training arrays. Known sessions use the original baseline hash split; unresolved captures stay train-only and are never presented as verified independent sessions. The immutable index checks every row, duplicate identity and cross-split overlap. Validation uses the same manifest/index, bounded to the first32 eligible heldout rows of at most32768 tokens. This is not a full-corpus quality score.

The new runner starts from the pinned original model with fresh optimizer state, preserving the proven CP8/FSDP2, full262144 context, whole-block activation checkpointing, fused loss, no prefetch, CPU-offloaded synchronous checkpoints and LR1e-5 settings. Original code, datasets and checkpoints remain unchanged. Behavioral recovery must be measured by subsequent actual tool-use evaluations.

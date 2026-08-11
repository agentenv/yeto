# DeepSeek V4 Balanced Expert Mapping Design

## Goal

Distribute DeepSeek V4's 32 cloned routed experts evenly across the existing EP8 trainer topology, while preserving the checkpoint, canonical policy, and SGLang logical expert IDs.

## Layout Contract

External interfaces retain logical IDs: originals are `0..255` and clones are `256..287`. Megatron training alone uses a physical permutation with 36 contiguous slots per EP rank:

- physical offsets `0..31` contain that rank's 32 originals;
- physical offsets `32..35` contain that rank's four clones.

For logical original `e`, physical ID is `(e // 32) * 36 + e % 32`. For clone `256 + k`, physical ID is `(k // 4) * 36 + 32 + k % 4`. The inverse is defined for every ID in `0..287`, making the layout a bijection.

## Data Paths

The Megatron router expands the 256-way logical gate into the physical 288-way dispatch map. SGLang's compact top-k remap remains logical.

Megatron-Bridge translates physical expert names back to logical names when loading HF checkpoint weights and exporting base or adapter weights. Merged-LoRA export temporarily restores physical names while selecting packed adapter slices, then emits logical names.

Clone-only grouped LoRA activates the final four physical slots on every EP rank and records their logical clone IDs. A Yeto runtime hook replaces the two pinned Miles helpers that apply sparse canonical clone tensors and validate frozen original optimizer masters, so each rank reads its own four logical clones and writes physical offsets `32..35`.

Expert-full tuning maps every local physical parameter slot to its logical ID, enabling only the selected logical clone prefix. Its local canonical-state path translates Bridge task names to logical IDs before ownership filtering and matching. Miles base-weight sync instead selects the same local physical clone offsets on every EP rank, lets Bridge complete its EP collectives, and then filters the restored logical output names to the selected prefix. Existing PP stage-local layer mapping remains unchanged.

## Alternatives Rejected

Changing only the trainable mask would route logical experts to the wrong weights. Changing Megatron's dispatcher ownership would be substantially more invasive. Renumbering the checkpoint or canonical policy would break the existing SGLang and policy-sync contracts.

## Failure Handling

Mapping helpers reject IDs outside `0..287`. EP-dependent clone-only and expert-full paths require the established EP8 geometry and 36 local slots. Every rank must execute an identical expert-full Bridge collective sequence. Bridge remapping is enabled only for a fully attested expanded clone config, leaving ordinary 256-expert DeepSeek V4 unchanged and rejecting malformed non-empty contracts. Runtime installers are idempotent and fail closed on unsupported packed shapes or incomplete expert mappings.

## Verification

Tests cover the 288-ID bijection, per-rank ownership, dense trainer routing versus logical SGLang routing, Bridge load/export/adapter naming and contract validation, clone-only masks and Miles sparse apply on multiple ranks, expert-full ownership across all eight ranks, identical EP8 collective task topology, post-gather logical output filtering, and PP stage-local expert views. Related DeepSeek V4 regressions, runtime scripts, compilation, and whitespace checks run before synchronization.

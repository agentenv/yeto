# DeepSeek V4 Balanced Expert Mapping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every DeepSeek V4 EP8 rank 32 original and four cloned experts without changing external logical IDs.

**Architecture:** Introduce one audited logical/physical ID permutation and reuse it at every trainer boundary. Megatron routes and stores physical IDs; Bridge, Miles policy sync, canonical expert-full state, and SGLang translate to or retain logical IDs.

**Tech Stack:** Python 3, PyTorch, Megatron-Core, Megatron-Bridge, Miles, pytest

---

### Task 1: Define and exercise the balanced layout

**Files:**
- Modify: `tests/test_rl_deepseek_v4_expert_clone.py`
- Modify: `yeto/rl/deepseek_v4_expert_clone.py`

- [ ] Add tests proving all 288 IDs round-trip and every EP8 rank owns 32 originals plus four clones.
- [ ] Update route tests to compare Megatron's physical dense map with SGLang's logical compact IDs.
- [ ] Run the new tests and verify they fail because the mapping API/layout is absent.
- [ ] Add the two integer mapping functions, expert-name mapping helpers, and physical route expansion.
- [ ] Re-run the focused tests and verify they pass.

### Task 2: Preserve logical checkpoint and Bridge policy names

**Files:**
- Modify: `tests/test_rl_deepseek_v4_bridge.py`
- Modify: `yeto/rl/deepseek_v4_bridge.py`

- [ ] Add failing pure helper tests for physical-to-logical checkpoint load names, exported dictionary keys, adapter base names, and logical-to-physical merged-LoRA selection names.
- [ ] Gate the permutation on the attested E288 clone config.
- [ ] Translate names in `maybe_modify_loaded_hf_weight`, `maybe_modify_converted_hf_weight`, `_get_base_hf_param_names_for_adapter`, and `_merge_lora_adapter_weights`.
- [ ] Re-run Bridge tests and verify they pass without importing training-only dependencies.

### Task 3: Balance clone-only LoRA and pinned Miles policy apply

**Files:**
- Modify: `tests/test_rl_deepseek_v4_clone_lora.py`
- Modify: `tests/test_sitecustomize_diagnostics.py`
- Modify: `yeto/rl/deepseek_v4_clone_lora.py`
- Modify: `sitecustomize.py`

- [ ] Add failing rank-parameterized mask tests: each EP8 rank activates physical offsets `32..35` and owns four logical clones.
- [ ] Add fake-Miles tests proving rank 0 applies logical clones `256..259`, rank 7 applies `284..287`, and every rank checks the first 32 optimizer-master slots.
- [ ] Implement balanced masks plus an idempotent import-time patch for Miles `_sparse_expert_updates` and `_assert_original_packed_masters_zero`.
- [ ] Activate the hook only for `YETO_DSV4_CLONE_ONLY_LORA=1` in `sitecustomize.py`.
- [ ] Re-run clone-LoRA and sitecustomize tests and verify they pass.

### Task 4: Balance expert-full ownership and runtime conversion tasks

**Files:**
- Modify: `tests/test_rl_deepseek_v4_expert_full.py`
- Modify: `tests/test_rl_deepseek_v4_expert_full_runtime.py`
- Modify: `yeto/rl/deepseek_v4_expert_full.py`
- Modify: `yeto/rl/deepseek_v4_expert_full_runtime.py`

- [ ] Add failing tests proving all eight ranks collectively own every clone once and each owns four clones for a 32-expert policy.
- [ ] Add failing task/view tests showing physical expert 32 represents logical clone 256 while physical 256 is an original.
- [ ] Add a failing EP8 task-topology test proving every rank executes the same local clone offsets before Bridge collectives, then filters gathered logical names to the selected prefix.
- [ ] Derive local logical IDs from physical slots in expert configuration.
- [ ] Translate conversion-task expert names to logical IDs for local ownership and canonical matching; keep the collective filter based on identical physical offsets.
- [ ] Re-run expert-full tests, including the PP stage-local case, and verify they pass.

### Task 5: Update runtime validation and complete regression checks

**Files:**
- Modify: `scripts/validate_deepseek_v4_bridge_runtime.py`
- Modify: `scripts/validate_deepseek_v4_clone_lora_runtime.py`

- [ ] Make router validation address physical source/clone slots while reporting logical semantics.
- [ ] Make clone-LoRA runtime validation require the sparse logical clone set `256..287`.
- [ ] Run all DeepSeek V4 unit suites and relevant sitecustomize/export regressions.
- [ ] Run `python -m compileall -q yeto scripts`, `git diff --check`, and inspect the full diff.
- [ ] Fetch the remote branch, integrate only safe concurrent updates if present, commit as `AlexEisie <1987460907@qq.com>` with SSH signing, push `feat/secrlenv-infra-abort-v28`, and verify local/remote revisions match.

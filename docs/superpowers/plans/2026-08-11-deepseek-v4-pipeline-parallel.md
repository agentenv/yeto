# DeepSeek V4 Pipeline-Parallel Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the pinned DeepSeek V4 E288 clone-LoRA and expert-full RL recipes accept and correctly handle pipeline parallelism greater than one.

**Architecture:** Keep Yeto's global canonical policy and Miles' collective TP/PP conversion unchanged. Make only the DeepSeek V4 validation and custom expert-full runtime pipeline-local: remote Bridge tasks stay in the global contract, while local parameters are discovered and updated through their conversion tasks.

**Tech Stack:** Python 3, PyTorch, Megatron-Core, Megatron-Bridge, Miles, pytest

---

### Task 1: Add failing PP2 contract tests

**Files:**
- Modify: `tests/test_rl_launcher.py`
- Modify: `tests/test_rl_ssh_harness.py`
- Modify: `tests/test_rl_deepseek_v4_expert_full.py`
- Modify: `tests/test_rl_deepseek_v4_expert_full_runtime.py`

- [x] **Step 1: Make the public DeepSeek V4 launcher cases request PP2**

Change the two-node clone-LoRA and expert-full cases to pass `--pipeline-parallel 2` and assert that value. In the Miles argv test, retain `pipeline_parallel=2` when constructing both DeepSeek V4 recipe namespaces and assert the uneven 43-layer split is `22/21`.

- [x] **Step 2: Make the SSH expert-full plan request PP2**

Set `plan["learner"]["pipeline_parallel"] = 2`, keep the two eight-GPU hosts, and assert the forwarded learner argument remains two.

- [x] **Step 3: Add a pipeline-local expert configuration test**

Allow the synthetic individual model helper to construct a selected number of layers, then add:

```python
def test_pipeline_stage_validates_only_its_local_expert_layers():
    model = _IndividualModel(num_layers=22)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    records = configure_clone_expert_full(
        model,
        expert_count=32,
        expert_parallel_rank=7,
        expert_parallel_size=8,
    )
    assert len(records) == 22 * 2 * 36
```

- [x] **Step 4: Add remote attention and global expert-layer mapping tests**

Construct synthetic Bridge tasks where one attention side has a local parameter and another has `param_weight=None`; `_attention_sides()` must retain both. Construct a local fused expert parameter named as stage-local layer zero with a conversion task mapped to HF layer 22; `_expert_views()` must expose layer 22 gate/up views.

- [x] **Step 5: Run the new tests and verify RED**

Run:

```bash
python -m pytest -q \
  tests/test_rl_launcher.py::test_rl_accepts_two_node_deepseek_model_parallel_island \
  tests/test_rl_launcher.py::test_rl_accepts_attested_sixteen_expert_full_deepseek_recipe \
  tests/test_rl_launcher.py::test_miles_argv_uses_provider_capabilities_without_model_family_branches \
  tests/test_rl_ssh_harness.py::test_expert_full_plan_forwards_attestation_and_runtime_environment \
  tests/test_rl_deepseek_v4_expert_full.py::test_pipeline_stage_validates_only_its_local_expert_layers \
  tests/test_rl_deepseek_v4_expert_full_runtime.py
```

Expected: failures from the PP1 guards, the fixed 43-layer assertion, rejection of a remote attention parameter, and stage-local layer zero not matching canonical layer 22.

### Task 2: Implement pipeline-local DeepSeek V4 behavior

**Files:**
- Modify: `yeto/launcher.py`
- Modify: `yeto/rl/learner.py`
- Modify: `yeto/rl/ssh_harness.py`
- Modify: `yeto/rl/deepseek_v4_expert_full.py`
- Modify: `yeto/rl/deepseek_v4_expert_full_runtime.py`

- [x] **Step 1: Remove only the recipe-specific PP1 checks**

Keep TP8, EP8, eight-GPU rollout-engine, SGLang TP8/EP8, and global `TP*PP` divisibility validation. Remove `pipeline_parallel != 1` from the three DeepSeek V4 guard expressions and update their messages to describe TP8/EP8 pipeline stages and per-node rollout replicas.

- [x] **Step 2: Validate the expert grid for observed local layers**

Replace the fixed `NUM_LAYERS` record count and required-key construction with the nonempty set of layers parsed from local parameter names:

```python
local_layers = {record.layer for record in records}
expected = len(local_layers) * 2 * local_count
required = {
    (layer, branch, expert)
    for layer in local_layers
    for branch in ("linear_fc1", "linear_fc2")
    for expert in local_ids
}
```

Raise a fail-closed error when no individual expert layers exist locally.

- [x] **Step 3: Retain remote attention conversion sides**

In `_attention_sides()`, no longer reject `side.param_weight is None`. Continue requiring a complete, duplicate-free global canonical mapping. In `apply_hybrid_trainable_state()`, every rank still enters every tensor broadcast, but conversion and copy occur only when the side owns a local parameter.

- [x] **Step 4: Build expert views from global Bridge conversion tasks**

Use `filter_selected_expert_tasks(_actor_bridge(actor).get_conversion_tasks(actor.model), expert_count=...)`. Skip remote tasks, require every local task parameter to carry `_yeto_expert_full`, parse each task's HF names with `_EXPERT_WEIGHT`, split fused FC1 parameters for gate/up, and retain FC2 as down. Reject duplicates, out-of-contract names, and any local trainable expert parameter not covered by a task.

- [x] **Step 5: Run the RED selection and verify GREEN**

Run the Task 1 command. Expected: every selected test passes.

### Task 3: Verify, review, commit, and synchronize

**Files:**
- Verify all files changed in Tasks 1 and 2

- [x] **Step 1: Run related regression suites**

Run:

```bash
python -m pytest -q \
  tests/test_rl_deepseek_v4_expert_full.py \
  tests/test_rl_deepseek_v4_expert_full_runtime.py \
  tests/test_rl_launcher.py \
  tests/test_rl_ssh_harness.py \
  tests/test_rl_core.py \
  tests/test_rl_integration.py
```

Expected: zero failures.

- [x] **Step 2: Run static and diff checks**

Run:

```bash
python -m compileall -q yeto/rl
git diff --check
git status --short
```

Expected: compilation and whitespace checks exit zero; status lists only the intended implementation and test files plus this plan.

- [x] **Step 3: Review the exact patch**

Run `git diff --stat` and `git diff`. Confirm every changed line implements a stated PP2 requirement and no Miles pin, policy schema, checkpoint schema, or EP mapping changed.

- [ ] **Step 4: Commit with the configured signed identity**

Run:

```bash
git add docs/superpowers/plans/2026-08-11-deepseek-v4-pipeline-parallel.md \
  yeto/launcher.py yeto/rl/learner.py yeto/rl/ssh_harness.py \
  yeto/rl/deepseek_v4_expert_full.py \
  yeto/rl/deepseek_v4_expert_full_runtime.py \
  tests/test_rl_launcher.py tests/test_rl_ssh_harness.py \
  tests/test_rl_deepseek_v4_expert_full.py \
  tests/test_rl_deepseek_v4_expert_full_runtime.py
git commit -m "feat(rl): support DeepSeek V4 pipeline parallelism"
```

Expected: a signed commit authored by `AlexEisie <1987460907@qq.com>`.

- [ ] **Step 5: Push and verify synchronization**

Run:

```bash
git push origin feat/secrlenv-infra-abort-v28
git rev-parse HEAD
git rev-parse origin/feat/secrlenv-infra-abort-v28
```

Expected: both revisions are identical.

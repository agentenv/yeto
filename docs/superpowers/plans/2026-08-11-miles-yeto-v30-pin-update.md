# Miles `yeto-v30` Pin Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish Miles `ed9f8c36d89d560c9cd05820225f044cfef074f7` on `yeto-v30` and update Yeto `feat/secrlenv-infra-abort-v28` to pin its exact attested bundle.

**Architecture:** The Miles update is a guarded fast-forward of the existing `yeto-v30` branch. Yeto continues its existing vendored-bundle design: retain old bundles, add one immutable bundle, and update the three pin constants plus their explicit regression assertions.

**Tech Stack:** Git, Git bundle, SHA-256, Python 3, pytest

---

### Task 1: Publish the Existing Miles Commit

**Files:**
- No file changes; operate on `/root/.cache/yeto-rl-ssh/dsv4-e288-safety32-lora-r64-smoke-v41/miles` on `h200-n1`

- [ ] **Step 1: Verify the exact clean source commit**

Run:

```bash
MILES_REPO=/root/.cache/yeto-rl-ssh/dsv4-e288-safety32-lora-r64-smoke-v41/miles
git -C "$MILES_REPO" rev-parse HEAD
git -C "$MILES_REPO" status --porcelain --untracked-files=no
```

Expected: HEAD is `ed9f8c36d89d560c9cd05820225f044cfef074f7` and status is empty.

- [ ] **Step 2: Verify the live branch and fast-forward relation**

Run:

```bash
MILES_REPO=/root/.cache/yeto-rl-ssh/dsv4-e288-safety32-lora-r64-smoke-v41/miles
git -C "$MILES_REPO" ls-remote --heads origin refs/heads/yeto-v30
git -C "$MILES_REPO" merge-base --is-ancestor \
  fa47aec08c54fe6b69b6f302dc6deef1b0091fe5 \
  ed9f8c36d89d560c9cd05820225f044cfef074f7
```

Expected: the live branch is `fa47aec08c54fe6b69b6f302dc6deef1b0091fe5` and the ancestry command exits 0.

- [ ] **Step 3: Fast-forward `yeto-v30` without force**

Run:

```bash
MILES_REPO=/root/.cache/yeto-rl-ssh/dsv4-e288-safety32-lora-r64-smoke-v41/miles
git -C "$MILES_REPO" push origin \
  ed9f8c36d89d560c9cd05820225f044cfef074f7:refs/heads/yeto-v30
```

Expected: Git reports `fa47aec0..ed9f8c36  -> yeto-v30`.

- [ ] **Step 4: Verify the published ref**

Run:

```bash
MILES_REPO=/root/.cache/yeto-rl-ssh/dsv4-e288-safety32-lora-r64-smoke-v41/miles
git -C "$MILES_REPO" ls-remote --heads origin refs/heads/yeto-v30
```

Expected: `ed9f8c36d89d560c9cd05820225f044cfef074f7`.

### Task 2: Add a Failing Yeto Pin Regression

**Files:**
- Modify: `tests/test_rl_ssh_harness.py:506-512`

- [ ] **Step 1: Change the explicit Miles pin expectations**

Set the assertions to:

```python
assert MILES_COMMIT == "ed9f8c36d89d560c9cd05820225f044cfef074f7"
assert MILES_BUNDLE_SHA256 == (
    "4e2e86d5e144633a6cde95a5d0aa999fe0c7a9e3b4ff4521879444529c487d0d"
)
```

- [ ] **Step 2: Run the regression and confirm it fails for the old pin**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider \
  tests/test_rl_ssh_harness.py::test_miles_and_sglang_pins_include_the_compatible_builds
```

Expected: FAIL because `MILES_COMMIT` is still `fa47aec08c54fe6b69b6f302dc6deef1b0091fe5`.

### Task 3: Vendor and Activate the New Miles Bundle

**Files:**
- Create: `yeto/rl/vendor/miles-ed9f8c36d89d560c9cd05820225f044cfef074f7.bundle`
- Modify: `yeto/rl/__init__.py:4-9`
- Test: `tests/test_rl_ssh_harness.py`
- Test: `tests/test_rl_launcher.py`

- [ ] **Step 1: Verify the n1 source bundle against the Miles repository**

Run on n1:

```bash
MILES_REPO=/root/.cache/yeto-rl-ssh/dsv4-e288-safety32-lora-r64-smoke-v41/miles
SOURCE_BUNDLE=/root/.cache/yeto-rl-ssh/dsv4-e288-safety32-lora-r64-smoke-v41/source/yeto/rl/vendor/miles-ed9f8c36d89d560c9cd05820225f044cfef074f7.bundle
sha256sum "$SOURCE_BUNDLE"
git -C "$MILES_REPO" bundle verify "$SOURCE_BUNDLE"
git bundle list-heads "$SOURCE_BUNDLE"
```

Expected: SHA256 is `4e2e86d5e144633a6cde95a5d0aa999fe0c7a9e3b4ff4521879444529c487d0d`, verification succeeds, and HEAD is `ed9f8c36d89d560c9cd05820225f044cfef074f7`.

- [ ] **Step 2: Copy and reverify the immutable bundle**

Run from the Yeto repository:

```bash
SOURCE_BUNDLE=/root/.cache/yeto-rl-ssh/dsv4-e288-safety32-lora-r64-smoke-v41/source/yeto/rl/vendor/miles-ed9f8c36d89d560c9cd05820225f044cfef074f7.bundle
scp "root@h200-n1:$SOURCE_BUNDLE" \
  yeto/rl/vendor/miles-ed9f8c36d89d560c9cd05820225f044cfef074f7.bundle
sha256sum yeto/rl/vendor/miles-ed9f8c36d89d560c9cd05820225f044cfef074f7.bundle
git bundle list-heads \
  yeto/rl/vendor/miles-ed9f8c36d89d560c9cd05820225f044cfef074f7.bundle
```

Expected: copied SHA256 and HEAD match Step 1.

- [ ] **Step 3: Update the three Yeto pin constants**

Set `yeto/rl/__init__.py` to:

```python
MILES_COMMIT = "ed9f8c36d89d560c9cd05820225f044cfef074f7"
MILES_BUNDLE_PATH = (
    "yeto/rl/vendor/miles-ed9f8c36d89d560c9cd05820225f044cfef074f7.bundle"
)
MILES_BUNDLE_SHA256 = "4e2e86d5e144633a6cde95a5d0aa999fe0c7a9e3b4ff4521879444529c487d0d"
```

Leave `MILES_UPSTREAM_COMMIT`, `MILES_PEFT_VERSION`, `MILES_IMAGE`, SGLang pins, and old bundle files unchanged.

- [ ] **Step 4: Run the targeted regression tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider \
  tests/test_rl_ssh_harness.py::test_miles_and_sglang_pins_include_the_compatible_builds \
  tests/test_rl_ssh_harness.py::test_plan_digest_and_current_miles_pin_are_validated \
  tests/test_rl_launcher.py::test_miles_task_checks_out_exact_commit_and_builds_multinode_ray \
  tests/test_rl_launcher.py::test_miles_runtime_requires_exact_detached_clean_checkout
```

Expected: four tests pass.

- [ ] **Step 5: Run the complete affected test modules**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider \
  tests/test_rl_ssh_harness.py tests/test_rl_launcher.py
```

Expected: all tests pass with zero failures.

### Task 4: Commit and Publish the Yeto Pin

**Files:**
- Create: `yeto/rl/vendor/miles-ed9f8c36d89d560c9cd05820225f044cfef074f7.bundle`
- Modify: `yeto/rl/__init__.py`
- Modify: `tests/test_rl_ssh_harness.py`

- [ ] **Step 1: Review the exact implementation diff**

Run:

```bash
git status --short
git diff --check
git diff -- yeto/rl/__init__.py tests/test_rl_ssh_harness.py
git diff --stat
```

Expected: only the new bundle and the two pin files are uncommitted; documentation is already committed.

- [ ] **Step 2: Create the signed implementation commit**

Run:

```bash
git add yeto/rl/__init__.py tests/test_rl_ssh_harness.py \
  yeto/rl/vendor/miles-ed9f8c36d89d560c9cd05820225f044cfef074f7.bundle
git commit -S -m "fix(rl): pin latest Miles disk restore"
```

Expected: commit succeeds as `AlexEisie <1987460907@qq.com>` with a valid signature.

- [ ] **Step 3: Verify the branch can advance without rewriting history**

Run:

```bash
git fetch origin feat/secrlenv-infra-abort-v28
git merge-base --is-ancestor origin/feat/secrlenv-infra-abort-v28 HEAD
```

Expected: both commands exit 0.

- [ ] **Step 4: Push the Yeto feature branch without force**

Run:

```bash
git push origin HEAD:refs/heads/feat/secrlenv-infra-abort-v28
```

Expected: the remote feature branch advances to the signed implementation commit.

- [ ] **Step 5: Verify both published repositories and local cleanliness**

Run:

```bash
git status --porcelain
git ls-remote --heads origin refs/heads/feat/secrlenv-infra-abort-v28
ssh root@h200-n1 \
  'git -C /root/.cache/yeto-rl-ssh/dsv4-e288-safety32-lora-r64-smoke-v41/miles ls-remote --heads origin refs/heads/yeto-v30'
```

Expected: local status is empty, Yeto remote equals local HEAD, and Miles `yeto-v30` equals `ed9f8c36d89d560c9cd05820225f044cfef074f7`.

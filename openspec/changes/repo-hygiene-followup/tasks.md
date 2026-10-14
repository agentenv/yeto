# Tasks

Commit and PR text use English only. Before each PR merges, run the local safe test set (exclude `mismatch_tape` and `upstream_parse_args`). For the user's decisions, see "Decided" in proposal.md.

## 1. 0.4 and log removal (ordering requirement cancelled)

- [x] 1.1 On 2026-10-13 the user cancelled the requirement "log cleanup merges before #238" (checked: #238 adds no logs); the decision is recorded in proposal.md
- [x] 1.2 Log removal has its own PR #240 (https://github.com/agentenv/yeto/pull/240, branch `s23-move-run-logs`, based on `agentenv/main` c9d25b69)
- [ ] 1.3 After #240 merges: #239 drops these two commits; confirm that the #239 diff no longer contains the log removal changes
- [x] 1.4 Tell the main agent to check 0.4 in `rl-code-quality-cleanup/tasks.md` (this change does not edit that directory)

## 2. Commit language check (local hook only)

- [x] 2.1 Add the check script: read the commit message file, check subject and body, ban CJK character ranges and full-width punctuation, skip `#` comment lines
- [x] 2.2 Unit tests for the script: pure English passes; a Han character in the subject fails; full-width punctuation in the body fails; `→` and `é` pass
- [x] 2.3 Add a `commit-msg` stage hook to `.pre-commit-config.yaml`; write one line with the install command (`pre-commit install --hook-type commit-msg`) in README or CONTRIBUTING
- [x] 2.4 Install the hook in each local worktree; use a test commit message with Han characters to confirm it is rejected (no commit is created) (tested: the script rejects a message file with Han characters; the shared hooks directory had no commit-msg, now installed)

## 3. Fork sync and upstream

- [ ] 3.1 See the fork merge plan; not done in this change (user, 2026-10-13: all development goes into agentenv from now on; the personal fork content merges into agentenv)
- [x] 3.2 Leave the `upstream` remote as is for now; this change does not delete it
- [x] 3.3 Record in the P6 document: judge ahead/behind by `agentenv/main`

## 4. Fill .gitignore

- [x] 4.1 Run `git check-ignore -v` for each path listed by `git status --ignored`; record the source rule
- [x] 4.2 Add rules only for gaps (evidence work directories, common caches); do not change lines 82–91 (added only .mypy_cache/, .ruff_cache/, .ipynb_checkpoints/, .DS_Store; the evidence work directories have no gap)
- [x] 4.3 Confirm that `git ls-files -ci --exclude-standard` has no new hits (tracked files are not hit by the new rules)

## 5. Track Chinese openspec directories (Chinese first, translate to English before they enter the repo)

- [x] 5.1 After the user reviews the Chinese drafts, translate `rl-code-quality-cleanup`, `rl-verl-agentenv-codex`, and this change directory to English
- [x] 5.2 Spot-check the "unverified" kind of sentences in the translation; confirm the degree of certainty is unchanged
- [x] 5.3 Commit the English version with an English commit message; after it enters the repo, run `tests/test_rl_sync_mode_table.py` and confirm it no longer skips and passes (run in s23-t10: 3 passed, no skips)
- [x] 5.4 Run `openspec validate --strict` on the three directories

## 6. Evidence log location

- [x] 6.1 Keep evidence logs at `/home/michael/work/infra-drafts/evidence/`; do not upload them to object storage; do not add them to the repo

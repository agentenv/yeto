# Tasks

Commit and PR text use English only. Before each PR merges, run the local safe test set (excluding `mismatch_tape` and `upstream_parse_args`). For the user's decisions, see "Decided" in proposal.md.

## 1. 0.4 and moving logs out (the order requirement is cancelled)

- [x] 1.1 On 2026-10-13 the user cancelled the requirement "clean up logs before #238 merges" (checked: #238 added no logs); the decision is recorded in proposal.md
- [x] 1.2 Moving logs out has its own PR #240 (https://github.com/agentenv/yeto/pull/240, branch `s23-move-run-logs`, based on `agentenv/main` c9d25b69)
- [ ] 1.3 After #240 merges: #239 drops these two commits; confirm that the diff of #239 no longer contains the log-removal changes
- [x] 1.4 Tell the main agent to tick 0.4 in `rl-code-quality-cleanup/tasks.md` (this change does not edit that directory)

## 2. Commit language check (local hook only)

- [x] 2.1 Add a check script: read the commit message file, check subject and body, forbid CJK character ranges and fullwidth punctuation, and skip `#` comment lines
- [x] 2.2 Unit tests for the script: pure English passes; a subject with Chinese characters fails; a body with fullwidth punctuation fails; `→` and `é` pass
- [x] 2.3 Add a `commit-msg` stage hook to `.pre-commit-config.yaml`; write one line with the install command in README or CONTRIBUTING (`pre-commit install --hook-type commit-msg`)
- [x] 2.4 Install the hook in each local worktree, and use a test commit message with Chinese characters to confirm it is rejected (no commit is created) (checked: the script rejects a message file with Han characters; the shared hooks directory had no commit-msg, so it was installed)

## 3. Fork sync and upstream

- [ ] 3.1 See the fork merge plan; not executed in this change (user, 2026-10-13: from now on all development is in agentenv, and the content of the personal fork merges into agentenv)
- [x] 3.2 Leave the `upstream` remote alone for now; this change does not delete it
- [x] 3.3 Record in the P6 document: judge ahead and behind against `agentenv/main`

## 4. Complete .gitignore

- [x] 4.1 For each path listed by `git status --ignored`, run `git check-ignore -v` and record the source rule
- [x] 4.2 Add rules only for gaps (evidence work directories, common caches); do not change lines 82-91 (only .mypy_cache/, .ruff_cache/, .ipynb_checkpoints/, .DS_Store added; no gap for evidence work directories)
- [x] 4.3 Confirm that `git ls-files -ci --exclude-standard` has no new hits (tracked files are not hit by mistake by the new rules)

## 5. Chinese openspec directories enter the repo (Chinese first, translate to English before they enter)

- [x] 5.1 After the user finishes reviewing the Chinese draft, translate `rl-code-quality-cleanup`, `rl-verl-agentenv-codex`, and the directory of this change entirely into English
- [x] 5.2 Spot-check the "unverified" kind of sentences in the translation, and confirm the degree of certainty is unchanged
- [x] 5.3 Commit the English version with an English commit message; after they enter the repo, run `tests/test_rl_sync_mode_table.py` and confirm it no longer skips and passes (run in s23-t10: 3 passed, none skipped)
- [x] 5.4 Run `openspec validate --strict` on the three directories

## 6. Location of evidence logs

- [x] 6.1 Put evidence logs in `/home/michael/work/infra-drafts/evidence/`; do not upload them to object storage and do not commit them

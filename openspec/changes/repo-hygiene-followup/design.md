## Context

PM items 4 and 9 say: commits and PRs use English only; run logs stay out of the repo. Moving the logs out of the repo is already done by `rl-code-quality-cleanup` 0.1-0.3 (4559533b), and it has its own PR #240. This design covers the rest. For the numbers, see "Key numbers" in proposal.md. For the user's decisions on 2026-10-13, see "Decided" in proposal.md.

Current state:
- `.pre-commit-config.yaml` has only one local hook, `yeto-precommit-check` (size, secret, and email checks). It has no `commit-msg` stage.
- Remotes: `agentenv` (main repo), `origin` (fork michaellchung/yeto), `upstream` (same URL as agentenv). This change does not change any remote.

## Goals / Non-Goals

**Goals**
- A language rule for new commits, with a machine check on the local machine.
- `.gitignore` covers the artifacts that are currently ignored in the working tree but only by scattered rules.
- Decide how the two untracked openspec directories enter the repo (draft in Chinese first, translate to English before they enter the repo).

**Non-Goals**
- Do not rewrite history commits. Do not force-push a shared branch.
- Do not redo log detection, `git rm --cached`, or `test_no_tracked_logs.py`.
- Do not add a CI English check.
- Do not sync the fork. Do not touch the `upstream` remote (see the fork merge plan).
- Do not handle the existing red state of main CI.
- Do not upload evidence logs to object storage.

## Decisions

### D1 Constrain only the future, do not change history
Rewriting the 77 commits that contain Chinese would change the hashes of the 620 commits since 2026-10-05. The SHAs referenced in PR #238/#239/#240, in every worktree branch, and in documents would all become invalid. The only gain is a cleaner history. So history is not changed (decided by the user).

### D2 Check rule: forbid CJK character ranges
- Check target: subject and body.
- Rule: fail if any character in U+4E00-U+9FFF, U+3400-U+4DBF, the fullwidth punctuation U+3000-U+303F, or U+FF00-U+FFEF appears.
- Do not forbid other non-ASCII characters, to avoid false hits on arrows and accented names.
- Skip comment lines that start with `#` (the git editor template).
- The alternative "allow only ASCII" is rejected: names or symbols other than `Co-Authored-By` would be rejected by mistake.

### D3 Local hook only
- Add a hook with `stages: [commit-msg]` to `.pre-commit-config.yaml`. The hook calls the check script.
- Each person runs `pre-commit install --hook-type commit-msg` once. Without it, the hook has no effect. This is an inherent limit of a local hook. The user accepted it, and there is no CI fallback.
- PR titles and bodies are outside the machine check. Review covers them.

### D4 0.4 and moving logs out
- The user decided to cancel the order requirement "clean up logs before #238 merges". Checked: #238 added no logs.
- Moving the logs out has its own PR #240 (branch `s23-move-run-logs`, based on `agentenv/main` c9d25b69).
- After #240 merges, #239 drops these two commits (git drops them during rebase, or they are dropped by hand; not tested).
- This change no longer cherry-picks 4559533b to #238.

### D5 Complete `.gitignore`
`git status --ignored` shows that the artifacts below are currently ignored by the 0.2 rule or by other rules. But they have no matching explicit rule, or the rule scope is too wide. Check each one before adding rules:
- `.txt`, `.json`, `.sh` under `openspec/changes/*/run-logs/` (currently ignored by `**/run-logs/**`, which already covers them; only confirm this in tasks).
- `openspec/changes/*/evidence/**/work/` (work directories, currently ignored; the source rule is not checked one by one).
- Common caches: `.mypy_cache/`, `.ruff_cache/`, `.ipynb_checkpoints/`, `.DS_Store` (they do not currently appear in the working tree; this is precaution).
At execution, use `git check-ignore -v <path>` to confirm the source rule of each item, and add rules only for gaps. Do not change lines 82-91 written by 0.2.

### D6 Chinese openspec directories: draft in Chinese, translate to English before they enter the repo
- Drafting and the review by the user use Chinese.
- Before they enter the repo, translate `rl-code-quality-cleanup`, `rl-verl-agentenv-codex`, and the directory of this change entirely into English. Only the English version enters the repo.
- The translation MUST keep the original degree of certainty (an "unverified" statement stays "unverified" and is not changed to an affirmative sentence).
- After they enter the repo, `tests/test_rl_sync_mode_table.py` no longer skips. Run it once in the PR that adds them.

### D7 Location of evidence logs
Evidence logs go first to the local `/home/michael/work/infra-drafts/evidence/`. They are not uploaded to object storage and do not enter the repo.

## Risks / Trade-offs

- The local hook depends on each person installing it, and has no CI fallback. A Chinese commit may then enter a PR. Check by eye during PR review.
- Translating about 108KB of text has a cost, and the translation may lose the degree of certainty. After translation, the user or another agent spot-checks the "unverified" kind of sentences.
- Evidence logs exist only on the local machine. If the machine is damaged, they are lost. Backup relies on workspace-backup (this change does not handle it).

## Context

PM rules 4 and 9: commits and PRs use English only; run logs stay out of the repo. Log removal is done by `rl-code-quality-cleanup` 0.1–0.3 (4559533b) and has its own PR #240. This design covers the rest. For the numbers, see "Key numbers" in proposal.md. For the user's decisions of 2026-10-13, see "Decided" in proposal.md.

Current state:
- `.pre-commit-config.yaml` has one local hook, `yeto-precommit-check` (size, secret, and email checks). It has no `commit-msg` stage.
- Remotes: `agentenv` (main repo), `origin` (fork michaellchung/yeto), `upstream` (same URL as agentenv). This change does not change any remote.

## Goals / Non-Goals

**Goals**
- The language rule for new commits has a local machine check.
- `.gitignore` covers the products in the workspace that are now ignored only by scattered rules.
- Decide how to track the two untracked openspec directories (draft in Chinese, translate to English before they enter the repo).

**Non-Goals**
- Do not rewrite past commits. Do not force-push shared branches.
- Do not redo log identification, `git rm --cached`, or `test_no_tracked_logs.py`.
- Do not add a CI English check.
- Do not sync the fork. Do not touch the `upstream` remote (see the fork merge plan).
- Do not fix the existing red state of main CI.
- Do not upload evidence logs to object storage.

## Decisions

### D1 Constrain only the future, do not change history
Rewriting 77 commits that contain Chinese would change the hashes of the 620 commits since 2026-10-05. Every SHA cited in PRs #238/#239/#240, in worktree branches, and in documents would become invalid. The only gain is a cleaner history. So we do not rewrite (decided by the user).

### D2 Check rule: ban CJK character ranges
- Check target: subject and body.
- Verdict: the check fails if any character is in U+4E00–U+9FFF, U+3400–U+4DBF, the full-width punctuation range U+3000–U+303F, or U+FF00–U+FFEF.
- Do not ban other non-ASCII characters. This avoids false hits on arrows and accented names.
- Skip comment lines that start with `#` (git editor template).
- The alternative "allow only ASCII" is rejected: names and symbols outside `Co-Authored-By` would be rejected by mistake.

### D3 Local hook only
- Add a hook with `stages: [commit-msg]` to `.pre-commit-config.yaml`. It calls the check script.
- Each person runs `pre-commit install --hook-type commit-msg` once. The hook has no effect until it is installed. This is a built-in limit of a local hook. The user accepted it. There is no CI fallback.
- PR titles and bodies are outside the machine check. Review covers them.

### D4 0.4 and log removal
- The user cancelled the requirement "log cleanup merges before #238". Checked: #238 adds no logs.
- Log removal has its own PR #240 (branch `s23-move-run-logs`, based on `agentenv/main` c9d25b69).
- After #240 merges, #239 drops these two commits (git drops them during rebase, or drop them by hand; not tested).
- This change no longer cherry-picks 4559533b into #238.

### D5 Fill `.gitignore` gaps
`git status --ignored` shows that the products below are now ignored by the 0.2 rules or other rules, but they have no matching explicit rule, or the rule is too wide. Check each one before adding a rule:
- `.txt`, `.json`, and `.sh` files under `openspec/changes/*/run-logs/` (now ignored by `**/run-logs/**`, so already covered; only confirm it in tasks).
- `openspec/changes/*/evidence/**/work/` (work directories; now ignored, but the source rule is not checked one by one).
- Common caches: `.mypy_cache/`, `.ruff_cache/`, `.ipynb_checkpoints/`, `.DS_Store` (they do not appear in the workspace now; this is prevention).
At execution, run `git check-ignore -v <path>` for each item to confirm the source rule. Add rules only for gaps. Do not change lines 82–91 written by 0.2.

### D6 Chinese openspec directories: draft in Chinese, translate to English before they enter the repo
- Draft and review with the user in Chinese.
- Before they enter the repo, translate `rl-code-quality-cleanup`, `rl-verl-agentenv-codex`, and this change directory to English. Only the English version enters the repo.
- The translation must keep the original degree of certainty (a claim marked unverified stays unverified; do not turn it into an affirmative sentence).
- After they enter the repo, `tests/test_rl_sync_mode_table.py` no longer skips. Run it once in the PR that adds them.

### D7 Evidence log location
Keep evidence logs first on the local machine at `/home/michael/work/infra-drafts/evidence/`. Do not upload them to object storage. Do not add them to the repo.

## Risks / Trade-offs

- The local hook depends on each person installing it, with no CI fallback. A Chinese commit may enter a PR. Check by eye during PR review.
- Translating about 108KB of text has a cost, and the translation may lose the degree of certainty. After translation, the user or another agent spot-checks sentences of the "unverified" kind.
- Evidence logs exist only on the local machine. If the machine fails, they are lost. Backup relies on workspace-backup (this change does not handle it).

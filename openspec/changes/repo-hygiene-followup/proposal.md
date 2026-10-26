## Why

P6 (repo rules and fork merge) is only half done. Log removal from the repo is done in commit 4559533b (`rl-code-quality-cleanup` 0.1–0.3) and now has its own PR #240. The rest is not done. The English rule for commits and PRs lives only in agent memory, with no machine check. `.gitignore` still has gaps. Two Chinese openspec change directories are not tracked. Fork sync now belongs to the fork merge plan (see below).

## What Changes

- Add an English check for commit messages. Use only a local `commit-msg` hook (it checks both subject and body). Do not add a CI check. Do not change past commits.
- Fill gaps in `.gitignore`. Cover the evidence work directories and similar items that are now ignored only by scattered rules in the workspace. See design.md for details.
- Decide how to track the two untracked openspec directories (`rl-code-quality-cleanup`, `rl-verl-agentenv-codex`). Draft and review in Chinese. Translate all of them to English before they enter the repo.
- Keep evidence logs first on the local machine at `/home/michael/work/infra-drafts/evidence/`. Do not upload them to object storage.
- The ordering requirement of 0.4 is cancelled. Log removal goes through PR #240.
- Fork sync (fast-forward of origin) is not done in this change. See the fork merge plan. Leave the `upstream` remote as is for now.

## Boundary with rl-code-quality-cleanup phase 0

- This change does **not** do: log identification and `git rm --cached` (done in 0.1 and 0.2), `tests/test_no_tracked_logs.py` (done in 0.3), removal of reference stubs and pin unification (done in 0.5–0.8).
- 0.4 no longer requires "log cleanup merges before #238" (see "Decided"). The check mark for 0.4 stays in `rl-code-quality-cleanup/tasks.md`. That change checks it. This change does not edit that directory.
- `.gitignore` gets only rules that 0.2 did not cover. Do not touch the log rule section that 0.2 added (lines 82–91).

## Key numbers (read on 2026-10-13 in worktree `/home/michael/work/s23-code-quality`)

- `agentenv/main` has 2085 commits. 15 have a Chinese subject. 77 have Chinese in the subject or body.
- The first Chinese subject: 2026-10-05 `elastic-ops: Flash-Next <Chinese text> (5a)`. There are 620 commits since 2026-10-05.
- `s22-agentic-sft-modal`, `s23-p0-verified`, and `s23-code-quality` are 4, 15, and 21 commits ahead of `agentenv/main`. The number of Chinese subjects in each is 0.
- `origin/main` stays at 2026-09-24 (c40a32c8). It is 1679 commits behind `agentenv/main` and 0 ahead. `upstream/main` stays at 2026-09-30 (7fb30fe8). It is 1638 behind and 0 ahead. `upstream` and `agentenv` point to the same URL.
- `s22-agentic-sft-modal` is 4 commits ahead of `agentenv/main` and 0 behind (direction checked).
- A trial merge of 4559533b into `s22-agentic-sft-modal` with `git merge-tree` has no conflict (trial merge only; no real cherry-pick; tests not run).
- Of the 32 openspec change directories already in the repo, the `.md` files of 32 contain Chinese. 113 tracked files outside openspec contain Chinese.

## Decided (user, 2026-10-13)

1. **The 0.4 ordering requirement is cancelled.** I checked that #238 adds no logs. So log cleanup no longer has to merge before #238. Log removal has its own PR #240 (https://github.com/agentenv/yeto/pull/240, branch `s23-move-run-logs`, based on `agentenv/main` c9d25b69). After #240 merges, #239 drops these two commits. The old plans A/B/C are void.
2. **Chinese openspec directories.** Draft and review with the user in Chinese. Translate all of them to English before they enter the repo. Do not track them "as is in Chinese" any more.
3. **Do not rewrite past Chinese commits.**
4. **Check scope.** Check both the commit subject and the body. Ban only Han characters and full-width punctuation (do not ban `→` or `é`). Implement it with a local git hook (`commit-msg`).
5. **No CI English check.** Delete the CI job and the related requirement.
6. **Leave the `upstream` remote as is for now.** The task to delete `upstream` is cancelled.
7. **Evidence logs.** Keep them first on the local machine at `/home/michael/work/infra-drafts/evidence/`. Do not upload them to object storage.
8. **Fork sync.** The user decided on 2026-10-13: all development goes into agentenv from now on, and the personal fork content merges into agentenv. A merge plan will be written separately. Tasks in this change that relate to the fork fast-forward become "see the fork merge plan; not done in this change".

## Capabilities

### New Capabilities
- `repo-hygiene`: checkable requirements for repo commit rules (language of commit text, ignore rules).

### Modified Capabilities
(none)

## Impact

- Add `scripts/check_commit_language.py` (name to be decided). Add a `commit-msg` stage hook to `.pre-commit-config.yaml`. Do not change `ci.yml`.
- `.gitignore` gets a few lines.
- No remote operation (fork sync: see the fork merge plan; #240 is a separate PR).
- Do not change history. Do not force-push `agentenv/main`.

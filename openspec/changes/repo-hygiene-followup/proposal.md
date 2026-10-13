## Why

P6 (repo conventions and fork merge) is only half done. Moving logs out of the repo is done in commit 4559533b (`rl-code-quality-cleanup` 0.1-0.3), and it already has its own PR #240. The remaining parts are not in place. The English rule for commits and PRs exists only in agent memory and has no machine check. `.gitignore` still has gaps. Two Chinese openspec change directories are not in the repo. Fork sync is now handled by the fork merge plan (see below).

## What Changes

- Add an English check for commit messages: only a local `commit-msg` hook (both subject and body are checked). No CI check. History commits are not changed.
- Complete `.gitignore`: cover the evidence work directories and other items that are currently scattered in the working tree in an "ignored" state. For details, see design.md.
- Decide how the two untracked openspec directories (`rl-code-quality-cleanup`, `rl-verl-agentenv-codex`) enter the repo: drafting and review use Chinese, and everything is translated to English before it enters the repo.
- Evidence logs go first to the local `/home/michael/work/infra-drafts/evidence/` and are not uploaded to object storage.
- The order requirement of 0.4 is cancelled. Moving logs out goes through PR #240.
- Fork sync (fast-forward of origin) is not executed in this change. See the fork merge plan. The `upstream` remote is left alone for now.

## Boundary with phase 0 of rl-code-quality-cleanup

- This change **does not do**: log detection and `git rm --cached` (done in 0.1 and 0.2), `tests/test_no_tracked_logs.py` (done in 0.3), removal of reference stubs and unifying pins (done in 0.5-0.8).
- 0.4 has cancelled the order requirement "clean up logs before #238 merges" (see "Decided"). The tick for 0.4 is still recorded in `rl-code-quality-cleanup/tasks.md` and is ticked by that change. This change does not edit that directory.
- `.gitignore` gets only rules that 0.2 did not cover. This change does not touch the log rule section added by 0.2 (lines 82-91).

## Key numbers (read on 2026-10-13 in the worktree `/home/michael/work/s23-code-quality`)

- `agentenv/main` has 2085 commits in total. 15 have Chinese in the subject. 77 have Chinese in the subject or body.
- The first Chinese subject: 2026-10-05 `elastic-ops: Flash-Next hardware checklist (5a)` (the original subject is in Chinese). There are 620 commits in total since 2026-10-05.
- `s22-agentic-sft-modal`, `s23-p0-verified`, and `s23-code-quality` have 4, 15, and 21 commits on top of `agentenv/main`. The count of Chinese subjects is 0 in all three.
- `origin/main` stops at 2026-09-24 (c40a32c8). It is 1679 commits behind `agentenv/main` and 0 ahead. `upstream/main` stops at 2026-09-30 (7fb30fe8). It is 1638 behind and 0 ahead. `upstream` and `agentenv` point to the same URL.
- `s22-agentic-sft-modal` is 4 commits ahead of `agentenv/main` and 0 behind (direction verified).
- A trial merge of 4559533b into `s22-agentic-sft-modal` with `git merge-tree`: no conflict (trial merge only; no actual cherry-pick, no tests run).
- Of the 32 openspec change directories already in the repo, the `.md` files of 32 contain Chinese. Outside openspec, 113 tracked files in the repo contain Chinese.

## Decided (user, 2026-10-13)

1. **The order requirement of 0.4 is cancelled**: checked that #238 added no logs, so logs no longer have to be cleaned up before #238 merges. Moving logs out has its own PR #240 (https://github.com/agentenv/yeto/pull/240, branch `s23-move-run-logs`, based on `agentenv/main` c9d25b69). After #240 merges, #239 drops these two commits. The original plans A/B/C are void.
2. **Chinese openspec directories**: drafting and the review by the user use Chinese; everything is translated to English before it enters the repo. They no longer enter the repo as-is in Chinese.
3. **History commits with Chinese are not rewritten.**
4. **Check scope**: both the commit title and the body are checked. Only Chinese characters and fullwidth punctuation are forbidden (`→` and `é` are not forbidden). It is implemented with a local git hook (`commit-msg`).
5. **No CI English check**: remove the CI job and the related requirement.
6. **The `upstream` remote is left alone for now**: the task to delete `upstream` is cancelled.
7. **Evidence logs**: go first to the local `/home/michael/work/infra-drafts/evidence/`, and are not uploaded to object storage.
8. **Fork sync**: the user decided on 2026-10-13 that all development is in agentenv from now on, and the content of the personal fork merges into agentenv. The merge plan is made separately. The tasks in this change related to the fork fast-forward become "see the fork merge plan, not executed in this change".

## Capabilities

### New Capabilities
- `repo-hygiene`: Checkable requirements for repo commit conventions (the language of commit text, and ignore rules).

### Modified Capabilities
(None)

## Impact

- Add `scripts/check_commit_language.py` (name to be decided), and add a `commit-msg` stage hook to `.pre-commit-config.yaml`. `ci.yml` is not changed.
- `.gitignore` gets some new lines.
- No remote operation (for fork sync, see the fork merge plan; #240 is a separate PR).
- History is not changed. `agentenv/main` is not force-pushed.

# Miles `yeto-v30` Pin Update Design

Date: 2026-08-11

Status: Approved for implementation

## Goal

Publish the Miles commit currently used by the two-node secrlenv training run,
`ed9f8c36d89d560c9cd05820225f044cfef074f7`, by fast-forwarding the Miles
remote branch `yeto-v30`. Then update Yeto branch
`feat/secrlenv-infra-abort-v28` to pin that exact commit and its attested Git
bundle.

## Current State

- Miles remote branch `yeto-v30` points to
  `fa47aec08c54fe6b69b6f302dc6deef1b0091fe5`.
- The n1 and n2 training plan and clean Miles checkouts use
  `ed9f8c36d89d560c9cd05820225f044cfef074f7`.
- `ed9f8c36` is a linear descendant of `fa47aec0` by seven commits.
- The Yeto branch currently pins `fa47aec0` with bundle SHA256
  `1755daa082c522365ad332dca32988e1ef89e38127086a9f27b6c0848c6e5e14`.
- The training source snapshot contains the `ed9f8c36` bundle with SHA256
  `4e2e86d5e144633a6cde95a5d0aa999fe0c7a9e3b4ff4521879444529c487d0d`.

## Scope

1. Verify that the live Miles remote branch has not moved and that the update
   remains a fast-forward.
2. Push `ed9f8c36` to `agentenv/miles:yeto-v30` with a normal, non-force push.
3. Copy the attested `miles-ed9f8c36...bundle` from the n1 training source into
   `yeto/rl/vendor/` while retaining prior bundles.
4. Update only the Yeto Miles commit, bundle path, bundle SHA256, and their
   explicit test expectations.
5. Commit and push the Yeto change to
   `feat/secrlenv-infra-abort-v28`.

## Non-Goals

- Do not update SGLang, the Miles image digest, PEFT, agent code, or model pins.
- Do not rewrite, squash, or force-push Miles history.
- Do not restart or otherwise change the running n1/n2 training task.
- Do not remove historical vendor bundles.

## Safety and Failure Handling

- Re-read the live Miles ref immediately before pushing.
- Require the live ref to be an ancestor of `ed9f8c36`; abort on divergence.
- Use a normal push so a concurrent remote update is rejected automatically.
- Verify the copied bundle SHA256 and HEAD before editing the Yeto pin.
- Do not push the Yeto update until the Miles remote ref resolves to
  `ed9f8c36`.
- Abort before committing if tests or pin-integrity checks fail.

## Verification

- `git ls-remote` reports Miles `refs/heads/yeto-v30` at `ed9f8c36`.
- The copied bundle SHA256 is
  `4e2e86d5e144633a6cde95a5d0aa999fe0c7a9e3b4ff4521879444529c487d0d`
  and its advertised HEAD is `ed9f8c36`.
- Targeted pin, plan, launcher, and bundle tests pass.
- The Yeto diff contains only the new bundle, pin constants, test expectations,
  and this approved design/implementation documentation.
- The Yeto worktree is clean after commit, and the remote feature branch equals
  the local HEAD after push.

## Commit Identity

Use `AlexEisie <1987460907@qq.com>` and sign commits when a usable signing key
is configured.

# yeto release image (rl-identity-tiering L9.12)

Status: build script only. No image is built or pushed. The launcher release
mode below is not implemented.

## What the image is

- Base: the pinned miles-ports image (`MILES_NEXT_IMAGE` in
  `yeto/rl/adapters/miles/pins.py`), by digest.
- One added layer:
  - `/opt/yeto/src/`: `git archive` of one clean yeto commit (tracked files only).
  - `/opt/yeto/release-manifest.json`: `yeto_commit`, `base_image`, `source_root`.
- The layer bytes depend only on the commit (file times are the commit time), so
  one commit gives one layer digest.

Build: `scripts/build_yeto_release_image.sh` (local image tarball) or
`scripts/build_yeto_release_image.sh --push` (push and print the digest). The
script refuses a dirty work tree. It writes `build-record.json` with the commit,
the base image, the tag, the layer sha256 and (after a push) the image digest.

## Release mode (design D2; to implement)

1. `--rl-code-pack release --rl-release-image <repo>@sha256:<digest>`.
   The launcher takes the image digest as `code_pack.digest`, with
   `code_pack.mode = "release"` and `code_pack.uri` = the image reference.
   Nothing is uploaded.
2. Islands run the release image instead of `MILES_NEXT_IMAGE`. The launcher does
   not ship the local tree: no sky `workdir` (`yeto/launch/learner_task.py`, the
   `code_workdir` value), no Modal `add_local_dir` (`yeto/modal_runner.py`).
   The island workdir is `/opt/yeto/src`.
3. Island start: read `/opt/yeto/release-manifest.json` and refuse to start when
   its `yeto_commit` differs from the run manifest. The image digest itself is
   checked by the registry pull (the image is pulled by digest).
4. The island contract (v3) carries the image digest in the code pack slot, so the
   syncer refuses an island from another release image.
5. Upgrade (`--rl-code-pack-upgrade`) from one release image to another follows
   design D4: allowed only with an equal semantic hash; the manifest records
   "image A -> image B".

## Not verified

- No image is built; `crane append -o` (local tarball) and the push path were not run.
- Acceptance of L9.12 needs one launch on each cloud that uses the image code only
  (GPU, review before launch). Not estimated.
- Layer size at commit 4702dfbe: about 25.5 MB (whole repository, including
  `openspec/` and evidence files). A release may want a smaller path set.

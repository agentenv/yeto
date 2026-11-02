# yeto release image

Release mode (`--rl-code-pack release`) runs islands from an image that carries
one clean yeto commit, instead of shipping the local work tree.

## What the image is

- Base: the pinned miles-ports image (`MILES_NEXT_IMAGE` in
  `yeto/rl/adapters/miles/pins.py`), by digest.
- One added layer:
  - `/opt/yeto/src/`: `git archive` of one clean yeto commit (tracked files only).
  - `/opt/yeto/release-manifest.json`: `yeto_commit`, `base_image`, `source_root`,
    `source_tree_sha256` (`yeto.provenance.source_tree_sha256` of `/opt/yeto/src/yeto`).
  - `~/sky_workdir -> /opt/yeto/src` in the root user's home: island scripts
    `cd ~/sky_workdir`.
- The layer bytes depend only on the commit (file times are the commit time), so
  one commit gives one layer digest.

Build: `scripts/build_yeto_release_image.sh` (local image tarball) or
`scripts/build_yeto_release_image.sh --push` (push and print the digest). The
script refuses a dirty work tree. It writes `build-record.json` with the commit,
the base image, the tag, the layer sha256 and (after a push) the image digest.

## Release mode

1. `--rl-code-pack release --rl-image docker:<repo>@sha256:<digest>` (add
   `--rl-image-private` for a private package; a new GHCR package is private).
   The launcher takes the image digest as `code_pack.digest`, with
   `code_pack.mode = "release"`, `code_pack.uri` = the image reference and no
   commit (the island reads it from the image). Nothing is uploaded. Miles backend only.
2. Islands run the release image. The launcher ships no local tree: no sky
   `workdir`, no Modal `add_local_dir` (`yeto.launch.code_pack.code_workdir` is None).
   Clouds that run the image with in-VM docker (Verda) refuse release mode: that
   path bind-mounts the host workdir over the image link.
3. Island start (`--code-pack-mode release`): `yeto.codepack.verify_release_tree`
   refuses to start unless the island code root is the manifest's `source_root`
   and its `yeto/` source hash equals `source_tree_sha256`. The island then
   reports that source hash instead of the launcher's `--source-sha256`. The image
   digest itself is checked by the registry pull (the image is pulled by digest).
4. The island contract carries the image digest in the code pack slot, so the
   syncer refuses an island from another release image.
5. A run moves to another image, or between git-patch and release, only with
   `--rl-code-pack-upgrade REASON` (the semantic hash must be equal; the manifest
   records the step with `from_mode` / `to_mode`).
6. The syncer head still runs from the local tree; release mode covers islands only.

## Not verified

- Byte-identical layers from two builds of one commit (designed for it; not compared).
- sky clouds (docker `image_id`) in release mode; only Modal has been launched.

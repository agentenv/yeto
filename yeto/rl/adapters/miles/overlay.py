"""Miles source overlay for the critic family (rl-algo-critic-family, S13).

The ports image ships Miles c35702e (``yeto.rl.MILES_NEXT_COMMIT``) installed
editable at ``/root/miles`` (= ``~/miles``) and records that commit in
``/opt/yeto/image-manifest.json``.  The critic-family flags live on the fork
branch ``yeto-critic-c357`` (michaellchung/miles, local, NOT pushed): the
critic-family commits rebased onto c35702e.  Instead of a push + new image,
the island setup applies ``c35702e..CRITIC_C357_RESULT_COMMIT`` as a
``git apply`` patch shipped inside the frozen code snapshot (the workdir).

Safety and honesty rules (progress "S13 fork rebase c35702e + overlay"):

* the overlay is refused unless ``~/miles`` is the image's own checkout of
  c35702e (HEAD == c35702e, no refresh by the setup, image manifest naming
  c35702e at that path) -- it never patches some other tree;
* the patch file's sha256 is checked in the container before ``git apply``;
* after applying, a record is written to :data:`OVERLAY_RECORD_PATH`; the run
  manifest and the runtime manifest report "image c35702e + overlay <sha256>"
  and flag that the image manifest (and ``git rev-parse HEAD``) no longer
  describe the running code.

Off by default: ``--rl-miles-overlay auto`` (the default) enables it only for
a spec whose Miles argv uses a fork-only flag; ``critic-c357`` forces it,
``off`` never applies it.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

from yeto.rl import MILES_NEXT_COMMIT, MILES_NEXT_IMAGE_MANIFEST

CRITIC_C357 = "critic-c357"
OVERLAY_CHOICES = ("auto", "off", CRITIC_C357)

# The fork commit the patch reproduces (git diff --binary c35702e..this).
CRITIC_C357_BASE_COMMIT = MILES_NEXT_COMMIT
CRITIC_C357_RESULT_COMMIT = "6574a9c82edeed17f4b00bc64db6c7cd6b45cb39"
# git tree of CRITIC_C357_RESULT_COMMIT: the patched worktree must hash to exactly this.
CRITIC_C357_RESULT_TREE = "d65bda3de776a5701e9d0d2fc2bd51cc254779bd"
CRITIC_C357_PATCH = "yeto/rl/adapters/miles/overlays/miles-critic-c357.patch"  # relative to the workdir
CRITIC_C357_PATCH_SHA256 = "64f69bbf8815eeed7ad1b02f783f8ed903d3c5b2aa5b6e2a3c7e537961bdca3b"

# Written by the island setup; outside ~/miles so the record is not part of the patch.
OVERLAY_RECORD_PATH = "~/.yeto-miles-overlay.json"

# Miles flags that exist only with the overlay (git diff c35702e..result of arguments.py).
FORK_ONLY_FLAGS = frozenset({
    "--critic-freeze-attention", "--critic-updates-per-step", "--num-critic-epochs",
    "--gae-critic-lambd", "--gae-lambd-mode", "--gae-length-alpha", "--gae-variant",
    "--hl-gauss-sigma-ratio", "--policy-objective",
    "--positive-example-lm-loss-coef", "--positive-example-reward-threshold",
    "--positive-example-source", "--sao-dis-eps-high", "--sao-dis-eps-low",
    "--value-loss-type", "--value-num-bins", "--value-reward-range", "--value-target-type",
})


def repo_patch_path() -> Path:
    return Path(__file__).resolve().parents[4] / CRITIC_C357_PATCH


def spec_fork_flags(spec_json: str | None) -> list[str]:
    """Fork-only Miles flags the prepared spec's argv uses (empty = none / no spec)."""

    if not spec_json:
        return []
    from yeto.rl.engine.algorithm import AlgorithmSpec
    from yeto.rl.adapters.miles import algo_flag_rows as rows
    from yeto.rl.adapters.miles.algorithm_flags import algorithm_argv

    spec = AlgorithmSpec.from_dict(json.loads(spec_json))
    argv = (algorithm_argv(spec) + rows.critic_argv(spec)
            + rows.positive_lm_argv(spec) + rows.sao_fork_argv(spec))
    return sorted({a for a in argv if a in FORK_ONLY_FLAGS})


def resolve_overlay(args) -> str | None:
    """The overlay to apply for this launch (None = off)."""

    choice = getattr(args, "rl_miles_overlay", None) or "auto"
    if choice not in OVERLAY_CHOICES:
        raise ValueError(f"--rl-miles-overlay must be one of {OVERLAY_CHOICES}, got {choice!r}")
    if (getattr(args, "training_mode", "sft") != "rl"
            or getattr(args, "rl_engine", "ports") != "ports" or choice == "off"):
        if choice == CRITIC_C357:
            raise ValueError("--rl-miles-overlay critic-c357 needs --training-mode rl on the ports engine")
        return None
    if choice == CRITIC_C357:
        return CRITIC_C357
    return CRITIC_C357 if spec_fork_flags(getattr(args, "rl_algorithm_spec_json", None)) else None


def overlay_record(overlay: str | None) -> dict[str, Any] | None:
    """What the run manifest records for the overlay (None when off)."""

    if overlay is None:
        return None
    if overlay != CRITIC_C357:
        raise ValueError(f"unknown Miles overlay {overlay!r}")
    return {
        "name": CRITIC_C357,
        "base_commit": CRITIC_C357_BASE_COMMIT,
        "result_commit": CRITIC_C357_RESULT_COMMIT,
        "result_commit_pushed": False,
        "patch": CRITIC_C357_PATCH,
        "patch_sha256": CRITIC_C357_PATCH_SHA256,
        "summary": f"image miles {CRITIC_C357_BASE_COMMIT[:7]} + overlay {CRITIC_C357_PATCH_SHA256}",
        "image_manifest_matches_code": False,
        "note": ("image-manifest.json and git HEAD of ~/miles still name the base commit; "
                 "the running Miles code is base + this patch"),
    }


def overlay_setup(overlay: str | None) -> str:
    """Shell appended to the ports Miles setup (after the checkout / install
    steps, which set ``MILES_REFRESHED``); empty when off."""

    record = overlay_record(overlay)
    if record is None:
        return ""
    base = CRITIC_C357_BASE_COMMIT
    patch = f"~/sky_workdir/{CRITIC_C357_PATCH}"
    manifest = MILES_NEXT_IMAGE_MANIFEST
    refuse = f"echo '[yeto-setup] REFUSING miles overlay {CRITIC_C357}:"
    record_py = (
        "import json, os, sys; r = json.loads(sys.argv[1]); "
        "r['applied'] = True; "
        "json.dump(r, open(os.path.expanduser(sys.argv[2]), 'w'), indent=1, sort_keys=True)"
    )
    return (
        f'if [ "$MILES_REFRESHED" != 0 ]; then {refuse} ~/miles was not the image checkout of {base}' + "'"
        " >&2; exit 1; fi\n"
        f'if [ "$(git -C ~/miles rev-parse HEAD)" != {base} ]; then {refuse} ~/miles HEAD is not {base}'
        + "' >&2; exit 1; fi\n"
        "if ! python3 -c 'import json, os, sys; "
        f'm = json.load(open("{manifest}"))["miles"]; '
        f'sys.exit(0 if m["commit"] == "{base}" and '
        'os.path.realpath(m["path"]) == os.path.realpath(os.path.expanduser("~/miles")) else 1)'
        f"' 2>/dev/null; then {refuse} image manifest does not name {base} at ~/miles' >&2; exit 1; fi\n"
        f'if [ -n "$(git -C ~/miles status --porcelain --untracked-files=all)" ]; then '
        f"{refuse} ~/miles is dirty' >&2; exit 1; fi\n"
        f"printf '%s  %s\\n' {CRITIC_C357_PATCH_SHA256} {patch} | sha256sum --check -\n"
        f"git -C ~/miles apply --check {patch}\n"
        f"git -C ~/miles apply {patch}\n"
        f"python3 -c {shlex.quote(record_py)} {shlex.quote(json.dumps(record, sort_keys=True))} "
        f"{OVERLAY_RECORD_PATH}\n"
        f"echo '[yeto-setup] miles overlay applied: {record['summary']} "
        f"(= fork {CRITIC_C357_RESULT_COMMIT}, not pushed; image manifest no longer matches the code)'"
    )


def read_applied_record(path: str = OVERLAY_RECORD_PATH) -> dict[str, Any] | None:
    """The record the setup wrote in this container (None if no overlay)."""

    try:
        return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

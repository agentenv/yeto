"""Learner image selection and the Miles source setup."""

from __future__ import annotations

import shlex
import sys

from ..gpu_spec import ClusterSpec
from .rl_backend import _miles_overlay


# AWS keeps this SSM parameter pointing at the CURRENT Deep Learning Base
# OSS-NVIDIA-driver AMI per region (open kernel modules, Blackwell-ready),
# so resolving at launch time needs no region x AMI table of our own.
_DLAMI_SSM_PARAM = (
    "/aws/service/deeplearning/ami/x86_64/"
    "base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id"
)


def resolve_blackwell_image(region: str) -> str | None:
    """Current Blackwell-capable DLAMI for a region, or None (callers then
    rely on the setup-time driver remediation instead)."""
    try:
        import boto3

        ssm = boto3.client("ssm", region_name=region)
        return ssm.get_parameter(Name=_DLAMI_SSM_PARAM)["Parameter"]["Value"]
    except Exception as e:  # noqa: BLE001 - degrade to driver remediation
        print(
            f"[launcher] could not resolve a Blackwell DLAMI for {region} ({e}); "
            "relying on setup-time driver install",
            file=sys.stderr,
        )
        return None


# Internal image-override table: (cloud, GPU) pairs whose provider-default
# image is known stale/broken, mapped to a `region -> image id` resolver.
# First entry earned in production: sky's pinned AMI ships driver 535,
# which never binds to SM100 silicon. Second class earned the same way:
# NVSwitch instances (p4d/p4de A100, p5 H100, p5e H200) need
# nvidia-fabricmanager running before CUDA will initialize at all
# (cudaGetDeviceCount -> Error 802), and sky's pinned AMI does not ship it;
# the DL Base GPU AMI does, preinstalled and enabled. Extend here as new
# GPU generations outpace provider image pins; an explicit --learner-image
# always wins, and a resolver returning None degrades to the setup-time
# driver install.
GPU_IMAGE_OVERRIDES: dict[tuple[str, str], object] = {
    ("aws", "B200"): resolve_blackwell_image,
    # NVSwitch (fabric manager required); same current DL Base GPU AMI.
    ("aws", "A100"): resolve_blackwell_image,
    ("aws", "A100-80GB"): resolve_blackwell_image,
    ("aws", "H100"): resolve_blackwell_image,
    ("aws", "H200"): resolve_blackwell_image,
}


# The Megatron backend runs INSIDE an NGC container that ships the whole
# stack prebuilt (torch + Transformer Engine + megatron-core + megatron-bridge)
# — the pip-on-DLAMI install proved intractable (two shakedowns; see
# docs/MEGATRON.md). sky runs the task in this container via `image_id:
# docker:...`; the host still supplies the GPU kernel driver, so B200 needs
# sky's default AWS docker host AMI to carry driver >=570 open-kernel (the one
# integration unknown that still needs a live B200 check). NeMo images ship
# megatron-bridge; pull requires nvcr.io auth (an NGC key on the host).
MEGATRON_IMAGE = "docker:nvcr.io/nvidia/nemo:25.09"


def learner_image_for(args, spec: ClusterSpec, learner_id: int | None = None):
    """The image for a learner cluster: explicit flag > megatron container >
    internal override table > None (provider default + setup-time remediation)."""
    explicit = parse_image_spec(getattr(args, "learner_image", None))
    if explicit is not None:
        if isinstance(explicit, dict) and learner_id is not None:
            for key in (str(learner_id), f"l{learner_id}"):
                if key in explicit:
                    return explicit[key]
        return explicit
    if getattr(args, "island_backend", "torch") == "megatron":
        return MEGATRON_IMAGE
    resolver = GPU_IMAGE_OVERRIDES.get((spec.cloud, spec.gpu))
    if resolver is None or not spec.region:
        return None
    image = resolver(spec.region)
    if image:
        print(f"[launcher] {spec.gpu} learner: pinning image {image} ({spec.region})")
    return image


def parse_image_spec(value: str | None):
    """--learner-image: a single image id/tag applied everywhere, or
    comma-separated region=id pairs -> the region dict sky expects. Numeric
    keys are learner ids and are resolved before region keys; this covers
    providers where a saved OS volume is not a reusable image."""
    if not value:
        return None
    if "=" not in value:
        return value
    images = {}
    for pair in value.split(","):
        key, _, image = pair.partition("=")
        if not key or not image:
            raise ValueError(
                f"bad --learner-image entry {pair!r}; expected region=image-id "
                "or learner-id=image-id"
            )
        images[key.strip()] = image.strip()
    return images


# Exit codes of the image-manifest check (:func:`image_manifest_check`).
MANIFEST_MATCH, MANIFEST_ABSENT, MANIFEST_MISMATCH = 0, 3, 4
# Stdlib-only: reading the manifest must not import the packages it describes
# (``import peft`` ~27 s, ``import sglang`` ~18 s; coldstart-gpu-review 6.1).
_MANIFEST_CHECK_PY = """\
import json, os, sys
path, package, key, want, want_path = sys.argv[1:6]
try:
    entry = json.load(open(path)).get(package) or {}
except FileNotFoundError:
    sys.exit(3)
have = entry.get(key)
if have is None:
    sys.exit(3)
if have != want:
    print(f"[yeto-setup] image manifest {path}: {package} {key} {have} "
          f"!= pinned {want}", file=sys.stderr)
    sys.exit(4)
if want_path and os.path.realpath(entry.get("path", "")) != os.path.realpath(want_path):
    print(f"[yeto-setup] image manifest {path}: {package} path {entry.get('path')} "
          f"!= {want_path}", file=sys.stderr)
    sys.exit(4)
"""


def image_manifest_check(package: str, key: str, want: str, *, fallback: str,
                         path: str = "", on_pass: str = ":") -> str:
    """Shell lines: check ``package`` against the image manifest, without importing it.

    Manifest entry equal to ``want`` (and at ``path`` when given): run
    ``on_pass``.  Manifest or entry absent: record ``image_manifest_fallback``
    and run ``fallback`` (the old import check).  Different value: fail the
    setup (before Ray starts), naming the package and both values.
    """

    from ..rl import MILES_NEXT_IMAGE_MANIFEST
    from ..rl.engine.start_phases import shell_event

    argv = " ".join(shlex.quote(a) for a in (MILES_NEXT_IMAGE_MANIFEST, package, key, want, path))
    event = shell_event("image_manifest_fallback",
                        {"package": package, "manifest": MILES_NEXT_IMAGE_MANIFEST,
                         "check": "import"})
    return (
        f"rc=0; python3 -c {shlex.quote(_MANIFEST_CHECK_PY)} {argv} || rc=$?\n"
        'case "$rc" in\n'
        f"{MANIFEST_MATCH}) echo '[yeto-setup] image manifest: {package} {key} {want}'; {on_pass} ;;\n"
        f"{MANIFEST_ABSENT})\n{event}\n{fallback} ;;\n"
        f"*) echo \"[yeto-setup] image manifest check of {package} failed (exit $rc); "
        "refusing to start\" >&2; exit 1 ;;\n"
        "esac"
    )


def _miles_source_setup(overlay: str | None = None) -> tuple[str, str]:
    """Return the (miles_setup, sglang_setup) remote steps of the pinned Miles/SGLang forks.

    ``overlay`` (default None = off): a :mod:`yeto.rl.adapters.miles.overlay`
    patch applied to the image's ~/miles after the checkout checks.
    """

    from ..rl import (
        MILES_NEXT_COMMIT,
        MILES_NEXT_IMAGE_MANIFEST,
        MILES_NEXT_IMAGE_SGLANG_ROOT,
        MILES_NEXT_REPOSITORY,
        MILES_PEFT_VERSION,
        SGLANG_NEXT_COMMIT,
        SGLANG_NEXT_REPOSITORY,
    )

    def checkout(
        path: str, repository: str, commit: str, refreshed_flag: str = ""
    ) -> str:
        repo = shlex.quote(repository)
        # ``refreshed_flag``: shell variable set to 1 when the checkout
        # had to move to the pin (so the caller knows the image's
        # install of it can no longer be trusted as-is).
        mark = f"{refreshed_flag}=1\n" if refreshed_flag else ""
        return (
            f"if [ ! -d {path}/.git ]; then git clone --no-checkout "
            f"{repo} {path}; fi\n"
            f"git -C {path} remote set-url origin {repo}\n"
            f'test "$(git -C {path} config --get remote.origin.url)" = '
            f"{repo}\n"
            # MILES_NEXT_IMAGE already has the pinned fork checked out
            # at ~/miles (/root/miles); only fetch when it is not HEAD.
            f'if [ "$(git -C {path} rev-parse HEAD 2>/dev/null)" != '
            f"{commit} ]; then\n"
            f"git -C {path} fetch --depth 1 origin {commit}\n"
            f"git -C {path} checkout --detach {commit}\n"
            + mark
            + "fi\n"
            f'test "$(git -C {path} rev-parse HEAD)" = {commit}\n'
            f'test "$(git -C {path} rev-parse --abbrev-ref HEAD)" = HEAD\n'
            f'test -z "$(git -C {path} status --porcelain '
            '--untracked-files=all)"\n'
        )

    # MILES_NEXT_IMAGE ships the pinned SGLang fork installed editable
    # at /sgl-workspace/sglang: ~/sglang (first on the island's
    # PYTHONPATH) becomes a link to it.  Any other image, or an existing
    # ~/sglang directory, gets the clone + install at ~/sglang as before.
    image_root = shlex.quote(MILES_NEXT_IMAGE_SGLANG_ROOT)
    sglang_checkout_in_image = (
        f'{{ [ ! -e ~/sglang ] || [ "$(readlink ~/sglang)" = {image_root} ]; }} && '
        f'[ "$(git -C {image_root} rev-parse HEAD 2>/dev/null)" = '
        f"{SGLANG_NEXT_COMMIT} ] && "
        f'[ "$(git -C {image_root} config --get remote.origin.url)" = '
        f"{shlex.quote(SGLANG_NEXT_REPOSITORY)} ] && "
        f'[ -z "$(git -C {image_root} status --porcelain '
        '--untracked-files=all)" ]'
    )
    # Fallback for an image without a manifest entry: the old import check.
    sglang_import_ok = (
        "python3 -c 'import os, sys, sglang; sys.exit(0 if os.path.realpath("
        f'sglang.__file__).startswith("{MILES_NEXT_IMAGE_SGLANG_ROOT}/python/'
        "\") else 1)'"
    )
    # MILES_NEXT_IMAGE also ships the pinned Miles fork installed
    # editable at /root/miles (= ~/miles) and records it in
    # MILES_NEXT_IMAGE_MANIFEST.  When ~/miles was already at the pin
    # (no fetch above) and the manifest names that commit at that path,
    # the image's install is kept: the editable re-install is skipped
    # and only a missing/other peft is (re)installed.  Any other image,
    # a moved checkout, or a manifest mismatch gets the full install.
    manifest = shlex.quote(MILES_NEXT_IMAGE_MANIFEST)
    miles_in_image = (
        '[ "$MILES_REFRESHED" = 0 ] && '
        f"python3 -c 'import json, os, sys; "
        f'm = json.load(open("{MILES_NEXT_IMAGE_MANIFEST}"))["miles"]; '
        f'sys.exit(0 if m["commit"] == "{MILES_NEXT_COMMIT}" and '
        'os.path.realpath(m["path"]) == '
        "os.path.realpath(os.path.expanduser(\"~/miles\")) else 1)' "
        "2>/dev/null"
    )
    peft_ok = (
        "python3 -c 'import peft, sys; "
        f"sys.exit(0 if peft.__version__ == \"{MILES_PEFT_VERSION}\" else 1)' "
        "2>/dev/null"
    )
    peft_check = image_manifest_check(
        "peft", "version", MILES_PEFT_VERSION,
        fallback=f"{peft_ok} || python3 -m pip install -q 'peft=={MILES_PEFT_VERSION}'",
    )
    sglang_check = image_manifest_check(
        "sglang", "commit", SGLANG_NEXT_COMMIT, path=MILES_NEXT_IMAGE_SGLANG_ROOT,
        on_pass="SGLANG_IN_IMAGE=1",
        fallback=f"if {sglang_import_ok}; then SGLANG_IN_IMAGE=1; fi",
    )
    return (
        "set -e\n"
        "MILES_REFRESHED=0\n"
        + checkout(
            "~/miles", MILES_NEXT_REPOSITORY, MILES_NEXT_COMMIT, "MILES_REFRESHED"
        )
        + f"if {miles_in_image}; then\n"
        f"echo \"[yeto-setup] image provides miles {MILES_NEXT_COMMIT} "
        f'({manifest}); editable install kept"\n'
        f"{peft_check}\n"
        "else\n"
        "python3 -m pip install -q --no-deps -e ~/miles "
        f"'peft=={MILES_PEFT_VERSION}'\n"
        "fi"
        + (f"\n{_miles_overlay().overlay_setup(overlay)}" if overlay else ""),
        "SGLANG_IN_IMAGE=0\n"
        f"if {sglang_checkout_in_image}; then\n"
        f"{sglang_check}\n"
        "fi\n"
        'if [ "$SGLANG_IN_IMAGE" = 1 ]; then\n'
        f"ln -sfn {image_root} ~/sglang\n"
        f"echo '[yeto-setup] image provides sglang {SGLANG_NEXT_COMMIT}'\n"
        "else\n"
        + checkout("~/sglang", SGLANG_NEXT_REPOSITORY, SGLANG_NEXT_COMMIT)
        + "python3 -m pip install -q --no-deps -e ~/sglang/python\n"
        "fi",
    )

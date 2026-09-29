"""Runtime manifest for the ports engine image (rl-infra-spec task 1.1).

Two halves:

* :func:`collect` runs INSIDE the pinned image/checkout (GPU host or image
  shell). It records the Miles/SGLang checkout commits, import paths,
  Megatron/Torch/CUDA/NCCL/torch_memory_saver/PEFT versions, the image digest
  given by the caller, and which fork interfaces exist (probed with
  ``hasattr`` on the imported classes, no GPU needed).
* :func:`check_manifest` / :func:`certify` are pure. The manifest must agree
  with ``MILES_NEXT_*`` / ``SGLANG_NEXT_*`` (and the image digest), and a
  capability may only be declared when every interface it needs is present
  ("缺接口组合拒绝认证").

``python -m yeto.rl.engine.runtime_manifest --image <digest> --out m.json``
writes the manifest; ``--check m.json`` validates one against the pins.
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA = "yeto-rl-runtime-manifest-v1"

# interface name -> (module, attribute path). Probed with getattr only.
INTERFACES: Mapping[str, tuple[str, str]] = {
    "run_plugin": ("miles.ray.train.group", "TrainerController.run_plugin"),
    "fork_m1_placement_map": ("miles.ray.placement_group", "parse_placement_map"),
    "fork_m2_start_stop_cells": ("miles.ray.rollout.inference_controller", "InferenceController.start_cells"),
    "fork_m3_router_cordon": ("miles.router.router", "MilesRouter.cordon_worker"),
    "fork_m4_member_publish": (
        "miles.ray.rollout.inference_controller",
        "InferenceController._start_member_update_weights",
    ),
    "fork_m6_rebuild_trainer": ("miles.ray.placement_group", "rebuild_training_models"),
}

# Declared capability -> interfaces it needs. Missing any => refused.
CAPABILITY_REQUIREMENTS: Mapping[str, tuple[str, ...]] = {
    "ports-colocated-serial": ("run_plugin",),
    "ports-partitioned-serial": ("run_plugin",),
    "fixed-partition-standby": ("run_plugin", "fork_m1_placement_map"),
    "RolloutPool.add_engines": ("fork_m2_start_stop_cells", "fork_m3_router_cordon"),
    "RolloutPool.remove_engines": ("fork_m2_start_stop_cells", "fork_m3_router_cordon"),
    "RolloutPool.drain": ("fork_m3_router_cordon",),
    "Publisher.publish(members)": ("fork_m4_member_publish",),
    "Placement.reconfigure": ("fork_m1_placement_map", "fork_m6_rebuild_trainer"),
}

VERSION_MODULES = ("torch", "megatron.core", "torch_memory_saver", "peft", "sglang", "ray")


class ManifestMismatch(ValueError):
    """The runtime does not match the pins, or a capability lacks an interface."""


def _probe(module: str, path: str) -> bool:
    try:
        obj: Any = importlib.import_module(module)
    except Exception:
        return False
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return False
    return True


def _git_head(path: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _package_root(module: str) -> Path | None:
    try:
        mod = importlib.import_module(module)
    except Exception:
        return None
    file = getattr(mod, "__file__", None)
    return Path(file).resolve().parent.parent if file else None


def _image_manifest() -> dict[str, Any]:
    try:
        from yeto.rl import MILES_NEXT_IMAGE_MANIFEST

        return json.loads(Path(MILES_NEXT_IMAGE_MANIFEST).read_text())
    except Exception:
        return {}


def collect(*, image: str | None) -> dict[str, Any]:
    """Probe the current interpreter (run inside the pinned image)."""

    versions: dict[str, str | None] = {}
    for name in VERSION_MODULES:
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, "__version__", None)
            if version is None:
                import importlib.metadata as md

                dist = {"torch_memory_saver": "torch-memory-saver"}.get(name, name.split(".")[0])
                try:
                    version = md.version(dist)
                except md.PackageNotFoundError:
                    version = None
            versions[name] = str(version) if version is not None else "unknown"
        except Exception:
            versions[name] = None
    cuda = nccl = None
    try:
        import torch

        cuda = torch.version.cuda
        nccl = ".".join(str(x) for x in torch.cuda.nccl.version())
    except Exception:
        pass
    roots = {"miles": _package_root("miles"), "sglang": _package_root("sglang")}
    commits: dict[str, str | None] = {}
    commit_source: dict[str, str] = {}
    image_manifest = _image_manifest()
    for name, root in roots.items():
        commits[name] = _git_head(root) if root else None
        if name == "sglang" and commits[name] is None and root is not None:
            commits[name] = _git_head(root.parent)  # sglang/python/sglang layout
        commit_source[name] = "git"
        if commits[name] is None and image_manifest.get(name, {}).get("commit"):
            # pure-Python overlay images carry no .git; the build record is the
            # only source. Recorded as such, together with the version string.
            commits[name] = image_manifest[name]["commit"]
            commit_source[name] = "image-manifest"
    return {
        "schema": MANIFEST_SCHEMA,
        "image": image,
        "python": sys.version.split()[0],
        "import_paths": {k: (str(v) if v else None) for k, v in roots.items()},
        "commits": commits,
        "commit_source": commit_source,
        "image_build_manifest": image_manifest or None,
        "versions": {**versions, "cuda": cuda, "nccl": nccl},
        "interfaces": {name: _probe(*where) for name, where in INTERFACES.items()},
    }


def expected_pins() -> dict[str, str]:
    from yeto.rl import MILES_NEXT_COMMIT, MILES_NEXT_IMAGE, SGLANG_NEXT_COMMIT

    return {"miles": MILES_NEXT_COMMIT, "sglang": SGLANG_NEXT_COMMIT, "image": MILES_NEXT_IMAGE}


def check_manifest(manifest: Mapping[str, Any], pins: Mapping[str, str]) -> list[str]:
    """Problems that make the manifest disagree with the pins (empty = consistent)."""

    problems = []
    if manifest.get("schema") != MANIFEST_SCHEMA:
        problems.append(f"unknown manifest schema {manifest.get('schema')!r}")
    commits = manifest.get("commits", {})
    for name in ("miles", "sglang"):
        if commits.get(name) != pins.get(name):
            problems.append(f"{name} commit {commits.get(name)!r} != pinned {pins.get(name)!r}")
    image, pinned = manifest.get("image"), pins.get("image")
    if pinned and (not image or image.split("@")[-1] != pinned.split("@")[-1]):
        problems.append(f"image {image!r} != pinned {pinned!r}")
    versions = manifest.get("versions", {})
    for name in ("torch", "megatron.core", "cuda", "nccl", "torch_memory_saver", "peft"):
        if not versions.get(name) or versions.get(name) == "unknown":
            problems.append(f"{name} version not recorded")
    return problems


def missing_interfaces(manifest: Mapping[str, Any], capability: str) -> list[str]:
    if capability not in CAPABILITY_REQUIREMENTS:
        return [f"unknown capability {capability!r}"]
    have = manifest.get("interfaces", {})
    return [i for i in CAPABILITY_REQUIREMENTS[capability] if not have.get(i)]


def certify(
    manifest: Mapping[str, Any], pins: Mapping[str, str], capabilities: Iterable[str]
) -> dict[str, Any]:
    """Refuse certification on any pin mismatch or missing interface."""

    problems = check_manifest(manifest, pins)
    for cap in capabilities:
        missing = missing_interfaces(manifest, cap)
        if missing:
            problems.append(f"capability {cap!r} lacks {missing}")
    if problems:
        raise ManifestMismatch("; ".join(problems))
    return {"certified": sorted(capabilities), "image": manifest.get("image"),
            "commits": dict(manifest.get("commits", {}))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", help="image reference with @sha256 digest")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check", type=Path, help="validate an existing manifest against pins")
    parser.add_argument("--capability", action="append", default=[])
    args = parser.parse_args(argv)
    if args.check:
        manifest = json.loads(args.check.read_text())
    else:
        manifest = collect(image=args.image)
        text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        (args.out.write_text(text) if args.out else sys.stdout.write(text))
    try:
        result = certify(manifest, expected_pins(), args.capability)
    except ManifestMismatch as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

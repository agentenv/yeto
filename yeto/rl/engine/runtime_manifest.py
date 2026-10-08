"""Runtime manifest for the ports engine image (rl-infra-spec task 1.1).

Two halves:

* :func:`collect` runs INSIDE the pinned image/checkout (GPU host or image
  shell). It records the backend checkout commits, import paths, versions,
  the image digest given by the caller, and which backend interfaces exist
  (probed with ``hasattr`` on the imported classes, no GPU needed). What to
  probe is the backend adapter's :class:`RuntimeDescription` (Miles:
  ``miles_adapter.runtime_manifest``; decoupling 2.5).
* :func:`check_manifest` / :func:`certify` are pure. The manifest must agree
  with the backend pins (Miles: ``MILES_NEXT_*`` / ``SGLANG_NEXT_*``, and the image digest), and a
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
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA = "yeto-rl-runtime-manifest-v1"



@dataclass(frozen=True)
class RuntimeDescription:
    """What a backend adapter declares for its runtime manifest (decoupling 2.5).

    ``interfaces``: name -> (module, attribute path), probed with getattr only;
    ``capability_requirements``: declared capability -> interfaces it needs;
    ``version_modules`` recorded, ``required_versions`` must be recorded;
    ``checkouts``: packages whose source commit is recorded (git HEAD, else the
    image build record); ``nested_checkouts`` are looked up one level higher too;
    ``overlay_record()`` describes code patched after the image build, kept under
    ``overlay_field`` and marking ``overlay_checkout``'s commit source.
    """

    interfaces: Mapping[str, tuple[str, str]]
    capability_requirements: Mapping[str, tuple[str, ...]]
    version_modules: tuple[str, ...]
    required_versions: tuple[str, ...]
    checkouts: tuple[str, ...]
    nested_checkouts: tuple[str, ...] = ()
    overlay_checkout: str | None = None
    overlay_field: str = "overlay"
    image_manifest: Callable[[], dict[str, Any]] = dict
    overlay_record: Callable[[], dict[str, Any] | None] = lambda: None
    expected_pins: Callable[[], dict[str, str]] = dict


# Backend name -> "module:attribute" of its RuntimeDescription, resolved by name so
# the core never imports an adapter (the command line keeps Miles as its default).
BACKENDS: Mapping[str, str] = {
    "miles": "yeto.rl.engine.miles_adapter.runtime_manifest:MILES_RUNTIME",
}
DEFAULT_BACKEND = "miles"


def runtime_description(backend: str = DEFAULT_BACKEND) -> RuntimeDescription:
    try:
        module, attr = BACKENDS[backend].split(":")
    except KeyError:
        raise ValueError(f"unknown backend {backend!r}; known: {sorted(BACKENDS)}") from None
    return getattr(importlib.import_module(module), attr)


def _runtime(runtime: RuntimeDescription | None) -> RuntimeDescription:
    return runtime if runtime is not None else runtime_description()


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


def collect(*, image: str | None, runtime: RuntimeDescription | None = None) -> dict[str, Any]:
    """Probe the current interpreter (run inside the pinned image)."""

    runtime = _runtime(runtime)
    versions: dict[str, str | None] = {}
    for name in runtime.version_modules:
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
    roots = {name: _package_root(name) for name in runtime.checkouts}
    commits: dict[str, str | None] = {}
    commit_source: dict[str, str] = {}
    image_manifest = runtime.image_manifest()
    for name, root in roots.items():
        commits[name] = _git_head(root) if root else None
        if name in runtime.nested_checkouts and commits[name] is None and root is not None:
            commits[name] = _git_head(root.parent)  # sglang/python/sglang layout
        commit_source[name] = "git"
        if commits[name] is None and image_manifest.get(name, {}).get("commit"):
            # pure-Python overlay images carry no .git; the build record is the
            # only source. Recorded as such, together with the version string.
            commits[name] = image_manifest[name]["commit"]
            commit_source[name] = "image-manifest"
    # Code patched after the image was built (Miles: S13 critic overlay): neither
    # git HEAD nor the image build manifest describes the running code.
    overlay = runtime.overlay_record()
    if overlay is not None and runtime.overlay_checkout in commit_source:
        commit_source[runtime.overlay_checkout] = f"{commit_source[runtime.overlay_checkout]}+overlay"
    return {
        "schema": MANIFEST_SCHEMA,
        "image": image,
        **({runtime.overlay_field: overlay,
            "image_manifest_matches_code": False} if overlay is not None else {}),
        "python": sys.version.split()[0],
        "import_paths": {k: (str(v) if v else None) for k, v in roots.items()},
        "commits": commits,
        "commit_source": commit_source,
        "image_build_manifest": image_manifest or None,
        "versions": {**versions, "cuda": cuda, "nccl": nccl},
        "interfaces": {name: _probe(*where) for name, where in runtime.interfaces.items()},
    }


def expected_pins(runtime: RuntimeDescription | None = None) -> dict[str, str]:
    return _runtime(runtime).expected_pins()


def check_manifest(manifest: Mapping[str, Any], pins: Mapping[str, str],
                   runtime: RuntimeDescription | None = None) -> list[str]:
    """Problems that make the manifest disagree with the pins (empty = consistent)."""

    runtime = _runtime(runtime)
    problems = []
    if manifest.get("schema") != MANIFEST_SCHEMA:
        problems.append(f"unknown manifest schema {manifest.get('schema')!r}")
    commits = manifest.get("commits", {})
    for name in runtime.checkouts:
        if commits.get(name) != pins.get(name):
            problems.append(f"{name} commit {commits.get(name)!r} != pinned {pins.get(name)!r}")
    image, pinned = manifest.get("image"), pins.get("image")
    if pinned and (not image or image.split("@")[-1] != pinned.split("@")[-1]):
        problems.append(f"image {image!r} != pinned {pinned!r}")
    versions = manifest.get("versions", {})
    for name in runtime.required_versions:
        if not versions.get(name) or versions.get(name) == "unknown":
            problems.append(f"{name} version not recorded")
    return problems


def missing_interfaces(manifest: Mapping[str, Any], capability: str,
                       runtime: RuntimeDescription | None = None) -> list[str]:
    requirements = _runtime(runtime).capability_requirements
    if capability not in requirements:
        return [f"unknown capability {capability!r}"]
    have = manifest.get("interfaces", {})
    return [i for i in requirements[capability] if not have.get(i)]


def certify(
    manifest: Mapping[str, Any], pins: Mapping[str, str], capabilities: Iterable[str],
    runtime: RuntimeDescription | None = None,
) -> dict[str, Any]:
    """Refuse certification on any pin mismatch or missing interface."""

    runtime = _runtime(runtime)
    problems = check_manifest(manifest, pins, runtime)
    for cap in capabilities:
        missing = missing_interfaces(manifest, cap, runtime)
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
    parser.add_argument("--backend", default=DEFAULT_BACKEND, choices=sorted(BACKENDS))
    args = parser.parse_args(argv)
    runtime = runtime_description(args.backend)
    if args.check:
        manifest = json.loads(args.check.read_text())
    else:
        manifest = collect(image=args.image, runtime=runtime)
        text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        (args.out.write_text(text) if args.out else sys.stdout.write(text))
    try:
        result = certify(manifest, expected_pins(runtime), args.capability, runtime)
    except ManifestMismatch as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

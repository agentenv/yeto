#!/usr/bin/env python3
"""Build the single overlay layer of the yeto miles-ports image.

The base (radixark/miles, upstream docker/Dockerfile) already installs Miles
editable from /root/miles and SGLang editable from
/sgl-workspace/sglang/python.  The pinned forks differ from the base's
checkouts only in pure-Python files (checked below), so the layer:

* replaces both source trees with ``git archive`` of the pinned fork commits
  (every tracked file rewritten; no fork deletes a base file -- asserted),
* replaces each ``.git`` with a one-commit shallow clone whose origin is the
  fork repository (opaque whiteout hides the base history),
* rewrites SGLang's editable-install metadata (dist-info, .pth, finder,
  ``sglang/_version.py``) for the fork's setuptools-scm version,
* unpacks extra pure-Python wheels (``--wheel``, e.g. peft) into the base's
  site-packages, as ``pip install --no-deps`` would,
* replaces the base's ``megatron/bridge/__init__.py`` (Megatron-Bridge
  8cd3466) with ``megatron_bridge_init.py``: the same file without its eager
  ``import modelopt.torch`` (rl-trainer-cold-start-cost L2.2),
* rewrites the Megatron-LM files (/root/Megatron-LM, base commit a84b105)
  touched by ``megatron_lm_lazy_modelopt.patch`` -- the patch applied to the
  commit's files from ``--megatron-lm-src`` -- so a trainer process does not
  import modelopt unless a model uses it (L2.2),
* adds /opt/yeto/image-manifest.json with every component SHA.

Approximate docker equivalent (unverified): docker/miles-ports/Dockerfile.
Rebuilds are file-content equivalent, not byte-identical: entries are
root-owned, fixed-mtime and sorted, .git/hooks samples are dropped, but
.git/index (stat data) and the fetched pack are regenerated each build.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path

SITE = "opt/sglang/lib/python3.12/site-packages"
# A fork may differ from the base's checkout only in Python modules of the
# editable-installed packages (plus tests/docs).  Anything else -- setup.py,
# pyproject.toml, setup.cfg, requirements*, sgl-kernel/**, rust/**, csrc,
# .cu/.cpp -- changes what a real install builds and needs a docker build.
# Not installed and not imported by the package: tests, docs, and standalone
# scripts under tools/ (e.g. Miles tools/convert_hf_to_fp8.py), which the
# layer still ships as plain files in the archived tree.
EXEMPT_PREFIXES = ("test/", "tests/", "docs/", "tools/")
# Megatron-Bridge as the base installs it (pip, not editable) and the file the
# layer writes over its package __init__ (see megatron_bridge_init.py).
MEGATRON_BRIDGE_COMMIT = "8cd3466d14d2337c8492827b3712482c2b3e4866"
MEGATRON_BRIDGE_INIT = Path(__file__).with_name("megatron_bridge_init.py")
# Megatron-LM as the base checks it out (/root/Megatron-LM, on sys.path) and
# the patch the layer applies to it (see the comments it adds).
MEGATRON_LM_COMMIT = "a84b105473eca9c9c6e49dbf1c8aa460c5cf796e"
MEGATRON_LM_PATCH = Path(__file__).with_name("megatron_lm_lazy_modelopt.patch")


def git(repo: Path, *argv: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *argv], check=True,
                          capture_output=True, text=True).stdout


def check_pure_python(repo: Path, base: str, commit: str, packages: tuple[str, ...]) -> list[str]:
    status = git(repo, "diff", "--name-status", "--no-renames", base, commit).splitlines()
    names = []
    for line in status:
        kind, name = line.split("\t", 1)
        if kind == "D":
            raise SystemExit(f"{name}: fork deletes a base file; add a whiteout first")
        in_package = name.startswith(packages) and name.endswith(".py")
        if not (in_package or name.startswith(EXEMPT_PREFIXES) or name.endswith(".md")):
            raise SystemExit(f"{name}: not a pure-Python package change; needs a real (docker) build")
        names.append(name)
    return names


def shallow_clone(src: Path, commit: str, origin: str, dest: Path) -> None:
    dest.mkdir(parents=True)
    run = lambda *a: subprocess.run(["git", "-C", str(dest), *a], check=True,
                                    capture_output=True)
    run("init", "-q")
    run("fetch", "-q", "--depth", "1", f"file://{src}", commit)
    run("-c", "advice.detachedHead=false", "checkout", "-q", "--detach", commit)
    run("remote", "add", "origin", origin)
    (dest / ".git" / "FETCH_HEAD").unlink(missing_ok=True)
    if git(dest, "rev-parse", "HEAD").strip() != commit:
        raise SystemExit(f"{dest}: HEAD mismatch")
    if git(dest, "status", "--porcelain", "--untracked-files=all").strip():
        raise SystemExit(f"{dest}: not clean")


def record_line(name: str, data: bytes | None) -> str:
    if data is None:
        return f"{name},,"
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return f"{name},sha256={digest},{len(data)}"


def version_file(version: str) -> bytes:
    parts = version.replace("+", ".").split(".")
    tup = tuple(int(p) if p.isdigit() else p for p in parts)
    return (
        "# file generated by setuptools-scm\n"
        "# don't change, don't track in version control\n"
        '__all__ = ["__version__", "__version_tuple__", "version", "version_tuple"]\n'
        f"__version__ = version = {version!r}\n"
        f"__version_tuple__ = version_tuple = {tup!r}\n"
    ).encode()


def wheel_files(wheel: Path) -> dict[str, bytes]:
    """``{site-packages relative path: bytes}`` of a pure-Python wheel.

    Only ``*-none-any`` wheels without a ``.data`` directory are accepted:
    those install by unpacking alone (no scripts, headers or compiled code).
    An ``INSTALLER`` file is added, as pip writes one.
    """

    if not wheel.name.endswith("-none-any.whl"):
        raise SystemExit(f"{wheel.name}: only pure-Python (*-none-any) wheels are supported")
    files: dict[str, bytes] = {}
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.endswith("/"):
                continue
            if ".data/" in name.split(".dist-info/")[0]:
                raise SystemExit(f"{wheel.name}: has a .data directory ({name}); not supported")
            files[name] = archive.read(name)
    dist_info = {n.split("/")[0] for n in files if n.split("/")[0].endswith(".dist-info")}
    if len(dist_info) != 1:
        raise SystemExit(f"{wheel.name}: expected one .dist-info directory, found {sorted(dist_info)}")
    files[f"{dist_info.pop()}/INSTALLER"] = b"yeto-make-layer\n"
    return files


def patched_files(src: Path, commit: str, patch: Path) -> dict[str, bytes]:
    """``{repo path: bytes}`` of the files ``patch`` touches, as ``commit`` plus the patch."""
    paths = [line.split("\t")[-1] for line in subprocess.run(
        ["git", "apply", "--numstat", str(patch)], check=True, capture_output=True, text=True
    ).stdout.splitlines()]
    with tempfile.TemporaryDirectory(prefix="img-patch-") as tmp:
        archive = subprocess.run(["git", "-C", str(src), "archive", commit, *paths],
                                 check=True, capture_output=True).stdout
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(tmp, filter="data")
        subprocess.run(["git", "apply", str(patch.resolve())], cwd=tmp, check=True, capture_output=True)
        return {p: Path(tmp, p).read_bytes() for p in paths}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest-json", required=True, help="image-manifest.json content file")
    p.add_argument("--miles-src", type=Path, required=True)
    p.add_argument("--miles-commit", required=True)
    p.add_argument("--miles-base-commit", required=True, help="commit in the base image")
    p.add_argument("--miles-origin", required=True)
    p.add_argument("--sglang-src", type=Path, required=True)
    p.add_argument("--sglang-commit", required=True)
    p.add_argument("--sglang-base-commit", required=True, help="commit in the base image")
    p.add_argument("--sglang-origin", required=True)
    p.add_argument("--sglang-base-version", required=True)
    p.add_argument("--sglang-version", required=True)
    p.add_argument("--sglang-base-metadata", type=Path, required=True,
                   help="dir with the base image's sglang site-packages metadata")
    p.add_argument("--wheel", type=Path, action="append", default=[],
                   help="pure-Python wheel unpacked into site-packages (repeatable)")
    p.add_argument("--megatron-lm-src", type=Path, required=True,
                   help=f"clone containing Megatron-LM {MEGATRON_LM_COMMIT[:7]} (the base's checkout)")
    p.add_argument("--mtime", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    check_pure_python(a.miles_src, a.miles_base_commit, a.miles_commit, ("miles/", "miles_plugins/"))
    check_pure_python(a.sglang_src, a.sglang_base_commit, a.sglang_commit, ("python/sglang/",))

    entries: dict[str, tuple[str, bytes | None, int, str | None]] = {}

    # Only directories we own get entries: the base's /root, /opt, ...
    # keep their modes (a parent entry would reset e.g. /root's 0700).
    def add_dir(path: str) -> None:
        entries.setdefault(path.strip("/") + "/", ("dir", None, 0o755, None))

    def add_file(path: str, data: bytes, mode: int = 0o644) -> None:
        entries[path] = ("file", data, mode, None)

    def add_tree(root: Path, dest: str) -> None:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames.sort()
            rel = os.path.relpath(dirpath, root)
            ddest = dest if rel == "." else f"{dest}/{rel}"
            add_dir(ddest)
            for name in sorted(filenames):
                full = Path(dirpath, name)
                target = f"{ddest}/{name}"
                if full.is_symlink():
                    entries[target] = ("sym", None, 0o777, os.readlink(full))
                else:
                    mode = 0o755 if os.access(full, os.X_OK) else 0o644
                    add_file(target, full.read_bytes(), mode)
            for name in dirnames:
                full = Path(dirpath, name)
                if full.is_symlink():
                    entries[f"{ddest}/{name}"] = ("sym", None, 0o777, os.readlink(full))

    with tempfile.TemporaryDirectory(prefix="img-layer-") as tmp:
        for src, commit, origin, dest in (
            (a.miles_src, a.miles_commit, a.miles_origin, "root/miles"),
            (a.sglang_src, a.sglang_commit, a.sglang_origin, "sgl-workspace/sglang"),
        ):
            clone = Path(tmp, dest.replace("/", "_"))
            shallow_clone(src, commit, origin, clone)
            for hook in (clone / ".git" / "hooks").glob("*.sample"):
                hook.unlink()
            add_tree(clone, dest)
            # hide the base image's .git (other history/origin) entirely
            add_file(f"{dest}/.git/.wh..wh..opq", b"", 0o644)

    # SGLang editable metadata: same install, fork version.
    old, new = a.sglang_base_version, a.sglang_version
    old_mod = "__editable___sglang_" + old.replace(".", "_").replace("+", "_")
    new_mod = "__editable___sglang_" + new.replace(".", "_").replace("+", "_")
    meta = a.sglang_base_metadata
    subst = lambda b: b.replace(old.encode(), new.encode()).replace(old_mod.encode(), new_mod.encode())
    for stale in (f"__editable__.sglang-{old}.pth", f"{old_mod}_finder.py",
                  f"sglang-{old}.dist-info", f"__pycache__/{old_mod}_finder.cpython-312.pyc"):
        d, n = os.path.split(f"{SITE}/{stale}")
        add_file(f"{d}/.wh.{n}", b"")
    files = {
        f"__editable__.sglang-{new}.pth": subst((meta / f"__editable__.sglang-{old}.pth").read_bytes()),
        f"{new_mod}_finder.py": subst((meta / f"{old_mod}_finder.py").read_bytes()),
    }
    dist = f"sglang-{new}.dist-info"
    for name in sorted(os.listdir(meta / f"sglang-{old}.dist-info")):
        if name != "RECORD":
            files[f"{dist}/{name}"] = subst((meta / f"sglang-{old}.dist-info" / name).read_bytes())
    old_record = (meta / f"sglang-{old}.dist-info" / "RECORD").read_text().splitlines()
    record = [l for l in old_record if l.startswith("../")]  # bin/ scripts unchanged
    record += [record_line(n, d) for n, d in sorted(files.items())]
    record += [f"__pycache__/{new_mod}_finder.cpython-312.pyc,,", f"{dist}/RECORD,,"]
    files[f"{dist}/RECORD"] = ("\n".join(sorted(record)) + "\n").encode()
    for name, data in files.items():
        add_file(f"{SITE}/{name}", data)
    add_file("sgl-workspace/sglang/python/sglang/_version.py", version_file(new))

    for wheel in a.wheel:
        for name, data in sorted(wheel_files(wheel).items()):
            add_file(f"{SITE}/{name}", data)

    add_file(f"{SITE}/megatron/bridge/__init__.py", MEGATRON_BRIDGE_INIT.read_bytes())
    for name, data in patched_files(a.megatron_lm_src, MEGATRON_LM_COMMIT, MEGATRON_LM_PATCH).items():
        add_file(f"root/Megatron-LM/{name}", data)

    add_dir("opt/yeto")
    add_file("opt/yeto/image-manifest.json", Path(a.manifest_json).read_bytes())

    with tarfile.open(a.out, "w", format=tarfile.PAX_FORMAT) as tar:
        for path in sorted(entries):
            kind, data, mode, link = entries[path]
            info = tarfile.TarInfo(path.rstrip("/"))
            info.mtime, info.uid, info.gid, info.uname, info.gname = a.mtime, 0, 0, "root", "root"
            info.mode = mode
            if kind == "dir":
                info.type = tarfile.DIRTYPE
                tar.addfile(info)
            elif kind == "sym":
                info.type, info.linkname = tarfile.SYMTYPE, link
                tar.addfile(info)
            else:
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    print(f"layer {a.out}: {len(entries)} entries")


if __name__ == "__main__":
    main()

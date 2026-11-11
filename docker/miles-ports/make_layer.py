#!/usr/bin/env python3
"""Build the single overlay layer of the yeto miles-ports image.

The base (radixark/miles, upstream docker/Dockerfile) already installs Miles
editable from /root/miles and SGLang editable from
/sgl-workspace/sglang/python.  The pinned forks differ from the base's
checkouts only in pure-Python files (checked below), so the layer:

* replaces both source trees with ``git archive`` of the pinned fork commits
  (every tracked file rewritten; a file the fork deleted becomes a whiteout),
* replaces each ``.git`` with a one-commit shallow clone whose origin is the
  fork repository (opaque whiteout hides the base history),
* rewrites SGLang's editable-install metadata (dist-info, .pth, finder,
  ``sglang/_version.py``) for the fork's setuptools-scm version,
* unpacks extra pure-Python wheels (``--wheel``, e.g. peft) into the base's
  site-packages, as ``pip install --no-deps`` would,
* replaces the base's ``megatron/bridge/__init__.py`` (Megatron-Bridge at
  ``--megatron-bridge-commit``; 8cd3466 and a0c6d6e ship the same file) with
  ``megatron_bridge_init.py``: the same file without its eager
  ``import modelopt.torch``,
* rewrites the Megatron-LM files (/root/Megatron-LM at ``--megatron-lm-commit``,
  the base's checkout) touched by ``megatron_lm_lazy_modelopt.patch`` -- the
  patch applied to the commit's files from ``--megatron-lm-src`` -- so a
  trainer process does not import modelopt unless a model uses it,
* adds /opt/yeto/image-manifest.json with every component SHA, and with
  ``--image-assert-dir`` the version lock the island checks at start
  (docker/image-assert: ``/opt/yeto/image/{assert_image_versions.py,
  entrypoint.sh,versions.lock.json,commits.json}``).

Nothing about the base is written into this file: the base's SGLang
editable-install metadata is read from the directory ``--sglang-base-metadata``
(its version and finder module are whatever ``sglang-*.dist-info`` says), and
the base layer that holds it is found by scanning (``find-layer`` /
``check-listing`` below), not by a layer number.

Helper modes for the build scripts (crane does the registry reads):

* ``make_layer.py find-layer --manifest M.json --config C.json``: the base
  layer indices to try, most likely first (the layers whose build step
  mentions the SGLang checkout and an editable install come first, then the
  rest from the last layer down);
* ``make_layer.py check-listing < names.txt``: exit 0 when a ``tar -t`` listing
  of a layer holds the SGLang editable-install metadata, 1 otherwise.

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
import re
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

SITE = "opt/sglang/lib/python3.12/site-packages"
IMAGE_ASSERT_DIR = "opt/yeto/image"  # where yeto.launch.image_check looks for the lock
# A fork may differ from the base's checkout in Python modules of the
# editable-installed packages and in files that no install step reads
# (tests, docs, examples, CI, recipes, READMEs): the layer ships the fork's
# whole tree as plain files, so those simply come along.  What it must not
# differ in is anything a real install *builds* or *resolves*: native sources
# and their build trees (sgl-kernel/**, rust/**, csrc/**, .cu/.cpp/.so/.rs,
# CMake), or a non-Python file inside an installed package.  Those need a
# docker build.  The install *metadata* (setup.py, setup.cfg, pyproject.toml,
# MANIFEST.in, requirements*) is a third class: the base's dist-info stays as
# it is, so a change there is refused unless the caller states it is only
# metadata (--allow-install-metadata-drift; e.g. a new extras_require entry
# or a requirement the layer itself adds as a wheel) -- recorded in the
# manifest.  A file the fork deletes becomes a whiteout, so the shipped
# checkout is exactly the fork commit (git status clean in the container).
NATIVE_PREFIXES = ("sgl-kernel/", "rust/", "csrc/", "kernels/")
NATIVE_SUFFIXES = (".cu", ".cuh", ".cpp", ".cc", ".c", ".h", ".hpp", ".so", ".rs", ".cmake", ".pyx", ".pyi.in")
INSTALL_METADATA = ("setup.py", "setup.cfg", "pyproject.toml", "MANIFEST.in")


def is_install_metadata(name: str) -> bool:
    base = name.rsplit("/", 1)[-1]
    return base in INSTALL_METADATA or (base.startswith("requirements") and base.endswith(".txt"))


def is_native(name: str) -> bool:
    return name.startswith(NATIVE_PREFIXES) or name.endswith(NATIVE_SUFFIXES) or name.rsplit("/", 1)[-1] == "CMakeLists.txt"
# The file the layer writes over Megatron-Bridge's package __init__ (the base
# installs Bridge with pip, not editable; see megatron_bridge_init.py).
MEGATRON_BRIDGE_INIT = Path(__file__).with_name("megatron_bridge_init.py")
# The patch the layer applies to the base's Megatron-LM checkout
# (/root/Megatron-LM, on sys.path; see the comments it adds).
MEGATRON_LM_PATCH = Path(__file__).with_name("megatron_lm_lazy_modelopt.patch")
IMAGE_ASSERT_SRC = Path(__file__).resolve().parent.parent / "image-assert"

_DIST_INFO = re.compile(r"^sglang-(?P<version>\d[^/]*)\.dist-info$")  # not sglang-router-*
_FINDER = re.compile(r"^__editable___sglang_(?P<module>.+)_finder\.py$")


def editable_module_name(version: str) -> str:
    """The setuptools editable finder module of ``sglang==version``."""
    return "__editable___sglang_" + version.replace(".", "_").replace("+", "_").replace("-", "_")


def sglang_metadata_names(names) -> dict[str, str] | None:
    """The base's SGLang editable-install files among ``names`` (paths of a layer or
    a directory listing, with or without the site-packages prefix): ``version``,
    ``dist_info``, ``pth`` and ``finder`` (file names); None when the listing has no
    ``sglang-*.dist-info``."""
    found: dict[str, str] = {}
    for raw in names:
        name = str(raw).strip().lstrip("./")
        if name.startswith(SITE + "/"):
            name = name[len(SITE) + 1:]
        head = name.split("/", 1)[0]
        dist = _DIST_INFO.match(head)
        if dist and "dist_info" not in found:
            found["dist_info"], found["version"] = head, dist["version"]
        elif "/" not in name:
            if name.startswith("__editable__.sglang-") and name.endswith(".pth"):
                found["pth"] = name
            elif _FINDER.match(name):
                found["finder"] = name
    if "dist_info" not in found:
        return None
    found.setdefault("pth", f"__editable__.sglang-{found['version']}.pth")
    found.setdefault("finder", f"{editable_module_name(found['version'])}_finder.py")
    return found


def layer_holds_sglang_metadata(names) -> bool:
    return sglang_metadata_names(names) is not None


_SGLANG_STEP_HINTS = ("/sgl-workspace/sglang", "sglang_deps", "pip install -e")


def candidate_metadata_layers(manifest: dict, config: dict) -> list[int]:
    """Indices into ``manifest["layers"]`` to scan for the SGLang editable-install
    metadata, most likely first: the layers whose build step (image config
    history, non-empty entries map 1:1 onto the layers) mentions the SGLang
    checkout and an editable install, newest first; then every other layer
    from the last one down (the newest write wins in an image, so the last
    layer that holds the files is the one that matters)."""
    layers = manifest.get("layers") or []
    steps = [h for h in config.get("history") or [] if not h.get("empty_layer")]
    if steps and len(steps) != len(layers):
        steps = []  # a history that does not line up is no hint at all
    scored: list[tuple[int, int]] = []
    for index in range(len(layers)):
        created = str(steps[index].get("created_by", "")) if steps else ""
        score = 0
        if "/sgl-workspace/sglang" in created and "install -e" in created:
            score = 2
        elif "sglang" in created and "install" in created:
            score = 1
        scored.append((score, index))
    scored.sort(key=lambda item: (-item[0], -item[1]))
    return [index for _score, index in scored]


def _helper_mode(argv: list[str]) -> int | None:
    if not argv or argv[0] not in ("find-layer", "check-listing"):
        return None
    if argv[0] == "check-listing":
        return 0 if layer_holds_sglang_metadata(sys.stdin.read().splitlines()) else 1
    p = argparse.ArgumentParser(prog="make_layer.py find-layer")
    p.add_argument("--manifest", type=Path, required=True, help="crane manifest of the base (amd64)")
    p.add_argument("--config", type=Path, required=True, help="crane config of the base (amd64)")
    a = p.parse_args(argv[1:])
    manifest = json.loads(a.manifest.read_text())
    config = json.loads(a.config.read_text())
    order = candidate_metadata_layers(manifest, config)
    print(json.dumps([{"index": i, "digest": manifest["layers"][i]["digest"],
                       "size": manifest["layers"][i].get("size")} for i in order]))
    return 0


def git(repo: Path, *argv: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *argv], check=True,
                          capture_output=True, text=True).stdout


def check_pure_python(repo: Path, base: str, commit: str, packages: tuple[str, ...],
                      *, allow_install_metadata_drift: bool = False) -> tuple[list[str], list[str], list[str]]:
    """``(changed, deleted, install_metadata)`` paths of ``base..commit``; exits when
    the fork touches a native source or a non-Python file of an installed package
    (``packages``), or install metadata without ``allow_install_metadata_drift``."""
    status = git(repo, "diff", "--name-status", "--no-renames", base, commit).splitlines()
    changed: list[str] = []
    deleted: list[str] = []
    metadata: list[str] = []
    for line in status:
        kind, name = line.split("\t", 1)
        if is_native(name):
            raise SystemExit(f"{name}: native source changed; needs a real (docker) build")
        if name.startswith(packages) and not (name.endswith(".py") or name.endswith(".md") or name.endswith(".txt")
                                              or name.endswith(".json") or name.endswith(".yaml") or name.endswith(".yml")):
            raise SystemExit(f"{name}: not a pure-Python package change; needs a real (docker) build")
        if is_install_metadata(name):
            if not allow_install_metadata_drift:
                raise SystemExit(f"{name}: install metadata differs from the base's (its dist-info is kept); "
                                 "pass --allow-install-metadata-drift when the change is metadata only")
            metadata.append(name)
        if kind == "D":
            deleted.append(name)
        else:
            changed.append(name)
    return changed, deleted, metadata


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


def wheel_files(wheel: Path, *, platform: str | None = None) -> dict[str, bytes]:
    """``{site-packages relative path: bytes}`` of a wheel that installs by
    unpacking alone (no scripts, headers or ``.data`` directory).

    ``*-none-any`` wheels (pure Python) are always accepted. With ``platform``
    (``manylinux_x86_64`` style suffix of the image's platform) a binary wheel of
    exactly that platform is accepted too: a ``*-abi3-manylinux*_x86_64`` wheel
    such as pyrage unpacks to ``.so`` files that pip would copy as they are.
    An ``INSTALLER`` file is added, as pip writes one.
    """

    if not wheel.name.endswith("-none-any.whl"):
        if platform is None or not re.search(rf"-(?:cp3\d+|abi3|py3)-manylinux[\w.]*_{re.escape(platform)}\.whl$",
                                             wheel.name):
            raise SystemExit(f"{wheel.name}: only pure-Python (*-none-any) wheels are supported"
                             + (f", or manylinux {platform} wheels with --wheel-platform" if platform else ""))
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
    p.add_argument("--sglang-version", required=True)
    p.add_argument("--sglang-base-metadata", type=Path, required=True,
                   help="dir with the base image's sglang site-packages metadata (its version is read "
                        "from the sglang-*.dist-info directory found there)")
    p.add_argument("--wheel", type=Path, action="append", default=[],
                   help="wheel unpacked into site-packages (repeatable): pure Python, or a manylinux "
                        "wheel of --wheel-platform")
    p.add_argument("--wheel-platform", default=None,
                   help="accept manylinux binary wheels of this platform (e.g. x86_64) besides pure-Python ones")
    p.add_argument("--replace-base-dist", action="append", default=[], metavar="NAME=VERSION",
                   help="whiteout the base's NAME-VERSION.dist-info (a wheel of the same distribution is "
                        "unpacked over it; with both present importlib.metadata picks whichever lists first)")
    p.add_argument("--megatron-lm-src", type=Path, required=True,
                   help="clone containing the base's Megatron-LM checkout commit (--megatron-lm-commit)")
    p.add_argument("--megatron-lm-commit", required=True, help="Megatron-LM commit the base checked out")
    p.add_argument("--megatron-bridge-commit", required=True,
                   help="Megatron-Bridge commit the base installed (recorded; its __init__ is replaced)")
    p.add_argument("--image-assert-dir", type=Path, default=None,
                   help="directory with versions.lock.json and commits.json: shipped with the "
                        "docker/image-assert checker under /opt/yeto/image")
    p.add_argument("--allow-install-metadata-drift", action="store_true",
                   help="accept setup.py/setup.cfg/pyproject.toml/requirements* changes in the forks "
                        "(the base's dist-info is kept; the paths are recorded in the manifest)")
    p.add_argument("--mtime", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    # tools/: stand-alone scripts run from the checkout (e.g. convert_hf_to_fp8.py), nothing built
    deleted: dict[str, list[str]] = {}
    drift: dict[str, list[str]] = {}
    for key, src, base, commit, packages in (
        ("miles", a.miles_src, a.miles_base_commit, a.miles_commit, ("miles/", "miles_plugins/", "tools/")),
        ("sglang", a.sglang_src, a.sglang_base_commit, a.sglang_commit, ("python/sglang/",)),
    ):
        _changed, deleted[key], drift[key] = check_pure_python(
            src, base, commit, packages, allow_install_metadata_drift=a.allow_install_metadata_drift)
    manifest = json.loads(Path(a.manifest_json).read_text())
    for key in ("miles", "sglang"):
        manifest[key]["deleted_from_base"] = deleted[key]
        manifest[key]["install_metadata_drift"] = drift[key]
    Path(a.manifest_json).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

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
            # files the fork deleted: whiteouts, so the checkout is exactly the fork commit
            for name in deleted["miles" if dest == "root/miles" else "sglang"]:
                d, n = os.path.split(f"{dest}/{name}")
                add_file(f"{d}/.wh.{n}", b"")

    # SGLang editable metadata: same install, fork version. The base's version
    # and finder module come from the files themselves (no version is written here).
    meta = a.sglang_base_metadata
    base_meta = sglang_metadata_names(os.listdir(meta))
    if base_meta is None:
        raise SystemExit(f"{meta}: no sglang-*.dist-info directory (not the base's site-packages metadata)")
    old, new = base_meta["version"], a.sglang_version
    old_finder, old_pth = base_meta["finder"], base_meta["pth"]
    old_mod = old_finder[: -len("_finder.py")]
    new_mod = editable_module_name(new)
    subst = lambda b: b.replace(old.encode(), new.encode()).replace(old_mod.encode(), new_mod.encode())
    for stale in (old_pth, old_finder, base_meta["dist_info"], f"__pycache__/{old_mod}_finder.cpython-312.pyc"):
        d, n = os.path.split(f"{SITE}/{stale}")
        add_file(f"{d}/.wh.{n}", b"")
    files = {
        f"__editable__.sglang-{new}.pth": subst((meta / old_pth).read_bytes()),
        f"{new_mod}_finder.py": subst((meta / old_finder).read_bytes()),
    }
    dist = f"sglang-{new}.dist-info"
    for name in sorted(os.listdir(meta / base_meta["dist_info"])):
        if name != "RECORD":
            files[f"{dist}/{name}"] = subst((meta / base_meta["dist_info"] / name).read_bytes())
    old_record = (meta / base_meta["dist_info"] / "RECORD").read_text().splitlines()
    record = [l for l in old_record if l.startswith("../")]  # bin/ scripts unchanged
    record += [record_line(n, d) for n, d in sorted(files.items())]
    record += [f"__pycache__/{new_mod}_finder.cpython-312.pyc,,", f"{dist}/RECORD,,"]
    files[f"{dist}/RECORD"] = ("\n".join(sorted(record)) + "\n").encode()
    for name, data in files.items():
        add_file(f"{SITE}/{name}", data)
    add_file("sgl-workspace/sglang/python/sglang/_version.py", version_file(new))

    for wheel in a.wheel:
        for name, data in sorted(wheel_files(wheel, platform=a.wheel_platform).items()):
            mode = 0o755 if name.endswith(".so") or "/bin/" in name else 0o644
            add_file(f"{SITE}/{name}", data, mode)
    for spec in a.replace_base_dist:
        dist, _, version = spec.partition("=")
        if not dist or not version:
            raise SystemExit(f"--replace-base-dist {spec!r}: expected NAME=VERSION")
        add_file(f"{SITE}/.wh.{dist.replace('-', '_')}-{version}.dist-info", b"")

    add_file(f"{SITE}/megatron/bridge/__init__.py", MEGATRON_BRIDGE_INIT.read_bytes())
    for name, data in patched_files(a.megatron_lm_src, a.megatron_lm_commit, MEGATRON_LM_PATCH).items():
        add_file(f"root/Megatron-LM/{name}", data)

    add_dir("opt/yeto")
    add_file("opt/yeto/image-manifest.json", Path(a.manifest_json).read_bytes())
    if a.image_assert_dir is not None:
        add_dir(IMAGE_ASSERT_DIR)
        for name in ("versions.lock.json", "commits.json"):
            add_file(f"{IMAGE_ASSERT_DIR}/{name}", (a.image_assert_dir / name).read_bytes())
        add_file(f"{IMAGE_ASSERT_DIR}/assert_image_versions.py",
                 (IMAGE_ASSERT_SRC / "assert_image_versions.py").read_bytes(), 0o755)
        add_file(f"{IMAGE_ASSERT_DIR}/entrypoint.sh", (IMAGE_ASSERT_SRC / "entrypoint.sh").read_bytes(), 0o755)

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
    code = _helper_mode(sys.argv[1:])
    if code is None:
        main()
    else:
        sys.exit(code)

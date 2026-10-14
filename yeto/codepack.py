"""Run code packs (rl-identity-tiering L9.4-L9.6, development mode).

A code pack fixes the code of one run. The launcher makes it once, at the first
start of the run, from the local git work tree:

* ``commit``: ``git rev-parse HEAD``;
* the dirty patch: ``git diff --binary --cached HEAD`` over a temporary index that
  holds every tracked and every untracked, not ignored file (the real index is not
  touched); ``patch_sha`` is its sha256, ``dirty`` is true when it is not empty;
* a deterministic tar of the resulting tree (sorted entries, zero times and owners,
  modes reduced to 0644/0755), with :data:`PACK_INFO` as the first entry;
* ``digest``: the sha256 of the tar bytes.

The tar is uploaded to the run's checkpoint store under ``code_packs/<digest>.tar``
and recorded in the run manifest. Every later start of the run (resume, island
replacement, re-JOIN) takes the same pack from the manifest, unpacks it into a local
cache and uses that directory as the island workdir. The island checks the unpacked
tree against the digest before it starts (:func:`verify_tree`) and refuses to start
on a mismatch. The pack changes only through :func:`upgrade_record` (explicit
upgrade): it requires an equal semantic hash and appends "pack A -> pack B" to the
manifest record.

Storage backends are pluggable (:func:`register_pack_store`); this module holds the
neutral core and a directory backend. Cloud backends live with their cloud code.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Protocol

CODE_PACK_MODE_GIT_PATCH = "git-patch"
PACK_INFO = "CODEPACK.json"
PACK_DIR = "code_packs"
PACK_FORMAT = 1
# Never packed: the same paths the Modal workdir mount drops, plus Python caches.
PACK_EXCLUDE = (".git", ".venv", ".claude", "syncer/target", "RLinf-main",
                "**/__pycache__", "**/*.pyc")


class CodePackError(RuntimeError):
    """A pack cannot be made, found, or does not match its digest."""


# ---------------------------------------------------------------------------
# pack record
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CodePack:
    """What the run manifest records for one pack."""

    commit: str
    dirty: bool
    patch_sha: str
    digest: str
    pack_uri: str | None = None
    mode: str = CODE_PACK_MODE_GIT_PATCH
    upgrades: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def record(self) -> dict[str, Any]:
        out = asdict(self)
        out["upgrades"] = [dict(u) for u in self.upgrades]
        return out

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "CodePack":
        if record.get("mode") != CODE_PACK_MODE_GIT_PATCH:
            raise CodePackError(f"code pack mode {record.get('mode')!r} is not {CODE_PACK_MODE_GIT_PATCH!r}")
        return cls(commit=str(record["commit"]), dirty=bool(record["dirty"]),
                   patch_sha=str(record["patch_sha"]), digest=str(record["digest"]),
                   pack_uri=record.get("pack_uri"), mode=str(record["mode"]),
                   upgrades=tuple(dict(u) for u in record.get("upgrades") or ()))


# ---------------------------------------------------------------------------
# building
# ---------------------------------------------------------------------------
def _git(repo: Path, *argv: str, env: Mapping[str, str] | None = None) -> bytes:
    try:
        return subprocess.run(["git", *argv], cwd=repo, check=True, capture_output=True,
                              env=None if env is None else {**os.environ, **env}).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        detail = getattr(e, "stderr", b"") or b""
        raise CodePackError(f"git {' '.join(argv)} failed in {repo}: "
                            f"{detail.decode(errors='replace').strip() or e}") from e


def _excluded(path: str) -> bool:
    parts = PurePosixPath(path).parts
    for pattern in PACK_EXCLUDE:
        if pattern.startswith("**/"):
            leaf = pattern[3:]
            if any(fnmatch.fnmatch(part, leaf) for part in parts):
                return True
        elif path == pattern or path.startswith(pattern + "/"):
            return True
    return False


def pack_file_list(repo: str | os.PathLike[str]) -> list[str]:
    """Tracked and untracked, not ignored files that exist now, sorted, minus
    :data:`PACK_EXCLUDE`."""

    raw = _git(Path(repo), "ls-files", "-z", "--cached", "--others", "--exclude-standard")
    names = {name for name in raw.decode("utf-8").split("\0") if name}
    root = Path(repo)
    return sorted(n for n in names
                  if not _excluded(n) and ((root / n).is_symlink() or (root / n).is_file()))


def dirty_patch(repo: str | os.PathLike[str]) -> bytes:
    """``HEAD`` → work tree as one binary patch, untracked (not ignored) files
    included. Uses a temporary index; the repository index is not changed."""

    root = Path(repo)
    git_dir = Path(_git(root, "rev-parse", "--absolute-git-dir").decode().strip())
    with tempfile.TemporaryDirectory() as tmp:
        index = Path(tmp) / "index"
        if (git_dir / "index").is_file():
            shutil.copyfile(git_dir / "index", index)
        env = {"GIT_INDEX_FILE": str(index)}
        _git(root, "add", "-A", "--", ".", env=env)
        return _git(root, "diff", "--binary", "--cached", "--no-color", "--no-ext-diff",
                    "HEAD", "--", ".", env=env)


def _tar_bytes(root: Path, files: Iterable[str], info: Mapping[str, Any]) -> bytes:
    """The deterministic tar of ``files`` under ``root``, :data:`PACK_INFO` first."""

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        def add(name: str, *, data: bytes | None = None, link: str | None = None,
                mode: int = 0o644) -> None:
            entry = tarfile.TarInfo(name)
            entry.mtime = 0
            entry.uid = entry.gid = 0
            entry.uname = entry.gname = ""
            entry.mode = mode
            if link is not None:
                entry.type = tarfile.SYMTYPE
                entry.linkname = link
                tar.addfile(entry)
            else:
                entry.size = len(data or b"")
                tar.addfile(entry, io.BytesIO(data or b""))

        add(PACK_INFO, data=(json.dumps(info, sort_keys=True, indent=1) + "\n").encode())
        for name in files:
            path = root / name
            if path.is_symlink():
                add(name, link=os.readlink(path), mode=0o777)
            else:
                executable = os.stat(path).st_mode & 0o111
                add(name, data=path.read_bytes(), mode=0o755 if executable else 0o644)
    return buf.getvalue()


def build_pack(repo: str | os.PathLike[str]) -> tuple[CodePack, bytes]:
    """Make the pack of the work tree at ``repo``: ``(record without URI, tar bytes)``."""

    root = Path(repo)
    commit = _git(root, "rev-parse", "HEAD").decode().strip()
    patch = dirty_patch(root)
    files = pack_file_list(root)
    patch_sha = hashlib.sha256(patch).hexdigest()
    info = {"format": PACK_FORMAT, "mode": CODE_PACK_MODE_GIT_PATCH, "commit": commit,
            "dirty": bool(patch), "patch_sha": patch_sha, "files": files}
    data = _tar_bytes(root, files, info)
    pack = CodePack(commit=commit, dirty=bool(patch), patch_sha=patch_sha,
                    digest=hashlib.sha256(data).hexdigest())
    return pack, data


# ---------------------------------------------------------------------------
# checking and unpacking
# ---------------------------------------------------------------------------
def read_pack_info(tree: str | os.PathLike[str]) -> dict[str, Any]:
    path = Path(tree) / PACK_INFO
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise CodePackError(f"{path}: no readable code pack info ({e})") from e


def tree_digest(tree: str | os.PathLike[str]) -> str:
    """Rebuild the tar from an unpacked tree (the files its :data:`PACK_INFO` lists)
    and return its sha256. A missing or changed listed file changes the digest."""

    root = Path(tree)
    info = read_pack_info(root)
    files = list(info.get("files") or [])
    missing = [n for n in files if not ((root / n).is_symlink() or (root / n).is_file())]
    if missing:
        raise CodePackError(f"{root}: {len(missing)} packed file(s) missing, e.g. {missing[0]}")
    return hashlib.sha256(_tar_bytes(root, files, info)).hexdigest()


def verify_tree(tree: str | os.PathLike[str], expected: str) -> str:
    """Refuse a tree whose rebuilt tar does not have the digest ``expected``."""

    actual = tree_digest(tree)
    if actual != expected.lower():
        raise CodePackError(f"code pack digest mismatch: run manifest has {expected.lower()}, "
                            f"{tree} has {actual}")
    return actual


def _safe_member(member: tarfile.TarInfo) -> None:
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts or not (member.isfile() or member.issym()):
        raise CodePackError(f"unsafe code pack entry {member.name!r}")
    if member.issym():
        target = PurePosixPath(member.linkname)
        if target.is_absolute() or ".." in (path.parent / target).parts:
            raise CodePackError(f"code pack symlink {member.name!r} leaves the tree")


def unpack(data: bytes, digest: str, cache_root: str | os.PathLike[str]) -> Path:
    """Check ``data`` against ``digest`` and unpack it to ``<cache_root>/<digest>``
    (reused when already there and intact). Returns the tree."""

    if hashlib.sha256(data).hexdigest() != digest.lower():
        raise CodePackError(f"downloaded code pack does not have digest {digest}")
    dest = Path(cache_root).expanduser() / digest.lower()
    if dest.is_dir():
        try:
            verify_tree(dest, digest)
            return dest
        except CodePackError:
            shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{digest[:12]}-", dir=dest.parent))
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tar:
            members = tar.getmembers()
            for member in members:
                _safe_member(member)
            tar.extractall(staging, members=members, filter="tar")
        verify_tree(staging, digest)
        staging.rename(dest)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return dest


# ---------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------
class PackStore(Protocol):
    def put(self, digest: str, data: bytes) -> str:
        """Store the pack; return its URI."""

    def get(self, uri: str) -> bytes:
        """Return the bytes stored at ``uri``."""


class DirPackStore:
    """A checkpoint store the launcher sees as a directory (local path, a shared
    file system mounted on the launcher too, or a test double)."""

    def __init__(self, root: str | os.PathLike[str], uri_base: str | None = None) -> None:
        self.root = Path(root).expanduser()
        self.uri_base = (uri_base or str(self.root)).rstrip("/")

    def _path(self, uri: str) -> Path:
        if not uri.startswith(self.uri_base + "/"):
            raise CodePackError(f"{uri!r} is not in store {self.uri_base!r}")
        return self.root / uri[len(self.uri_base) + 1:]

    def put(self, digest: str, data: bytes) -> str:
        uri = f"{self.uri_base}/{PACK_DIR}/{digest}.tar"
        path = self._path(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tar.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
        return uri

    def get(self, uri: str) -> bytes:
        try:
            return self._path(uri).read_bytes()
        except OSError as e:
            raise CodePackError(f"code pack {uri} not readable: {e}") from e


_PACK_STORES: dict[str, Callable[[str], PackStore]] = {}


def register_pack_store(scheme: str, factory: Callable[[str], PackStore]) -> None:
    """Register the pack store for checkpoint store URIs ``<scheme>://...``. The
    factory receives the whole store URI."""

    _PACK_STORES[scheme] = factory


def pack_store_for(store_uri: str) -> PackStore:
    """The pack store for ``--rl-checkpoint-store``. A path must be visible on the
    launcher (a share mounted there too); a ``scheme://`` URI needs a registered
    backend."""

    value = str(store_uri)
    if "://" in value:
        scheme = value.split("://", 1)[0]
        if scheme not in _PACK_STORES:
            _register_builtin_stores()
        if scheme not in _PACK_STORES:
            raise CodePackError(f"no code pack store for {scheme}:// (store {value!r})")
        return _PACK_STORES[scheme](value)
    root = Path(value).expanduser()
    if not root.is_dir():
        raise CodePackError(f"checkpoint store path {value!r} is not visible on the launcher; "
                            "the code pack needs a bucket, a modal-volume:// store, or the "
                            "shared file system mounted on the launcher too")
    return DirPackStore(root, uri_base=value.rstrip("/"))


class _S3PackStore:
    """``s3://bucket[/prefix]`` (boto3; also S3-compatible endpoints via the usual
    AWS environment)."""

    def __init__(self, store_uri: str) -> None:
        self.base = store_uri.rstrip("/")

    @staticmethod
    def _split(uri: str) -> tuple[str, str]:
        bucket, _, key = uri.split("://", 1)[1].partition("/")
        return bucket, key

    def put(self, digest: str, data: bytes) -> str:
        import boto3

        uri = f"{self.base}/{PACK_DIR}/{digest}.tar"
        bucket, key = self._split(uri)
        boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=data)
        return uri

    def get(self, uri: str) -> bytes:
        import boto3

        bucket, key = self._split(uri)
        return boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()


class _ModalVolumePackStore:
    """``modal-volume://NAME[/PREFIX]`` (Modal SDK)."""

    def __init__(self, store_uri: str) -> None:
        from .cloud.modal_ckpt_store import parse_modal_volume_uri

        self.base = store_uri.rstrip("/")
        self.name, self.prefix = parse_modal_volume_uri(store_uri)

    def _volume(self, *, create: bool = False):
        import modal

        return modal.Volume.from_name(self.name, create_if_missing=create)

    def put(self, digest: str, data: bytes) -> str:
        # The first start stores the pack before the island creates the run's
        # checkpoint volume (modal_runner, create_if_missing), so create it here too.
        rel = "/".join(p for p in (self.prefix, PACK_DIR, f"{digest}.tar") if p)
        with self._volume(create=True).batch_upload(force=True) as batch:
            batch.put_file(io.BytesIO(data), "/" + rel)
        return f"{self.base}/{PACK_DIR}/{digest}.tar"

    def get(self, uri: str) -> bytes:
        rel = uri[len(self.base):].lstrip("/")
        rel = "/".join(p for p in (self.prefix, rel) if p)
        return b"".join(self._volume().read_file("/" + rel))


def _register_builtin_stores() -> None:
    _PACK_STORES.setdefault("s3", _S3PackStore)
    _PACK_STORES.setdefault("modal-volume", _ModalVolumePackStore)


# ---------------------------------------------------------------------------
# run level
# ---------------------------------------------------------------------------
def make_and_store(repo: str | os.PathLike[str], store: PackStore) -> tuple[CodePack, bytes]:
    pack, data = build_pack(repo)
    uri = store.put(pack.digest, data)
    return CodePack(commit=pack.commit, dirty=pack.dirty, patch_sha=pack.patch_sha,
                    digest=pack.digest, pack_uri=uri), data


def fetch_tree(pack: CodePack, store: PackStore, cache_root: str | os.PathLike[str]) -> Path:
    """Download the manifest's pack and unpack it (checked) into the cache."""

    if not pack.pack_uri:
        raise CodePackError("code pack record has no pack_uri")
    return unpack(store.get(pack.pack_uri), pack.digest, cache_root)


def upgrade_record(old: CodePack, new: CodePack, *, old_semantic: str | None,
                   new_semantic: str | None, reason: str,
                   now: float | None = None) -> CodePack:
    """Explicit upgrade (L9.6): allowed only when both semantic hashes are known and
    equal. Returns ``new`` carrying the old history plus "old → new"."""

    if not reason.strip():
        raise CodePackError("a code pack upgrade needs a reason")
    if not old_semantic or not new_semantic:
        raise CodePackError("code pack upgrade refused: the semantic hash of the old or the "
                            "new algorithm spec is unknown")
    if old_semantic != new_semantic:
        raise CodePackError(f"code pack upgrade refused: semantic hash changes "
                            f"({old_semantic} -> {new_semantic}); start a new run instead")
    step = {"from": old.digest, "to": new.digest, "from_commit": old.commit,
            "to_commit": new.commit, "semantic_sha256": new_semantic, "reason": reason,
            "unix": time.time() if now is None else now}
    return CodePack(commit=new.commit, dirty=new.dirty, patch_sha=new.patch_sha,
                    digest=new.digest, pack_uri=new.pack_uri,
                    upgrades=(*old.upgrades, step))


@dataclass(frozen=True)
class RunPack:
    pack: CodePack
    tree: Path
    reused: bool


def resolve_run_pack(*, previous: Mapping[str, Any] | None, repo: str | os.PathLike[str],
                     store: PackStore, cache_root: str | os.PathLike[str],
                     semantic_sha256: str | None, upgrade_reason: str | None = None,
                     log: Callable[[str], None] = print) -> RunPack:
    """The pack this start of the run uses.

    * no pack in ``previous`` (the old run manifest): make, store and unpack one;
    * a pack in ``previous``: take that same pack (resume / replacement), even when
      the local tree changed since;
    * with ``upgrade_reason``: make a pack from the local tree and switch to it
      only under :func:`upgrade_record`'s rule."""

    old_record = (previous or {}).get("code_pack")
    if not old_record:
        pack, data = make_and_store(repo, store)
        log(f"[codepack] new run pack {pack.digest[:16]} (commit {pack.commit[:12]}"
            f"{', dirty' if pack.dirty else ''}) -> {pack.pack_uri}")
        return RunPack(pack, unpack(data, pack.digest, cache_root), reused=False)
    old = CodePack.from_record(old_record)
    if upgrade_reason is not None:
        candidate, data = make_and_store(repo, store)
        if candidate.digest == old.digest:
            log("[codepack] upgrade requested but the local tree equals the run pack")
            return RunPack(old, fetch_tree(old, store, cache_root), reused=True)
        new = upgrade_record(old, candidate, old_semantic=(previous or {}).get("semantic_sha256"),
                             new_semantic=semantic_sha256, reason=upgrade_reason)
        log(f"[codepack] upgrade {old.digest[:16]} -> {new.digest[:16]}: {upgrade_reason}")
        return RunPack(new, unpack(data, new.digest, cache_root), reused=False)
    log(f"[codepack] reusing run pack {old.digest[:16]} from the run manifest")
    return RunPack(old, fetch_tree(old, store, cache_root), reused=True)


# ---------------------------------------------------------------------------
# CLI (island setup / operators)
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m yeto.codepack")
    sub = parser.add_subparsers(dest="cmd", required=True)
    verify = sub.add_parser("verify", help="check an unpacked tree against a digest")
    verify.add_argument("--dir", default=".")
    verify.add_argument("--digest", required=True)
    build = sub.add_parser("digest", help="print the pack digest of a git work tree")
    build.add_argument("--repo", default=".")
    args = parser.parse_args(argv)
    try:
        if args.cmd == "verify":
            print(verify_tree(args.dir, args.digest))
        else:
            print(build_pack(args.repo)[0].digest)
    except CodePackError as e:
        print(f"codepack: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

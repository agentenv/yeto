"""The yeto tree a node runs, by hash.

The launcher builds one tar of ``yeto/`` (``git archive`` of a commit, or the
working tree for a test) and puts it once at ``node-code/<sha>.tar`` in the run
store (:func:`publish_code`, an exclusive create: a second publisher of the same
hash writes nothing). The node keeps one tree per hash at ``<work>/yeto/<sha>``
(:func:`ensure_code`): a tree already there is used as it is, otherwise the tar
is read from the store, checked against its sha256 and unpacked under a
temporary name, then moved into place. ``PYTHONPATH=<work>/yeto/<sha>`` then
imports that tree.
"""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import subprocess
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .records import CodeRef

CODE_DIR = "node-code"
REPO_ROOT = Path(__file__).resolve().parents[2]


class CodeError(RuntimeError):
    pass


def code_key(sha: str) -> str:
    return f"{CODE_DIR}/{sha}.tar"


def not_git_tree(repo: Path | str, why: str) -> str:
    """Why yeto at ``repo`` cannot ship its code (``why``: what git said), and what to do."""
    return (f"{repo} is not a git work tree ({why}): yeto ships its code to the nodes as a git archive of a "
            "commit and names the node's tree by that commit, so it must run from a git clone. Use a `git clone` "
            "of yeto (not a `git archive` export or a copied tree) as --yeto-dir / PYTHONPATH, or make the tree "
            "a repository first (`git init && git add yeto && git commit -m tree`)")


def git_tree_problem(repo: Path | str = REPO_ROOT, *, run: Callable[..., Any] = subprocess.run) -> str | None:
    """None when ``repo`` is inside a git work tree with a commit; else :func:`not_git_tree`."""
    try:
        # git needs none of the run's service settings (a sky API server endpoint is not passed on)
        env = {k: v for k, v in os.environ.items() if not k.startswith("SKYPILOT_")}
        done = run(["git", "-C", str(repo), "rev-parse", "--verify", "--quiet", "HEAD^{commit}"],
                   capture_output=True, text=True, timeout=60, env=env)
    except (OSError, subprocess.SubprocessError) as error:
        return not_git_tree(repo, f"git did not run: {error}")
    if done.returncode != 0:
        return not_git_tree(repo, f"git rev-parse HEAD: exit {done.returncode}")
    return None


def code_sha(ref: str = "HEAD", *, repo: Path | str = REPO_ROOT, run: Callable[..., Any] = subprocess.run) -> str:
    """The short commit of ``ref`` in ``repo`` (``git rev-parse --short=12``); a tree that
    is no git work tree raises :class:`CodeError` saying so (:func:`not_git_tree`)."""
    try:
        return run(["git", "-C", str(repo), "rev-parse", "--short=12", ref], capture_output=True, text=True,
                   check=True).stdout.strip()
    except subprocess.CalledProcessError as error:
        raise CodeError(not_git_tree(repo, f"git rev-parse {ref}: exit {error.returncode}")) from error
    except FileNotFoundError as error:
        raise CodeError(not_git_tree(repo, f"git did not run: {error}")) from error


def git_code_tar(ref: str = "HEAD", *, repo: Path = REPO_ROOT,
                 run: Callable[..., Any] = subprocess.run) -> tuple[str, bytes]:
    """``(short sha, tar of yeto/)`` of the commit ``ref`` (``git archive``)."""
    sha = code_sha(ref, repo=repo, run=run)
    data = run(["git", "-C", str(repo), "archive", "--format=tar", ref, "yeto"], capture_output=True,
               check=True).stdout
    return sha, data


def tree_code_tar(root: Path = REPO_ROOT) -> tuple[str, bytes]:
    """``(content hash, tar of root/yeto)`` of a working tree (tests, a dirty checkout):
    sorted members, fixed mtimes and owners, no ``__pycache__``, so one tree gives one hash."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        base = Path(root)
        for path in sorted((base / "yeto").rglob("*")):
            rel = path.relative_to(base)
            if "__pycache__" in rel.parts or path.suffix in (".pyc", ".pyo") or not (path.is_file() or path.is_dir()):
                continue
            info = tar.gettarinfo(str(path), arcname=rel.as_posix())
            info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
            if path.is_file():
                with path.open("rb") as handle:
                    tar.addfile(info, handle)
            else:
                tar.addfile(info)
    data = buf.getvalue()
    return "t" + hashlib.sha256(data).hexdigest()[:11], data


def publish_code(objects: Any, sha: str, tar: bytes) -> CodeRef:
    """Put the tar at ``node-code/<sha>.tar`` unless it is there; its reference."""
    key = code_key(sha)
    digest = hashlib.sha256(tar).hexdigest()
    created = objects.create(key, tar) if callable(getattr(objects, "create", None)) else None
    if created is None:
        if objects.read(key) is None:
            objects.write(key, tar)
    elif not created:
        have = objects.read(key)
        if have is not None and hashlib.sha256(have).hexdigest() != digest:
            raise CodeError(f"{key} in the store is another tree than {sha} (sha256 differs)")
    return CodeRef(sha=sha, key=key, sha256=digest, bytes=len(tar))


def code_dir(work: str, sha: str) -> str:
    return f"{work.rstrip('/')}/yeto/{sha}"


def ensure_code(objects: Any, code: CodeRef, work: str) -> str:
    """The tree of ``code`` at ``<work>/yeto/<sha>`` (read from the store when absent)."""
    dest = Path(code_dir(work, code.sha))
    if (dest / "yeto").is_dir():
        return str(dest)
    data = objects.read(code.key)
    if data is None:
        raise CodeError(f"the code tar {code.key} is not in the store")
    if hashlib.sha256(data).hexdigest() != code.sha256:
        raise CodeError(f"the code tar {code.key} does not match its sha256 {code.sha256[:12]}")
    tmp = dest.with_name(f"{dest.name}.tmp.{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tar:
        for member in tar.getmembers():
            target = (tmp / member.name).resolve()
            if not str(target).startswith(str(tmp.resolve()) + os.sep) or member.issym() or member.islnk():
                raise CodeError(f"the code tar {code.key} holds an unsafe member {member.name!r}")
        tar.extractall(tmp)  # noqa: S202 - members checked above
    if not (tmp / "yeto").is_dir():
        shutil.rmtree(tmp, ignore_errors=True)
        raise CodeError(f"the code tar {code.key} holds no yeto/ tree")
    try:
        tmp.rename(dest)
    except OSError:
        shutil.rmtree(tmp, ignore_errors=True)
        if not (dest / "yeto").is_dir():  # another runtime of this node won the rename
            raise
    return str(dest)

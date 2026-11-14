"""How much to re-hash release and cut files that already carry a recorded sha256.

A large policy release (2 bytes per parameter: ~240 GB at 120B, ~1.5 TB for an
MXFP4 2.8T MoE) used to be fully re-hashed by every hop that touched it: the
writer, the mirror that builds the spot manifest, the spot placement, every
sampler before it loads, and a cut restore on the driver (one file at a time).
The bytes are the same; only the first hash on each disk tells anything new.

``YETO_VERIFY`` picks the consumer-side policy (the producer always hashes once
and records the digest; nothing here changes a recorded format):

* ``full``  -- the default: every check hashes every file again (the earlier
  behavior, now with files hashed in parallel); nothing is cached or written;
* ``fast``  -- opt in. A file is hashed in full the first time it is seen
  on this disk; the result is cached under its identity
  ``(st_dev, st_ino, size, mtime_ns, ctime_ns)``. A later check of the same
  unchanged file reuses that digest. Any write, truncation, rename-over or
  replacement changes the identity (ctime moves even when mtime is kept), so
  the file is hashed again. Sizes are always compared. A cached digest is
  compared with the expected one like a fresh one, so a file moved under the
  name of another (misplacement) is still refused;
* ``off``   -- sizes only (still catches truncation and missing files; not
  corruption or swapped same-size files). For tests and trusted local disks.

``YETO_VERIFY_WORKERS`` (default 8): files hashed at a time (``hashlib``
releases the GIL). ``YETO_VERIFY_CACHE``: the cache directory (default
``~/.cache/yeto/verified``); an unwritable cache only costs re-hashing.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FULL, FAST, OFF = "full", "fast", "off"
MODES = (FULL, FAST, OFF)
ENV_MODE = "YETO_VERIFY"
ENV_WORKERS = "YETO_VERIFY_WORKERS"
ENV_CACHE = "YETO_VERIFY_CACHE"
DEFAULT_WORKERS = 8
CHUNK = 8 << 20


def verify_mode(environ: Mapping[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    mode = (env.get(ENV_MODE) or FULL).strip().lower()
    if mode not in MODES:
        raise ValueError(f"{ENV_MODE}={mode!r}: expected one of {', '.join(MODES)}")
    return mode


def hash_workers(environ: Mapping[str, str] | None = None) -> int:
    env = os.environ if environ is None else environ
    try:
        return max(1, int(env.get(ENV_WORKERS) or DEFAULT_WORKERS))
    except ValueError:
        return DEFAULT_WORKERS


def cache_dir(environ: Mapping[str, str] | None = None) -> Path:
    env = os.environ if environ is None else environ
    raw = env.get(ENV_CACHE)
    return Path(raw) if raw else Path.home() / ".cache" / "yeto" / "verified"


def _identity(st: os.stat_result) -> tuple[int, int, int, int, int]:
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def _cacheable(st: os.stat_result) -> bool:
    # a filesystem with whole-second timestamps cannot tell a rewrite within the
    # same second from the file that was hashed: never trust a cache entry there
    return bool(st.st_mtime_ns % 1_000_000_000 or st.st_ctime_ns % 1_000_000_000)


def _entry(root: Path, st: os.stat_result) -> Path:
    return root / f"{st.st_dev}-{st.st_ino}.json"


def cached_sha256(path: str | os.PathLike[str], *, cache: Path | None = None) -> str | None:
    """The cached digest of ``path`` when the file is unchanged since it was hashed."""
    try:
        st = os.stat(path)
        if not _cacheable(st):
            return None
        raw = json.loads(_entry(cache or cache_dir(), st).read_text())
        if tuple(raw["identity"]) != _identity(st):
            return None
        return str(raw["sha256"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _remember(path: str | os.PathLike[str], before: os.stat_result, sha: str, cache: Path) -> None:
    try:
        after = os.stat(path)
        if _identity(after) != _identity(before) or not _cacheable(after):
            return  # changed while it was hashed: the digest describes no stable file
        cache.mkdir(parents=True, exist_ok=True)
        target = _entry(cache, after)
        tmp = target.with_name(f".{target.name}.{os.getpid()}.{id(sha)}.tmp")
        tmp.write_text(json.dumps({"identity": list(_identity(after)), "sha256": sha,
                                   "path": os.fspath(path)}))
        os.replace(tmp, target)
    except OSError:
        pass


def sha256_file(path: str | os.PathLike[str], *, chunk: int = CHUNK, cache: Path | None = None,
                remember: bool | None = None) -> str:
    """Hash ``path`` in full (never from the cache); record the result for later
    ``fast`` checks when ``remember`` (default: ``YETO_VERIFY`` is ``fast``)."""
    if remember is None:
        remember = verify_mode() == FAST
    before = os.stat(path)
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    sha = digest.hexdigest()
    if remember:
        _remember(path, before, sha, cache or cache_dir())
    return sha


def digests(paths: Iterable[str | os.PathLike[str]], *, mode: str | None = None,
            workers: int | None = None, cache: Path | None = None) -> dict[str, str | None]:
    """{path: sha256} of ``paths`` under ``mode`` (module doc): None for every path
    when ``mode`` is ``off``. Missing files raise ``FileNotFoundError``."""
    mode = verify_mode() if mode is None else mode
    if mode not in MODES:
        raise ValueError(f"verify mode {mode!r}: expected one of {', '.join(MODES)}")
    keys = [os.fspath(p) for p in paths]
    if mode == OFF:
        return {k: None for k in keys}
    root = cache or cache_dir()
    out: dict[str, str | None] = {}
    todo = []
    for k in keys:
        hit = cached_sha256(k, cache=root) if mode == FAST else None
        if hit is None:
            todo.append(k)
        else:
            out[k] = hit
    n = max(1, min(workers or hash_workers(), len(todo) or 1))
    remember = mode == FAST
    if n == 1:
        out.update((k, sha256_file(k, cache=root, remember=remember)) for k in todo)
    else:
        with ThreadPoolExecutor(max_workers=n, thread_name_prefix="yeto-verify") as pool:
            out.update(zip(todo, pool.map(lambda k: sha256_file(k, cache=root, remember=remember), todo)))
    return out

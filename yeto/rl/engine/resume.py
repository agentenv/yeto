"""Resume a run from a checkpoint store across launches (openspec rl-resume-from-checkpoint).

Framework-neutral core (design §7): the store abstraction, hash manifests, the
``LATEST`` switch, retention, the run fingerprint and the resume checks. The
backend adapter only writes/reads the trainer shards (Miles: ``cut_plugin``).

Store layout (one run per store path)::

    <store>/round-cuts/<cut_id>/...        trainer shards + cut manifest.json (adapter)
    <store>/state/<seq:08d>/...            copy of the state dir (ledger, pointer, ...)
    <store>/state/<seq:08d>/STATE-MANIFEST.json   every file with sha256 + bytes, written last
    <store>/LATEST                         {"seq", "manifest_sha256", "cut_id", ...}, written last

A snapshot is complete only when ``LATEST`` names it; a reader checks every file
of the snapshot against its manifest (and the manifest against ``LATEST``) before
it trusts anything. Writing never overwrites the snapshot ``LATEST`` names, so a
sync torn by a crash/preemption leaves the previous snapshot valid.

Without ``--rl-elastic`` the round cuts are driven by :class:`ResumeController`
(the subset of the island controller the round cut uses: state dir, store,
incarnation, journal-ish record). ``--rl-elastic`` keeps its reconfiguration
controller, which mirrors its state dir into the same :class:`CheckpointStore`
interface in the flat layout (:func:`sync_flat_mirror` / :func:`restore_flat_mirror`)::

    <store>/...                            flat copy of the state dir
    <store>/STORE-MANIFEST.json            every file with sha256 + bytes, written last

What a save describes and the one save path are in :mod:`yeto.rl.engine.recovery`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LATEST = "LATEST"
STATE_DIR = "state"
STATE_MANIFEST = "STATE-MANIFEST.json"
ROUND_CUTS = "round-cuts"
FLAT_MANIFEST = "STORE-MANIFEST.json"  # flat layout (--rl-elastic): written last
INCARNATIONS = "incarnations.jsonl"  # under the state dir: one line per launch
RESUME_JOURNAL = "resume-journal.jsonl"  # under the state dir
SKIP_NAMES = frozenset({"journal.lock", STATE_MANIFEST, LATEST})
DEFAULT_CUT_KEEP = 2
DEFAULT_CUT_EVERY = 1


class StoreIntegrityError(RuntimeError):
    """A store copy does not match its manifest (fail closed, never resume from it)."""


class ResumeRefused(RuntimeError):
    """A resume check failed (config changed, version/lr/data do not continue)."""


# ---------------------------------------------------------------------------
# hashing
# ---------------------------------------------------------------------------
def sha256_file(path: str | os.PathLike[str], *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync(path: Path) -> None:
    with open(path, "rb") as fh:
        os.fsync(fh.fileno())


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> str:
    text = json.dumps(dict(payload), sort_keys=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    _fsync(tmp)
    os.replace(tmp, path)
    return hashlib.sha256(text.encode()).hexdigest()


def copy_verified(src: Path, dst: Path) -> tuple[str, int]:
    """Copy ``src`` to ``dst`` (tmp + fsync + rename) and check the copy's sha256 against
    the source's; returns (sha256, bytes). Raises :class:`StoreIntegrityError`."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    want = sha256_file(src)
    tmp = dst.with_name(f".{dst.name}.tmp")
    shutil.copyfile(src, tmp)
    _fsync(tmp)
    os.replace(tmp, dst)
    got = sha256_file(dst)
    if got != want:
        raise StoreIntegrityError(f"copy of {src} to {dst}: sha256 {got} != source {want}")
    return want, dst.stat().st_size


def _tree_files(root: Path, *, skip: Iterable[str] = ()) -> list[Path]:
    skip = set(skip) | set(SKIP_NAMES)
    out = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if path.is_file() and not (set(rel.parts) & skip) and not path.name.endswith(".tmp"):
            out.append(rel)
    return out


# ---------------------------------------------------------------------------
# the store interface + backends: local directory (this class), bucket mount,
# Modal Volume (``yeto.cloud.modal_ckpt_store``), critic (``yeto.rl.critic_state``)
# ---------------------------------------------------------------------------
class CheckpointStore:
    """The checkpoint store interface, and its local-directory backend.

    A store is a directory (``root``) the trainer, the state sync and a later launch
    all see. ``commit`` makes what was written durable/visible for the next reader
    (no-op locally) and returns the seconds it took; ``reload`` makes what another
    writer committed visible here. Backends override ``kind`` and, when the medium
    needs it, ``commit``/``reload``. Saving and restoring code takes a store and
    never branches on the backend.

    ``objects``: the same store addressed by object key
    (:class:`~yeto.rl.engine.object_store.ObjectStore`), or None. A backend whose
    mount cannot carry state that one host rewrites while another reads it (a
    bucket mount) sets it; the sample buffer, the trainer commands and the policy
    release pointer then use it instead of the files under ``root``."""

    kind = "local"
    objects: Any = None

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).expanduser()

    def commit(self) -> float:
        """Returns the seconds the commit took."""
        return 0.0

    def reload(self) -> None:
        return None

    def describe(self) -> dict[str, Any]:
        return {"kind": self.kind, "root": str(self.root)}

    def list(self, sub: str = "") -> list[str]:
        base = self.root / sub
        return sorted(p.name for p in base.iterdir()) if base.is_dir() else []


class BucketMountStore(CheckpointStore):
    """sky Storage MOUNT of a bucket (s3://, gs://, r2://): files are uploaded by the
    mount; fsync on every write is the only commit we can issue."""

    kind = "bucket-mount"


STORE_IMPL_ENV = "YETO_RL_STORE_IMPL"  # "package.module:callable", exported by the launch layer
STORE_ENV_PREFIX = "YETO_RL_STORE_"  # the implementation's own settings (read from the environ)
_STORE_IMPLS: dict[str, Callable[..., CheckpointStore]] = {}


def register_store_impl(name: str, factory: Callable[..., CheckpointStore]) -> None:
    """Register a store implementation under ``name`` (what :data:`STORE_IMPL_ENV` may
    name instead of a dotted path). Cloud-specific stores (e.g. the Modal Volume one in
    ``yeto.cloud.modal_ckpt_store``) live outside the neutral core and are selected by
    the launch layer, which exports :data:`STORE_IMPL_ENV` on the island."""
    _STORE_IMPLS[name] = factory


def _store_impl(spec: str) -> Callable[..., CheckpointStore]:
    if spec in _STORE_IMPLS:
        return _STORE_IMPLS[spec]
    module, sep, attr = spec.partition(":")
    if not sep or not module or not attr:
        raise ValueError(f"{STORE_IMPL_ENV}={spec!r}: expected 'package.module:callable'")
    import importlib

    return getattr(importlib.import_module(module), attr)


def store_for(path: str | os.PathLike[str], *, environ: Mapping[str, str] | None = None,
              **impl_kwargs: Any) -> CheckpointStore:
    """The store implementation for a store path on the island. When the launch layer
    exported :data:`STORE_IMPL_ENV`, that implementation builds the store (it reads its
    own settings from ``environ``; ``impl_kwargs`` are passed through, e.g. a test
    double); otherwise a bucket mount or a plain shared directory."""
    env = os.environ if environ is None else environ
    spec = env.get(STORE_IMPL_ENV)
    if spec:
        return _store_impl(spec)(path, environ=env, **impl_kwargs)
    if str(path).startswith(("~/yeto-checkpoint-store", str(Path("~/yeto-checkpoint-store").expanduser()))):
        return BucketMountStore(path)
    return CheckpointStore(path)


# ---------------------------------------------------------------------------
# snapshots + LATEST
# ---------------------------------------------------------------------------
def read_latest(store_root: Path) -> dict[str, Any] | None:
    path = Path(store_root) / LATEST
    if not path.is_file():
        return None
    return dict(json.loads(path.read_text(encoding="utf-8")))


def write_snapshot(store: CheckpointStore, state_dir: Path, info: Mapping[str, Any], *,
                   skip: Iterable[str] = (), commit: bool = True) -> dict[str, Any]:
    """Copy ``state_dir`` into a NEW snapshot dir (every file hash-checked after the
    copy), write its manifest, then ``LATEST``, then commit. Returns the LATEST record
    plus timings/bytes.

    ``commit=False`` leaves the commit to the caller (``commit_s`` is None): until it
    commits, a reader of a committing store still sees the previous LATEST."""
    root = store.root
    started = time.monotonic()
    latest = read_latest(root)
    seq = int(latest["seq"]) + 1 if latest else 0
    snap = root / STATE_DIR / f"{seq:08d}"
    if snap.exists():  # a torn earlier attempt at this seq (never named by LATEST)
        shutil.rmtree(snap)
    files: dict[str, dict[str, Any]] = {}
    total = 0
    for rel in _tree_files(Path(state_dir), skip=skip):
        digest, size = copy_verified(Path(state_dir) / rel, snap / rel)
        files[str(rel)] = {"sha256": digest, "bytes": size}
        total += size
    manifest = {"seq": seq, "files": files, **{k: v for k, v in info.items() if k != "files"}}
    manifest_sha = _write_json_atomic(snap / STATE_MANIFEST, manifest)
    copied_s = time.monotonic() - started
    record = {"seq": seq, "manifest_sha256": manifest_sha, "bytes": total,
              **{k: info[k] for k in ("cut_id", "next_rollout_id", "incarnation", "reason") if k in info}}
    _write_json_atomic(root / LATEST, record)
    commit_s = store.commit() if commit else None
    return {**record, "copy_s": copied_s, "commit_s": commit_s}


def verify_snapshot(store_root: Path, latest: Mapping[str, Any]) -> dict[str, Any]:
    """The manifest of the snapshot ``latest`` names, after checking the manifest hash
    and every file's sha256/bytes. Raises :class:`StoreIntegrityError`."""
    snap = Path(store_root) / STATE_DIR / f"{int(latest['seq']):08d}"
    mpath = snap / STATE_MANIFEST
    if not mpath.is_file():
        raise StoreIntegrityError(f"LATEST names snapshot {latest['seq']} but {mpath} is missing")
    text = mpath.read_text(encoding="utf-8")
    if hashlib.sha256(text.encode()).hexdigest() != latest.get("manifest_sha256"):
        raise StoreIntegrityError(f"snapshot {latest['seq']} manifest hash differs from LATEST")
    manifest = json.loads(text)
    problems = []
    for rel, meta in sorted(manifest["files"].items()):
        path = snap / rel
        if not path.is_file():
            problems.append(f"{rel}: missing")
        elif path.stat().st_size != int(meta["bytes"]):
            problems.append(f"{rel}: {path.stat().st_size} bytes != {meta['bytes']}")
        elif sha256_file(path) != meta["sha256"]:
            problems.append(f"{rel}: sha256 mismatch")
    if problems:
        raise StoreIntegrityError(f"snapshot {latest['seq']}: " + "; ".join(problems))
    return manifest


def restore_snapshot(store_root: Path, state_dir: Path, latest: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the named snapshot and copy it into ``state_dir`` (each copy hash-checked).
    An existing state dir is moved aside (``<state_dir>.prev-<ts>``), never merged: the
    store's LATEST is the authority for a resume."""
    manifest = verify_snapshot(store_root, latest)
    state_dir = Path(state_dir)
    moved = None
    if state_dir.exists() and any(state_dir.iterdir()):
        moved = state_dir.with_name(f"{state_dir.name}.prev-{int(time.time())}")
        state_dir.rename(moved)
    state_dir.mkdir(parents=True, exist_ok=True)
    snap = Path(store_root) / STATE_DIR / f"{int(latest['seq']):08d}"
    total = 0
    for rel, meta in manifest["files"].items():
        digest, size = copy_verified(snap / rel, state_dir / rel)
        if digest != meta["sha256"]:
            raise StoreIntegrityError(f"restored {rel}: sha256 {digest} != manifest {meta['sha256']}")
        total += size
    return {"seq": int(latest["seq"]), "bytes": total, "files": len(manifest["files"]),
            "moved_aside": None if moved is None else str(moved)}


def sync_flat_mirror(store: CheckpointStore, state_dir: Path, info: Mapping[str, Any]) -> dict[str, Any]:
    """Flat layout (``--rl-elastic``): copy every file of ``state_dir`` into the store
    root (each copy hash-checked) and write :data:`FLAT_MANIFEST` last. Does NOT
    commit: the caller commits outside any lock it holds (a Modal commit may take
    minutes). Returns the manifest (``info`` plus ``files``)."""
    root = store.root
    root.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(Path(state_dir).rglob("*")):
        rel = path.relative_to(state_dir)
        if (not path.is_file() or rel.parts[0] == FLAT_MANIFEST
                or path.name == "journal.lock" or path.name.endswith(".tmp")):
            continue
        digest, size = copy_verified(path, root / rel)
        files[str(rel)] = {"sha256": digest, "bytes": size}
    manifest = {**info, "files": files}
    tmp = root / (FLAT_MANIFEST + ".tmp")
    tmp.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    tmp.replace(root / FLAT_MANIFEST)
    return manifest


def restore_flat_mirror(store: CheckpointStore, state_dir: Path) -> dict[str, Any] | None:
    """Flat layout: copy a complete store copy (:data:`FLAT_MANIFEST` present) into
    ``state_dir``, every file checked against the manifest first (fail closed with
    :class:`StoreIntegrityError`). A manifest without ``files`` (written before the
    hashes) restores by plain copy. Round cuts stay in the store (read in place).
    Returns where the copy came from, or None when the store holds no copy."""
    store.reload()
    root = store.root
    manifest = root / FLAT_MANIFEST
    if not manifest.is_file():
        return None
    info = json.loads(manifest.read_text(encoding="utf-8"))
    origin = {"restored_from": {k: info.get(k) for k in ("incarnation", "reason", "config_epoch",
                                                         "wall_time", "seq")}}
    state_dir = Path(state_dir)
    files = info.get("files")
    if files is None:
        state_dir.mkdir(parents=True, exist_ok=True)
        _plain_copy(root, state_dir)
        return origin
    bad = [rel for rel, meta in sorted(files.items())
           if not (root / rel).is_file() or sha256_file(root / rel) != meta["sha256"]]
    if bad:
        raise StoreIntegrityError(f"checkpoint store {root}: {len(bad)} file(s) differ from "
                                  f"{FLAT_MANIFEST}: {bad[:8]}")
    state_dir.mkdir(parents=True, exist_ok=True)
    for rel in sorted(files):
        copy_verified(root / rel, state_dir / rel)
    return {**origin, "verified_files": len(files)}


def _plain_copy(root: Path, state_dir: Path) -> None:
    for child in root.iterdir():
        if child.name in (FLAT_MANIFEST, ROUND_CUTS):
            continue
        target = state_dir / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        else:
            shutil.copy2(child, target)


def prune(store_root: Path, *, keep: int, cut_ids_of: Callable[[Path], str | None]) -> list[str]:
    """Keep the ``keep`` newest snapshots (by seq) and only the cuts they point at.
    Runs only after LATEST names the new snapshot; returns what was deleted."""
    if keep < 1:
        raise ValueError("keep must be >= 1")
    root = Path(store_root)
    snaps = sorted((p for p in (root / STATE_DIR).glob("*") if p.is_dir()), key=lambda p: p.name)
    latest = read_latest(root)
    if latest is None:
        return []
    current = f"{int(latest['seq']):08d}"
    kept = [p for p in snaps if p.name <= current][-keep:]
    deleted = []
    for p in snaps:
        if p not in kept and p.name < current:
            shutil.rmtree(p, ignore_errors=True)
            deleted.append(f"{STATE_DIR}/{p.name}")
    wanted = {cid for cid in (cut_ids_of(p) for p in kept) if cid}
    # E1: a cut newer than every kept one may be being written right now (a periodic
    # cut is captured while the previous cut commits in the background): not pruned
    newest = max((r for r in map(_cut_rollout, wanted) if r is not None), default=None)
    cuts = root / ROUND_CUTS
    if cuts.is_dir():
        for p in sorted(cuts.iterdir()):
            rollout = _cut_rollout(p.name)
            if newest is not None and rollout is not None and rollout > newest:
                continue
            if p.is_dir() and p.name not in wanted:
                shutil.rmtree(p, ignore_errors=True)
                deleted.append(f"{ROUND_CUTS}/{p.name}")
    return deleted


def _cut_rollout(cut_id: str) -> int | None:
    """Rollout id of a round cut named ``r<rollout:06d>-<incarnation>`` (None: other name)."""
    head, sep, _ = cut_id.partition("-")
    if not sep or not head.startswith("r") or not head[1:].isdigit():
        return None
    return int(head[1:])


def directory_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


# ---------------------------------------------------------------------------
# run fingerprint (design §3): what must not change between the cut and the resume
# ---------------------------------------------------------------------------
def fingerprint_digest(fields: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(dict(fields), sort_keys=True, default=str).encode()).hexdigest()


def fingerprint_diff(saved: Mapping[str, Any] | None, now: Mapping[str, Any]) -> dict[str, Any]:
    saved = dict(saved or {})
    keys = sorted(set(saved) | set(now))
    return {k: {"cut": saved.get(k), "now": now.get(k)} for k in keys if saved.get(k) != now.get(k)}


def check_fingerprint(saved: Mapping[str, Any] | None, now: Mapping[str, Any], *,
                      allow_change: bool) -> dict[str, Any]:
    """Returns the differences (empty = identical). Raises :class:`ResumeRefused` on a
    difference unless ``allow_change`` (``--rl-resume-allow-config-change``)."""
    if saved is None:
        diff = {"run_fingerprint": {"cut": None, "now": "present"}}
    else:
        diff = fingerprint_diff(saved, now)
    if diff and not allow_change:
        raise ResumeRefused("the run configuration differs from the cut's "
                            f"({', '.join(sorted(diff))}); pass --rl-resume-allow-config-change "
                            "to resume anyway: " + json.dumps(diff, sort_keys=True, default=str)[:2000])
    return diff


def consumed_digest(group_ids: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(str(g) for g in group_ids)).encode()).hexdigest()


def check_lr_continues(expected: list[float] | None, applied: list[float] | None) -> str | None:
    """Design §4.5: the first resumed round trains with exactly the pointer's
    ``lr_at_next_round`` (bitwise; None = schedule cannot predict, not checked)."""
    if expected is None:
        return None
    if applied is None:
        return "the first resumed round reported no applied learning rate"
    if [float(x).hex() for x in applied] != [float(x).hex() for x in expected]:
        return f"first resumed round trained at lr {applied}, the cut expected {expected}"
    return None


def should_cut(rollout_id: int, *, every: int, final: bool = False, stop: bool = False) -> bool:
    """Cut at the safe point before ``rollout_id`` every ``every`` rounds, and always at
    the end (last round, our own stop)."""
    if every < 1:
        raise ValueError("--rl-cut-every must be >= 1")
    return rollout_id > 0 and (final or stop or rollout_id % every == 0)


# ---------------------------------------------------------------------------
# ResumeController: the round-cut controller of a run without --rl-elastic
# ---------------------------------------------------------------------------
@dataclass
class _Epochs:
    config_epoch: int = 0


class _JournalShim:
    def __init__(self) -> None:
        self.epochs = _Epochs()
        self.records: list[dict[str, Any]] = []


class ResumeController:
    """The recovery controller of a run (the ``--rl-elastic`` reconfiguration controller
    only mirrors its state dir through :func:`sync_flat_mirror` / :func:`restore_flat_mirror`).

    The surface :class:`~yeto.rl.adapters.miles.round_cut.RoundCutCheckpoint` needs,
    without the elastic controller: ``state_dir``, ``checkpoint_store`` (path),
    ``incarnation`` (``id`` + ``index``, 0 = first launch), ``recovery_required``,
    ``sync_checkpoint_store`` (snapshot + LATEST + commit + prune) and ``_record``.

    At construction, a store with ``LATEST`` is restored into the state dir (verified);
    a store that holds cuts but no ``LATEST`` is refused (never a silent restart at 0)."""

    def __init__(self, *, state_dir: str | os.PathLike[str], store: CheckpointStore,
                 runtime_fingerprint: str = "", keep: int = DEFAULT_CUT_KEEP,
                 wall_clock: Callable[[], float] = time.time) -> None:
        if keep < 1:
            raise ValueError("--rl-cut-keep must be >= 1")
        self.state_dir = Path(state_dir).expanduser()
        self.store = store
        self.checkpoint_store = store.root
        self.runtime_fingerprint = runtime_fingerprint
        self.keep = int(keep)
        self.recovery_required: str | None = None
        self.journal = _JournalShim()
        self.last_store_sync: dict[str, Any] | None = None
        self._wall = wall_clock
        store.root.mkdir(parents=True, exist_ok=True)
        store.reload()
        self.restored: dict[str, Any] | None = None
        latest = read_latest(store.root)
        started = time.monotonic()
        if latest is not None:
            self.restored = restore_snapshot(store.root, self.state_dir, latest)
            self.restored["latest"] = dict(latest)
            self.restored["seconds"] = time.monotonic() - started
        elif (store.root / ROUND_CUTS).is_dir() and any((store.root / ROUND_CUTS).iterdir()):
            raise StoreIntegrityError(f"{store.root} holds round cuts but no {LATEST}: refusing to "
                                      "start over at rollout 0 on top of an unfinished store")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        previous = self._incarnations()
        self.incarnation = {"id": uuid.uuid4().hex[:12], "index": len(previous), "pid": os.getpid(),
                            "wall_time": self._wall()}
        with open(self.state_dir / INCARNATIONS, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(self.incarnation, sort_keys=True) + "\n")
        self.previous_incarnations = previous
        if self.restored is not None:
            self._record("checkpoint_store", tx_id=None, action="restore", **{
                k: v for k, v in self.restored.items() if k != "latest"},
                latest_seq=int(latest["seq"]), store=str(store.root))

    def _incarnations(self) -> list[dict[str, Any]]:
        path = self.state_dir / INCARNATIONS
        if not path.is_file():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _record(self, kind: str, **fields: Any) -> dict[str, Any]:
        record = {"kind": kind, "seq": len(self.journal.records), "wall_time": self._wall(), **fields}
        self.journal.records.append(record)
        with open(self.state_dir / RESUME_JOURNAL, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        return record

    def sync_checkpoint_store(self, reason: str, *, skip_if_recovery: bool = False,
                              extra: Mapping[str, Any] | None = None,
                              defer_commit: bool = False) -> bool:
        """Snapshot the state dir into the store, then ``LATEST``, commit, prune. A
        failure is remembered (``last_store_sync``) and returned False: the previous
        LATEST stays valid.

        ``defer_commit``: stop after ``LATEST``; :meth:`commit_checkpoint_store` commits
        and prunes later (the round cut runs it off the round's critical path). Until
        then the store's committed LATEST is the previous one, so a crash in between
        resumes from the previous cut."""
        if skip_if_recovery and self.recovery_required:
            return False
        info = {"incarnation": self.incarnation["id"], "incarnation_index": self.incarnation["index"],
                "reason": reason, "wall_time": self._wall(), **dict(extra or {})}
        pointer = self.state_dir / "round-cut.json"
        if pointer.is_file():  # LATEST names the cut the snapshot's pointer names
            try:
                p = json.loads(pointer.read_text(encoding="utf-8"))
                info.setdefault("cut_id", p.get("cut_id"))
                info.setdefault("next_rollout_id", p.get("next_rollout_id"))
            except ValueError:
                pass
        try:
            result = write_snapshot(self.store, self.state_dir, info, commit=not defer_commit)
            if defer_commit:
                self.last_store_sync = {**info, **result, "ok": True, "committed": False, "pruned": None}
                return True
            deleted = self._prune()
        except Exception as exc:  # noqa: BLE001
            self.last_store_sync = {**info, "ok": False, "error": repr(exc)}
            logging.getLogger(__name__).warning("checkpoint store sync (%s) failed: %r", reason, exc)
            return False
        self.last_store_sync = {**info, **result, "ok": True, "committed": True, "pruned": deleted}
        return True

    def commit_checkpoint_store(self) -> dict[str, Any]:
        """Commit a ``defer_commit`` sync, then prune (pruning only after the new LATEST
        is committed, so a failed commit never loses the previous cut). Thread-safe
        against the caller as long as nothing else writes to the store meanwhile.
        Returns (and remembers in ``last_store_sync``) the outcome; never raises."""
        sync = dict(self.last_store_sync or {})
        try:
            sync["commit_s"] = self.store.commit()
            sync["committed"] = True
            sync["pruned"] = self._prune()
        except Exception as exc:  # noqa: BLE001
            sync.update(ok=False, error=repr(exc))
            logging.getLogger(__name__).warning("checkpoint store commit (%s) failed: %r",
                                                sync.get("reason"), exc)
        self.last_store_sync = sync
        return sync

    def _prune(self) -> list[str]:
        deleted = prune(self.store.root, keep=self.keep, cut_ids_of=_snapshot_cut_id)
        if deleted:
            self.store.commit()
        return deleted

    def close(self) -> None:
        return None


def _snapshot_cut_id(snapshot_dir: Path) -> str | None:
    pointer = snapshot_dir / "round-cut.json"
    if not pointer.is_file():
        return None
    return json.loads(pointer.read_text(encoding="utf-8")).get("cut_id")


# ---------------------------------------------------------------------------
# preemption notice (design 2.5): write a marker, do not try to save in 30 s
# ---------------------------------------------------------------------------
PREEMPT_MARKER = "PREEMPT-NOTICE.json"


def write_preempt_marker(store_root: Path, *, signal_name: str, incarnation: Mapping[str, Any],
                         rollout_id: int | None, store: CheckpointStore | None = None) -> dict[str, Any]:
    record = {"signal": signal_name, "incarnation": dict(incarnation), "rollout_id": rollout_id,
              "wall_time": time.time(), "saved": False}
    path = Path(store_root) / "preempt-notices"
    path.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(path / f"{incarnation.get('index', 0):04d}-{incarnation.get('id', 'x')}.json", record)
    if store is not None:
        try:
            store.commit()
        except Exception:  # noqa: BLE001
            pass
    return record


def install_preempt_handler(controller: "ResumeController", *, emit: Callable[..., Any] | None = None,
                            signals: Iterable[str] = ("SIGTERM", "SIGINT")) -> list[str]:
    """Design 2.5: on the platform's stop signal (Modal: SIGINT/SIGTERM, then a short
    grace before the kill) write only a preemption marker into the store and hand the
    signal on to the previous handler. No cut is attempted in the grace window: the
    next launch resumes from the newest complete cut (LATEST)."""
    import signal as _signal
    import threading

    if threading.current_thread() is not threading.main_thread():
        return []
    del emit  # no tape write from a signal handler (the tape writer may hold its lock)
    installed = []
    for name in signals:
        signum = getattr(_signal, name, None)
        if signum is None:
            continue
        previous = _signal.getsignal(signum)

        def handler(num, frame, _prev=previous, _name=name):
            try:
                record = write_preempt_marker(controller.checkpoint_store, signal_name=_name,
                                              incarnation=controller.incarnation, rollout_id=None,
                                              store=controller.store)
                print(f"[yeto] rl_preempt_notice {json.dumps(record, sort_keys=True)}", flush=True)
            except Exception:  # noqa: BLE001 - never block the shutdown
                pass
            if callable(_prev):
                return _prev(num, frame)
            if _prev == _signal.SIG_IGN:
                return None
            _signal.signal(num, _signal.SIG_DFL)
            os.kill(os.getpid(), num)
            return None

        _signal.signal(signum, handler)
        installed.append(name)
    return installed

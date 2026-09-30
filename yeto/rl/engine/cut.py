"""``ReconfigurationCut`` manifest (rl-infra-spec 4.1/4.2, design D5). Import-light.

A cut is a directory written once and then only read::

    <root>/<cut_id>/files...       rank shards written by the engine (each fsynced)
    <root>/<cut_id>/manifest.json  written LAST (tmp + fsync + rename + dir fsync)

A cut without ``manifest.json`` does not exist. The manifest lists every
file with its size and sha256, the progress counters, the algorithm
identity (alignment A3), the data cursor and the ledger summary. It is
separate from Miles' default checkpoint path (``--save``/``--load`` stay
unused by the ports engine; its ``--no-save-optim/--no-load-optim/
--no-save-rng/--no-load-rng`` flags are untouched): the cut is saved and
restored only through explicit port verbs.

:func:`verify_cut` rejects a missing/truncated/corrupted file, a manifest
whose own hash does not match, missing required state, progress counters
that disagree with each other or with the caller's expectation, and an
algorithm identity or parallel layout that differs from the current run.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CUT_SCHEMA = "yeto.reconfiguration_cut/v1"
MANIFEST_NAME = "manifest.json"
_CUT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class CutError(RuntimeError):
    """The cut is incomplete, corrupted or does not belong to this run."""


def sha256_file(path: str | os.PathLike[str], *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def fsync_file(path: str | os.PathLike[str]) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fsync_dir(path: str | os.PathLike[str]) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class CutFile:
    """One shard of the cut, relative to the cut directory."""

    path: str
    sha256: str
    bytes: int
    coord: Mapping[str, int] = field(default_factory=dict)  # tp/pp/dp/global rank

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "bytes": int(self.bytes),
                "coord": {k: int(v) for k, v in sorted(self.coord.items())}}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CutFile":
        path = str(raw["path"])
        if Path(path).is_absolute() or ".." in Path(path).parts:
            raise CutError(f"cut file path escapes the cut directory: {path!r}")
        return cls(path, str(raw["sha256"]), int(raw["bytes"]), dict(raw.get("coord") or {}))


@dataclass(frozen=True)
class AlgorithmIdentity:
    """alignment A3: what must be identical between the cut and the restoring run."""

    algorithm_spec_sha256: str
    plugin_sha256: tuple[str, ...] = ()  # sorted "<dotted path>@<source sha256>" of every PluginRef
    runtime_attrs_sha256: str | None = None  # hash of the yeto_algo_plugins runtime attrs
    ref_model: Mapping[str, Any] | None = None  # {"ref_load", "base_model_revision"} when a ref model is used

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm_spec_sha256": self.algorithm_spec_sha256,
            "plugin_sha256": sorted(self.plugin_sha256),
            "runtime_attrs_sha256": self.runtime_attrs_sha256,
            "ref_model": None if self.ref_model is None else dict(sorted(self.ref_model.items())),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AlgorithmIdentity":
        return cls(
            str(raw["algorithm_spec_sha256"]),
            tuple(sorted(raw.get("plugin_sha256") or ())),
            raw.get("runtime_attrs_sha256"),
            raw.get("ref_model"),
        )

    @classmethod
    def from_spec(
        cls,
        spec: Any,
        *,
        runtime_attrs: Mapping[str, Any] | None = None,
        ref_model: Mapping[str, Any] | None = None,
    ) -> "AlgorithmIdentity":
        """Identity of an ``AlgorithmSpec`` (duck-typed: ``sha256()``, PluginRef ``sha256``)."""
        plugins = sorted({f"{p.path}@{p.sha256}" for p in iter_plugin_refs(spec)})
        attrs = None
        if runtime_attrs:
            attrs = hashlib.sha256(canonical_json(_jsonable(runtime_attrs)).encode()).hexdigest()
        return cls(spec.sha256(), tuple(plugins), attrs, ref_model)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    to_dict = getattr(value, "to_dict", None)
    return _jsonable(to_dict()) if callable(to_dict) else repr(value)


def iter_plugin_refs(spec: Any, _seen: set[int] | None = None) -> Iterable[Any]:
    """Every PluginRef-like object (has ``path`` and ``sha256``) reachable from the spec."""
    seen = _seen if _seen is not None else set()
    if spec is None or id(spec) in seen or isinstance(spec, (str, bytes, int, float, bool)):
        return
    seen.add(id(spec))
    if hasattr(spec, "path") and hasattr(spec, "sha256") and not callable(getattr(spec, "sha256")):
        yield spec
        return
    if isinstance(spec, Mapping):
        children = list(spec.values())
    elif isinstance(spec, (list, tuple, set, frozenset)):
        children = list(spec)
    elif hasattr(spec, "__dataclass_fields__"):
        children = [getattr(spec, name, None) for name in spec.__dataclass_fields__]
    else:
        return
    for child in children:
        yield from iter_plugin_refs(child, seen)


@dataclass(frozen=True)
class CutProgress:
    """Counters that must agree (4.2: step mismatch is refused)."""

    local_step: int  # optimizer steps applied by this island
    scheduler_samples: int  # Megatron opt_param_scheduler.num_steps (in samples)
    global_batch_size: int
    next_rollout_id: int
    policy_version: int
    policy_hash: str

    def problems(self) -> list[str]:
        out = []
        if self.global_batch_size <= 0:
            out.append("global_batch_size must be positive")
        elif self.scheduler_samples != self.local_step * self.global_batch_size:
            out.append(
                f"scheduler progress {self.scheduler_samples} samples != local_step {self.local_step} "
                f"x global_batch_size {self.global_batch_size}"
            )
        if min(self.local_step, self.next_rollout_id, self.policy_version) < 0:
            out.append("negative progress counter")
        if not self.policy_hash:
            out.append("policy_hash is empty")
        return out

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


# State families that a complete cut must carry (design D5 table / 4.1 audit).
REQUIRED_SECTIONS = ("runtime", "progress", "algorithm", "data", "ledger", "outer", "files")


@dataclass(frozen=True)
class CutManifest:
    cut_id: str
    epoch: int
    runtime: Mapping[str, Any]  # backend fingerprint, layout, precision, rng policy
    progress: CutProgress
    algorithm: AlgorithmIdentity
    data: Mapping[str, Any]  # rollout data cursor (Miles RolloutDataSource state)
    ledger: Mapping[str, Any]  # 3.6 summary: carried_over, ready_unconsumed, counts
    outer: Mapping[str, Any]  # outer protocol position; settled must be True (D5)
    files: tuple[CutFile, ...]
    rank_summaries: tuple[Mapping[str, Any], ...] = ()
    schema: str = CUT_SCHEMA

    def body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "cut_id": self.cut_id,
            "epoch": int(self.epoch),
            "runtime": _jsonable(self.runtime),
            "progress": self.progress.to_dict(),
            "algorithm": self.algorithm.to_dict(),
            "data": _jsonable(self.data),
            "ledger": _jsonable(self.ledger),
            "outer": _jsonable(self.outer),
            "files": [f.to_dict() for f in sorted(self.files, key=lambda f: f.path)],
            "rank_summaries": [_jsonable(s) for s in self.rank_summaries],
        }

    def digest(self) -> str:
        return hashlib.sha256(canonical_json(self.body()).encode()).hexdigest()

    def to_json(self) -> str:
        return json.dumps({**self.body(), "manifest_sha256": self.digest()}, indent=1, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "CutManifest":
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise CutError(f"cut manifest is not JSON: {exc}") from exc
        if raw.get("schema") != CUT_SCHEMA:
            raise CutError(f"unknown cut schema {raw.get('schema')!r} (reader knows {CUT_SCHEMA})")
        missing = [k for k in (*REQUIRED_SECTIONS, "cut_id", "epoch") if raw.get(k) is None]
        if missing:
            raise CutError(f"cut manifest lacks {missing}")
        try:
            manifest = cls(
                cut_id=str(raw["cut_id"]),
                epoch=int(raw["epoch"]),
                runtime=raw["runtime"],
                progress=CutProgress(**raw["progress"]),
                algorithm=AlgorithmIdentity.from_dict(raw["algorithm"]),
                data=raw["data"],
                ledger=raw["ledger"],
                outer=raw["outer"],
                files=tuple(CutFile.from_dict(f) for f in raw["files"]),
                rank_summaries=tuple(raw.get("rank_summaries") or ()),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CutError(f"malformed cut manifest: {exc}") from exc
        if raw.get("manifest_sha256") != manifest.digest():
            raise CutError("cut manifest hash mismatch (edited or truncated)")
        return manifest

    def completeness_problems(self) -> list[str]:
        """Required state that is missing or inconsistent (4.1: missing state is refused)."""
        out = [f"progress: {p}" for p in self.progress.problems()]
        if not _CUT_ID.match(self.cut_id):
            out.append(f"invalid cut_id {self.cut_id!r}")
        if not self.algorithm.algorithm_spec_sha256:
            out.append("algorithm: algorithm_spec_sha256 missing")
        if not self.files:
            out.append("files: no trainer shard")
        paths = [f.path for f in self.files]
        if len(paths) != len(set(paths)):
            out.append("files: duplicate path")
        if len(self.rank_summaries) != len(self.files):
            out.append(f"rank_summaries: {len(self.rank_summaries)} for {len(self.files)} shards")
        for key in ("sample_offset", "epoch_id", "sample_group_index", "sample_index"):
            if not isinstance(self.data.get(key), int):
                out.append(f"data: cursor field {key!r} missing")
        if self.ledger.get("carried_over") != 0:
            # 4.1 audit: on the ports path Miles returns no reusable leftover
            # (partial rollout refused, surplus groups dropped, buffer not saved).
            out.append(f"ledger: carried_over must be 0 on the Miles ports path, got {self.ledger.get('carried_over')!r}")
        if self.ledger.get("ready_unconsumed") != 0:
            out.append("ledger: first-version cut requires no ready-unconsumed group (quiescent cut)")
        if self.outer.get("settled") is not True:
            out.append("outer: the outer commit of this cut is not settled (D5)")
        for key in ("backend_fingerprint", "layout", "rng_policy"):
            if not self.runtime.get(key):
                out.append(f"runtime: {key} missing")
        for i, summary in enumerate(self.rank_summaries):
            if summary.get("scheduler_samples") != self.progress.scheduler_samples:
                out.append(
                    f"rank {i}: scheduler at {summary.get('scheduler_samples')} samples, "
                    f"cut says {self.progress.scheduler_samples}"
                )
            if not summary.get("has_optimizer_state") or not summary.get("has_rng"):
                out.append(f"rank {i}: optimizer state or RNG missing")
        return out


def cut_dir(root: str | os.PathLike[str], cut_id: str) -> Path:
    if not _CUT_ID.match(cut_id):
        raise CutError(f"invalid cut_id {cut_id!r}")
    return Path(root) / cut_id


def commit_manifest(root: str | os.PathLike[str], manifest: CutManifest, *, check_files: bool = True) -> Path:
    """Validate completeness (and, when visible here, every shard), then publish the manifest atomically.

    The ranks have already fsynced their shards; ``check_files=False`` is for
    shards that live on other nodes (each rank re-verifies its own on restore).
    """
    problems = manifest.completeness_problems()
    directory = cut_dir(root, manifest.cut_id)
    if check_files:
        problems += file_problems(directory, manifest.files)
    if problems:
        raise CutError("refusing an incomplete cut: " + "; ".join(problems))
    final = directory / MANIFEST_NAME
    if final.exists():
        raise CutError(f"cut {manifest.cut_id!r} already has a manifest (cuts are immutable)")
    tmp = directory / (MANIFEST_NAME + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(manifest.to_json())
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, final)
    fsync_dir(directory)
    return final


def file_problems(directory: Path, files: Iterable[CutFile]) -> list[str]:
    out = []
    for f in files:
        path = directory / f.path
        if not path.is_file():
            out.append(f"missing shard {f.path}")
            continue
        size = path.stat().st_size
        if size != f.bytes:
            out.append(f"shard {f.path} is {size} bytes, manifest says {f.bytes} (truncated?)")
            continue
        if sha256_file(path) != f.sha256:
            out.append(f"shard {f.path} checksum mismatch")
    return out


def load_manifest(root: str | os.PathLike[str], cut_id: str) -> CutManifest:
    path = cut_dir(root, cut_id) / MANIFEST_NAME
    if not path.is_file():
        raise CutError(f"cut {cut_id!r} has no committed manifest")
    manifest = CutManifest.from_json(path.read_text(encoding="utf-8"))
    if manifest.cut_id != cut_id:
        raise CutError(f"manifest names cut {manifest.cut_id!r}, expected {cut_id!r}")
    return manifest


@dataclass(frozen=True)
class RestoreExpectation:
    """What the restoring run is; every field is compared with the manifest."""

    algorithm: AlgorithmIdentity
    layout: Mapping[str, int]
    backend_fingerprint: str
    local_step: int
    policy_version: int
    epoch: int | None = None  # the cut's epoch must not be newer than this


def verify_cut(
    root: str | os.PathLike[str],
    cut_id: str,
    expect: RestoreExpectation,
    *,
    check_files: bool = True,
) -> CutManifest:
    """Load and fully verify a cut against the restoring run; raise :class:`CutError` on any problem."""
    manifest = load_manifest(root, cut_id)
    problems = manifest.completeness_problems()
    if check_files:
        problems += file_problems(cut_dir(root, cut_id), manifest.files)
    if manifest.algorithm != expect.algorithm:
        problems.append(
            "algorithm identity differs from the current run (spec/plugins/runtime attrs/ref model): "
            f"cut {manifest.algorithm.to_dict()} vs run {expect.algorithm.to_dict()}"
        )
    if dict(manifest.runtime.get("layout") or {}) != dict(expect.layout):
        problems.append(f"parallel layout {manifest.runtime.get('layout')} != current {dict(expect.layout)}")
    if manifest.runtime.get("backend_fingerprint") != expect.backend_fingerprint:
        problems.append("backend fingerprint differs")
    if manifest.progress.local_step != expect.local_step:
        problems.append(f"cut local_step {manifest.progress.local_step} != driver local_step {expect.local_step}")
    if manifest.progress.policy_version != expect.policy_version:
        problems.append(
            f"cut policy_version {manifest.progress.policy_version} != driver policy_version {expect.policy_version}"
        )
    if expect.epoch is not None and manifest.epoch > expect.epoch:
        problems.append(f"cut epoch {manifest.epoch} is newer than the current epoch {expect.epoch}")
    if problems:
        raise CutError(f"cut {cut_id!r} rejected: " + "; ".join(problems))
    return manifest

"""The rollout runtime's two data channels: where weights come from and where groups go.

There is one sampler runtime (:mod:`.engine_sampler_entry`: engine, plain or
agentic rounds, the sandbox gate, incarnations, reclaim). What differs between
deployments is only its data path, chosen by the spec key ``data_path``:

* ``r2`` (default): weights from the run's object store (R2, S3, OSS, a Modal
  Volume or a local directory; MXFP4 partial-expert or bf16 releases plus their
  ``LATEST`` record), groups into the store's sample buffer. No syncer. The
  async GRPO path (Kimi-K3.1 on NC MI355X).
* ``syncer``: weights from the actor syncer's bf16 subscriber broadcast, groups
  through the syncer's sample relay. No store credentials on the sampler. The
  decoupled DiLoCo / spot path (small bf16 models, single-rank trainer layout).
  Needs ``syncer`` (``host:port``), ``channel`` and ``work_dir`` in the spec;
  without ``data_path`` a spec with ``syncer`` selects it.

Each data path is a :class:`DataPath`: a :class:`WeightSource`, a
:class:`SampleSink`, the store that holds this sampler's member, lease and
handoff records, and the path's own stop condition. Everything else -- the
policy follower and its forced-reload bound (``reload_max_defer_s``), drain or
live weight switches, the post-load checks, unique recorder sessions, the
predecessor's orphan sweep, the group provenance the trainer's cross-hardware
guard keys on, the reclaim deadline -- lives in the runtime once and runs on
both paths.

Interfaces
----------

:class:`WeightSource` (what :class:`~.policy_release.PolicyFollower` follows):

* ``latest()`` -> the newest :class:`~.policy_release.PolicyRelease` or None.
  ``release.version`` is the policy version the engine serves after loading it
  (the behavior version every group carries); ``release.seq`` orders releases;
  ``release.policy_hash`` names the weights (release digest / cut digest);
  ``release.buffer_selected`` is the trainer's read position at that release.
* ``resolve(release, *, from_store=False)`` -> a local directory the engine
  loads (download plus digest check on r2, materialize on syncer); raises
  :class:`~.policy_release.ReleaseIncomplete` when the release is not (yet)
  loadable.
* ``records()`` -> the releases known now (newest last).
* ``behind(served_version)`` -> True while the served weights are past the
  run's policy-age limit (rounds wait for a newer load). r2: never.

When to load (immediately, deferred while agentic trials run, forced after
``reload_max_defer_s``, live during a generation) is the follower's decision,
not the source's: both sources get the same semantics.

:class:`SampleSink` (what :class:`~.buffer_payload.GroupWriter`,
:class:`~.sampler_loop.SamplerLoop` and :mod:`.sampler_resume` write and read):

* ``write_payload(sample_id, data)`` -> payload ref; ``append(entry)`` commits
  one group's index entry (its ``provenance`` and ``sampler_round`` travel
  with it on both paths); ``one_commit()``; ``mark_data_end()``;
* ``refresh()``, ``len()``, ``entries()`` (the resume point: what earlier
  incarnations of this shard wrote; the relay keeps none, so a new syncer
  incarnation starts at round 0 with its own sample ids), ``data_ended``;
* ``shard`` (None: unsharded), optional ``pending_groups()`` (the relay's
  backlog), ``shard_entries(shard)`` (a predecessor shard's entries for the
  handoff; the relay: none).

Incarnation and fence: every incarnation claims the island's incarnation lease
in :attr:`DataPath.member_store` and writes no group once fenced. On r2 that
store is shared by every host, so the lease and the pool's fence record decide.
On syncer it is local to the host (``<work_dir>/store``); the syncer's member
table is the fence there (a newer incarnation of the same sampler id
supersedes the older one; :meth:`DataPath.stopping` and
:meth:`DataPath.stop_reason` report it as FENCED).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

DATA_PATH_KEY = "data_path"
R2 = "r2"
SYNCER = "syncer"
DATA_PATHS = (R2, SYNCER)


@runtime_checkable
class WeightSource(Protocol):
    kind: str

    def latest(self) -> Any: ...

    def resolve(self, release: Any, *, from_store: bool = False) -> Path: ...

    def records(self) -> list[Any]: ...

    def behind(self, served_version: int | None) -> bool: ...


@runtime_checkable
class SampleSink(Protocol):
    kind: str
    shard: str | None

    def write_payload(self, sample_id: str, data: bytes) -> str: ...

    def append(self, entry: Any) -> None: ...

    def mark_data_end(self) -> None: ...

    def refresh(self) -> int: ...

    def __len__(self) -> int: ...

    def entries(self) -> list[Any]: ...

    def shard_entries(self, shard: str) -> list[Any]: ...


class DataPath:
    """One sampler process's weight source, sample sink, member store and stop."""

    kind = "custom"
    # whether incarnations on other hosts read the same member store (r2: the
    # bucket; syncer: no, the store is the host's own directory)
    shares_member_store = True

    def __init__(self, weights: Any, samples: Any, member_store: Any) -> None:
        self.weights, self.samples, self.member_store = weights, samples, member_store

    def stopping(self) -> bool:
        """The path's own end: the syncer ended the run, fenced this member, the
        link stayed down, the relay is gone. r2: never (the loop, the lease and
        the data end decide)."""
        return False

    def stop_reason(self) -> tuple[Any, dict[str, Any]] | None:
        """(LeaveReason, detail) to leave with instead of FINISHED, or None."""
        return None

    def close(self) -> None:
        pass


def resolve_data_path(spec: Mapping[str, Any]) -> str:
    """``spec["data_path"]``; without it ``syncer`` when the spec names a syncer, else ``r2``."""
    from .engine_sampler_entry import SamplerSpecError  # noqa: PLC0415

    kind = spec.get(DATA_PATH_KEY)
    if kind is None:
        return SYNCER if spec.get("syncer") else R2
    kind = str(kind)
    if kind not in DATA_PATHS:
        raise SamplerSpecError(f"{DATA_PATH_KEY} must be one of {DATA_PATHS}, got {kind!r}")
    if kind == SYNCER:
        for key in ("syncer", "channel", "work_dir"):
            if spec.get(key) is None:
                raise SamplerSpecError(f"{DATA_PATH_KEY} syncer needs spec.{key}")
    elif spec.get("store") is None:
        raise SamplerSpecError(f"{DATA_PATH_KEY} r2 needs spec.store")
    return kind


def normalize_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """The spec with ``data_path`` filled in and the syncer path's defaults (a local
    member store under ``work_dir``; no round limit: the syncer ends the run)."""
    spec = dict(spec)
    kind = resolve_data_path(spec)
    spec[DATA_PATH_KEY] = kind
    if kind == SYNCER:
        spec.setdefault("num_rollouts", NO_LIMIT_ROUNDS)
        spec.setdefault("store", str(Path(spec["work_dir"]).expanduser() / "store"))
    return spec


NO_LIMIT_ROUNDS = 1 << 40


# ------------------------------------------------------------------------ r2

def _r2_classes() -> tuple[type, type]:
    from .policy_release import PolicyReleaseStore  # noqa: PLC0415
    from .sample_buffer import FileSampleBuffer, shard_index  # noqa: PLC0415

    class R2Release(PolicyReleaseStore):
        """Policy releases (MXFP4 partial or bf16, plus ``LATEST``) in the run's store."""

        kind = R2

        def behind(self, served_version: int | None) -> bool:
            return False

    class R2Buffer(FileSampleBuffer):
        """The store's sample buffer (this sampler's shard index)."""

        kind = R2

        def shard_entries(self, shard: str) -> list[Any]:
            return FileSampleBuffer(self.store, batch_size=1, index=shard_index(str(shard)),
                                    create_index=False).entries()

    return R2Release, R2Buffer


_R2: tuple[type, type] | None = None


def r2_classes() -> tuple[type, type]:
    """``(R2Release, R2Buffer)`` (built once; the engine core imports them lazily)."""
    global _R2
    if _R2 is None:
        _R2 = _r2_classes()
    return _R2


class R2DataPath(DataPath):
    kind = R2
    shares_member_store = True


def open_r2(spec: Mapping[str, Any], environ: Mapping[str, str] | None = None) -> R2DataPath:
    """The r2 data path of ``spec`` (``store``, ``release_dir``, ``buffer_root``,
    ``buffer_shards``; :mod:`.engine_sampler_entry` module doc). Same objects as
    before the split, under their data-path names."""
    from .engine_sampler_entry import SamplerSpecError  # noqa: PLC0415
    from .policy_release import RELEASE_DIR  # noqa: PLC0415
    from .release_delta import default_mirror  # noqa: PLC0415
    from .resume import store_for  # noqa: PLC0415
    from .sample_buffer import INDEX, needs_shards, shard_index  # noqa: PLC0415

    R2Release, R2Buffer = r2_classes()
    if spec.get("store") is None:
        raise SamplerSpecError("spec.store is required")
    store = str(spec["store"])
    releases = R2Release(store_for(store, environ=environ),
                         release_dir=str(spec.get("release_dir") or RELEASE_DIR),
                         mirror=default_mirror(environ, store=store))  # delta releases
    buffer_store = store_for(str(spec.get("buffer_root") or store), environ=environ)
    if spec.get("sampler_id") is None:
        raise SamplerSpecError("spec.sampler_id is required")
    sampler_id = str(spec["sampler_id"])
    shards = [str(s) for s in spec.get("buffer_shards") or ()]
    if shards and sampler_id not in shards:
        raise SamplerSpecError(f"sampler {sampler_id!r} is not one of the buffer shards {shards}")
    index = shard_index(sampler_id) if needs_shards(buffer_store) or shards else INDEX
    buffer = R2Buffer(buffer_store, batch_size=1, index=index)  # as open_buffer(sampler=True)
    return R2DataPath(releases, buffer, buffer_store)


# ------------------------------------------------------------------- factory

class LegacyDataPath(DataPath):
    """A ``stores(incarnation, emit)`` pair of an older caller (no own stop)."""

    def __init__(self, weights: Any, samples: Any, member_store: Any) -> None:
        super().__init__(weights, samples, member_store)
        self.kind = str(getattr(samples, "kind", None) or getattr(weights, "kind", None) or "custom")


def open_data_path(spec: Mapping[str, Any], *, environ: Mapping[str, str] | None, incarnation: str,
                   emit: Callable[..., None]) -> DataPath:
    """The :class:`DataPath` of ``spec["data_path"]`` (after :func:`normalize_spec`)."""
    kind = resolve_data_path(spec)
    if kind == SYNCER:
        from yeto.rl.spot_rollout.syncer_path import open_syncer  # noqa: PLC0415

        return open_syncer(spec, environ, incarnation=incarnation, emit=emit)
    return open_r2(spec, environ)


def member_store_of(spec: Mapping[str, Any], environ: Mapping[str, str] | None = None) -> Any:
    from .resume import store_for  # noqa: PLC0415

    return store_for(str(spec.get("buffer_root") or spec["store"]), environ=environ)

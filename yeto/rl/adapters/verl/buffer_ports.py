"""verl buffer trainer: the ports that differ from the fully_async path.

:func:`install_buffer_ports` turns a :class:`~.fully_async_ports.FullyAsyncIsland`
into a trainer island over the sample buffer (``--rl-sample-role trainer``):

* the island's ``call`` becomes a :class:`~.buffer_feed.BufferCall`: the queue
  calls go to the :class:`~.buffer_feed.BufferFeed`, the rest to the trainer
  actor. The rollout port, the trainer port and the policy state are the
  fully_async ones, unchanged;
* :class:`BufferReleasePublisher` replaces the checkpoint-engine push: the
  trained adapter is merged into the base (:mod:`.merged_release`) and released
  into the store for the samplers (``PolicyReleaseStore``, full weights) with
  the trainer's buffer read position. The pairing ``param_version <-> outer
  version`` is recorded at every publication, as on the fully_async path.

The base policy (version 0, LoRA B all zero: the merged weights are the base)
is not released: the samplers start on the base weights as version 0. A
version-0 state with any non-zero B is released like any other version.

What a release writes comes from the publisher's ``weights`` (VERL-PARITY
#14): :func:`lora_release_weights` (the default: the adapter merged into the
base) or :func:`full_release_weights` (a full-parameter trainer: its FP32 masters
in the model's dtype, written by the island's HF writer, ``hf_export``; version
0 is the base checkpoint itself and is not released).

Synced trainers (``yeto.rl.engine.merged_release``; the same rules as
the Miles trainer island): every island records, at each publication, the
merged version (``merged_version``: the syncer's outer version of the applied
base) it first applied (:class:`~.buffer_feed.MergedVersions`; the feed turns a
group's merged behavior version into a local one with it). Only the publisher
(learner id 0) releases, once per new merged version, with the merged version
as release version, kept increasing by ``MergedReleaseGuard`` (a skipped
merge is reported as ``rl_policy_release_gap``). A publication without a new
merge (elastic: several local rounds per merge) releases nothing: its weights
are not a merged policy. The driver is bound after it is built
(:meth:`BufferReleasePublisher.bind`).

Delta releases (``--rl-release-delta-full-every N`` on a
full-parameter trainer): the release is written into a staging directory on the
island's local disk and published with
:meth:`~yeto.rl.engine.policy_release.PolicyReleaseStore.release_staged`, which
uploads the bf16 delta against the previous release (:class:`DeltaChain`)
instead of the weights. The first release of the process is a delta against the
release the store already holds when its files are readable there -- for a
cross-cloud run that is the base checkpoint both sides have, so the first
update sends only the words one optimizer step moved instead of the whole
checkpoint (Nemotron Nano: 63 GB). It needs the export to write the base's own
file names and sizes, which :mod:`.hf_export` does.

Relaunch (``resume_versions``): a buffer trainer without an
outer sync counts its policy versions itself, so a new process of it continues
the account above the releases its store already holds
(:class:`~yeto.rl.engine.merged_release.ResumedReleases`): the driver's rounds
start at the store's newest release version vK, the island's first publication
is vK and releases nothing, the versions in the policy-age window below vK are
restored with the hashes the store recorded, and the island judges behavior
versions with an outer <-> outer map seeded from those releases. Before this a
relaunched trainer discarded every group a sampler had written with a release
above v0 and re-used version numbers the store already held.

The publication is complete when the release record and ``LATEST.json`` are in
the store. Whether a sampler has loaded it is the sampler's report
(``rl_sampler_policy_loaded`` on its tape) and the behavior versions of the
groups it writes next; the trainer does not wait for it (the samplers run
elsewhere and may be stopped).
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from yeto.rl.contracts import InferencePublicationManifest
from yeto.rl.engine.merged_release import (BufferLane, MergedReleaseGuard, ResumedReleases, is_publisher,
                                           merged_version, resumed_releases)
from yeto.rl.engine.policy_release import FULL, BackgroundReleaser, PolicyReleaseStore
from yeto.rl.engine.sample_buffer import buffer_dir, write_read_position
from yeto.rl.engine.ports import PlacementDescription, PublicationResult

from .buffer_feed import BufferCall, BufferFeed, MergedVersions, RowShape
from .fully_async_ports import MEMBER, FullyAsyncIsland, FullyAsyncRolloutPool
from .fully_async_round import FULLY_ASYNC_PLACEMENT
from .fully_async_translate import VersionMap
from .merged_release import LORA_B, write_full_checkpoint, write_merged_checkpoint

BASE_POLICY_VERSION = 0
RELEASE_EVENT = "rl_policy_released"  # the Miles trainer island's event name
CONSUMED_LOG = "verl-buffer-consumed-{learner}.jsonl"
RELEASE_LOG = "verl-buffer-release-{learner}.jsonl"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def is_base_adapter(version: int, tensors: Mapping[str, Any]) -> bool:
    """Version 0 with every LoRA B zero: base + adapter is the base itself."""
    return int(version) == BASE_POLICY_VERSION and all(
        not bool(t.detach().abs().max() > 0) for name, t in tensors.items() if name.endswith(LORA_B))


class ReleaseWeights:
    """The weights one publication would release: ``write(directory)``
    writes them as a HF checkpoint and returns a summary; ``is_base(version)``: the
    released policy would be the base checkpoint the samplers start on (nothing to
    release); ``fallback_bytes``: the payload size reported when nothing is released."""

    def __init__(self, write: Callable[[Path], dict[str, Any]], is_base: Callable[[int], bool],
                 fallback_bytes: int) -> None:
        self.write = write
        self.is_base = is_base
        self.fallback_bytes = int(fallback_bytes)


def lora_release_weights(base_dir: str | Path, merge: Callable[..., dict[str, Any]] = write_merged_checkpoint):
    """A LoRA trainer's releases: the adapter merged into the base (module doc)."""
    base = Path(base_dir)

    def weights(state: Any) -> ReleaseWeights:
        tensors = dict(state.to_lora().tensors)
        return ReleaseWeights(lambda directory: merge(base, tensors, directory),
                              lambda version: is_base_adapter(version, tensors),
                              # the adapter in bf16 as on the fully_async path
                              sum(int(t.numel()) * 2 for t in tensors.values()))

    return weights


def delta_chain_for(releases: PolicyReleaseStore, full_every: int, *, emit: Callable[..., None] | None = None):
    """A :class:`DeltaChain` for this trainer, started from the release the store
    already holds when that release's files are readable in the store (module
    doc, delta releases); None when ``full_every`` is 0 (deltas off). A release
    that is itself a delta is rebuilt first when the store has a mirror,
    so a restarted island's next release stays a delta."""
    from yeto.rl.engine.policy_release import DeltaChain
    from yeto.rl.engine.release_delta import ReleaseDeltaError

    if int(full_every) < 1:
        return None
    chain = DeltaChain(int(full_every))
    latest = releases.latest()
    if latest is None:
        if emit is not None:
            emit("rl_policy_release_delta_base", adopted=False, rebuilt=False,
                 reason="the store holds no release")
        return chain
    # A delta release is rebuilt first: its own directory holds
    # ``.ydelta`` files, which are no base, but ``PolicyReleaseStore.resolve`` with a
    # mirror rebuilds it on local disk (Nemotron Nano: about 1 min) -- far cheaper than
    # the whole-checkpoint upload the alternative costs (63 GB, about 2.3 h on this link).
    rebuilt = latest.delta is not None
    retried = None
    try:
        try:
            directory = releases.resolve(latest)
        except ReleaseDeltaError as first:
            # A rebuild that fails its digest check may have started from a
            # stale copy in the mirror (shared by every run on the host); rebuild once
            # more from the full release in the store, as a sampler does, before the
            # next release falls back to a whole-checkpoint upload.
            retried = f"{type(first).__name__}: {first}"[:200]
            directory = releases.resolve(latest, from_store=True)
    except Exception as error:  # noqa: BLE001 -- no usable base: the next release goes out whole
        directory = None
        if emit is not None:
            emit("rl_policy_release_delta_base", adopted=False, rebuilt=rebuilt,
                 reason=f"{type(error).__name__}: {error}"[:200],
                 **({} if retried is None else {"first_error": retried}))
    if directory is not None and Path(directory).is_dir():
        # a delta LATEST continues its chain: the full_every bound counts its deltas too
        chain.adopt(latest, directory, since_full=releases.delta_depth(latest, limit=int(full_every)))
        if emit is not None:
            emit("rl_policy_release_delta_base", adopted=True, base_seq=latest.seq,
                 base_version=latest.version, base_path=str(directory), rebuilt=rebuilt,
                 **({} if retried is None else {"retried_from_store": True, "first_error": retried}))
    return chain


def full_release_weights(writer: Callable[[int, Path], None]):
    """A full-parameter trainer's releases: ``writer`` is the island's HF
    writer (``hf_export.VerlHfWriter`` over the trainer workers); version 0 is the
    base checkpoint. Fallback payload: the policy's own payload size."""

    def weights(state: Any) -> ReleaseWeights:
        return ReleaseWeights(lambda directory: write_full_checkpoint(writer, directory),
                              lambda version: int(version) == BASE_POLICY_VERSION,
                              int(state.payload_digest[1]))

    return weights


def release_weights_for(island: FullyAsyncIsland, plan: Mapping[str, Any]):
    """What a release of this buffer trainer writes (:class:`ReleaseWeights`):
    None for a LoRA trainer (the publisher's default: the adapter merged into the
    base); for a full-parameter trainer (VERL-PARITY #14) the trainer's
    own weights, written by the island's HF writer over its full-parameter group
    (``hf_export.VerlHfWriter``; ``island.full_group`` is set by
    ``fully_async_full.full_parameter_ports`` before the ports are installed)."""
    if plan.get("parameter_mode") != "full":
        return None
    from . import megatron_engine
    from .full_parameter_stream import _LoopRunner
    from .hf_export import VerlHfWriter

    if island.full_group is None:
        raise RuntimeError("a full-parameter buffer trainer installs its ports after its policy state")
    if plan.get("engine") == megatron_engine.ENGINE_MEGATRON:
        # A Megatron island's weights are written by its trainer ranks through
        # Megatron-Bridge (the HF boundary exports whole files and serves no single HF
        # tensor to the driver, so VerlHfWriter's export_parameters raises there). Same
        # ReleaseWeights contract, so the store, the delta chain and staging are unchanged.
        from .megatron_ports import MegatronReleaseWriter

        return full_release_weights(MegatronReleaseWriter(island, plan["model_path"]))
    return full_release_weights(VerlHfWriter(island.full_group, plan["model_path"], run=_LoopRunner().run))


class BufferReleasePublisher:
    """yeto publication as a store release of the merged policy (module doc).

    ``feed`` gives the buffer read position; ``base_dir`` is the base HF
    checkpoint the adapter trains over; ``lane``: a synced trainer's buffer lane
    (module doc; None for a single trainer); ``weights``: what a release writes
    (:func:`lora_release_weights` over ``base_dir`` and ``merge`` when None)."""

    def __init__(self, island: FullyAsyncIsland, releases: PolicyReleaseStore, feed: BufferFeed,
                 *, base_dir: str | Path, clock: Callable[[], float] = time.monotonic,
                 merge: Callable[..., dict[str, Any]] = write_merged_checkpoint,
                 lane: BufferLane | None = None,
                 weights: Callable[[Any], ReleaseWeights] | None = None,
                 staging: str | Path | None = None, delta: Any = None,
                 resumed: ResumedReleases | None = None, no_release: bool = False,
                 release_only: Iterable[int] | None = None, background: bool = False) -> None:
        """``no_release`` (``--rl-measure-no-release``): a measurement run
        (memory, MFU) writes no policy release at all. One release of the 120B is about
        225 GB written and a dozen minutes per round, which answers nothing such a run
        asks; every skipped round is logged as ``rl_policy_release_skipped`` with reason
        ``measurement_no_release``, so a sampler never silently misses a release it
        was meant to get. Refused on a synced trainer (its peers wait for the release).

        ``release_only`` (``--rl-release-only-versions``): release only these
        policy versions and skip the others the same way (reason ``not_a_release_version``).
        A multi-step run then trains several rounds on the samples it has and releases once,
        so the samplers load one new policy -- each 120B release costs about an hour of
        transfer on npu1's uplink. Refused on a synced trainer, like ``no_release``.

        ``staging``: publish each release from a directory under it instead of
        writing it into the store (module doc, delta releases); ``delta``: the
        :class:`~yeto.rl.engine.policy_release.DeltaChain` those staged releases
        follow (:func:`delta_chain_for`), which needs ``staging``; ``resumed``: the
        version account this process continues (module doc, relaunch).

        ``background`` (``--rl-release-background``): a staged release writes its
        weights on the training thread, as before (the next optimizer step changes
        them), but encodes, digests and uploads them on a worker thread while the
        next rounds train. One release is in flight at most: the next publication
        that releases (or :meth:`close`) waits for it first and then puts its
        record, events and heartbeat on the tape. A failed background release
        raises there. Single trainer only (a synced trainer's peers wait for
        the release)."""
        if delta is not None and staging is None:
            raise ValueError("delta releases need a staging directory")
        if background and (staging is None or lane is not None):
            raise ValueError("--rl-release-background needs staged releases on a single trainer")
        self.resumed = resumed
        self.staging = None if staging is None else Path(staging)
        self.delta = delta
        self.island = island
        self.releases = releases
        self.feed = feed
        self.base_dir = Path(base_dir)
        self.clock = clock
        self.merge = merge
        self.weights = weights  # None: the LoRA merge over base_dir with self.merge (read at publish)
        self.lane = lane
        self.driver: Any = None
        self.guard = MergedReleaseGuard(self._on_gap)
        self.releasing = lane is None or is_publisher(lane.index, lane.count)
        if (no_release or release_only is not None) and lane is not None:
            raise ValueError("--rl-measure-no-release is for a single trainer: a synced trainer's "
                             "peers wait for its releases")
        self.no_release = bool(no_release)
        self.release_only = None if release_only is None else frozenset(int(v) for v in release_only)
        self.background = bool(background)
        self.releaser = BackgroundReleaser("verl-policy-release") if self.background else None
        # the release in flight: (its record so far, its start, the worker's result)
        self._pending: tuple[dict[str, Any], float, dict[str, Any]] | None = None
        # publication records that came after the release in flight (committed by flush)
        self._deferred: list[tuple[dict[str, Any], Any, dict[str, Any]]] = []

    def bind(self, driver: Any) -> None:
        """The island's driver (synced trainers read its outer version; a resumed
        version account is wired into it here, module doc "relaunch")."""
        self.driver = driver
        if self.resumed is not None:
            # The island's rounds and publications continue at vK,
            # and the versions published below it are restored with the hashes the
            # store recorded, so a group inside the policy-age window is trained.
            resume_at = getattr(driver.sync, "resume_at", None)
            if not callable(resume_at):
                raise RuntimeError(
                    f"a buffer trainer resuming the store's version account needs a sync session "
                    f"that takes a start version; {type(driver.sync).__name__} does not")
            resume_at(self.resumed.start_version)
            driver.published_policy_history = self.resumed.history
        if self.lane is not None:
            self.island.emit("rl_policy_release_role", learner_id=self.lane.index,
                             num_learners=self.lane.count, publisher=self.releasing)

    def _on_gap(self, previous: int, version: int) -> None:
        self.island.emit("rl_policy_release_gap", previous_version=previous, policy_version=version)

    def _release_version(self, state: Any) -> int | None:
        """The version to release, or None (nothing to release). Single trainer:
        the policy version. Synced trainer: the merged version, on the publisher,
        the first time this island applies it (module doc)."""
        local = int(state.policy_version)
        if self.lane is None:
            if self.no_release:  # A measurement run (see __init__)
                self.island.emit("rl_policy_release_skipped", policy_version=local,
                                 reason="measurement_no_release")
                return None
            if self.release_only is not None and local not in self.release_only and local > 0:
                # Only the listed versions are released (see __init__)
                self.island.emit("rl_policy_release_skipped", policy_version=local,
                                 reason="not_a_release_version", release_versions=sorted(self.release_only))
                return None
            if self.resumed is not None and local <= self.resumed.base_version:
                # The first publication of a relaunched trainer is
                # the version its store already holds and nothing is trained yet, so
                # releasing it would write a second release with that version number
                # and other weights. The first trained round releases vK+1.
                self.island.emit("rl_policy_release_skipped", policy_version=local,
                                 latest_version=self.resumed.base_version,
                                 reason="resumed_base_version")
                return None
            return local
        if self.driver is None:
            raise RuntimeError("synced buffer trainer: publish before bind(driver)")
        merged = merged_version(self.driver, state)
        new = self.feed.merged.record(merged, local)
        if not (new and self.releasing):
            return None
        latest = self.releases.latest()
        if self.guard.check(merged, None if latest is None else latest.version) == "stale":
            self.island.emit("rl_policy_release_skipped", policy_version=merged, local_version=local,
                             latest_version=latest.version, reason="not_newer_than_latest")
            return None
        return merged

    def _release_from_staging(self, version: int, digest: str, write: Callable[[Path], None],
                              position: int | None, shards: dict[str, int] | None, state: Any):
        """Write the release into a staging directory and publish it from there,
        as the delta against the chain's base when it has one (module doc).
        The staged directory stays only while it is the next delta's base."""
        staged = self._write_staged(version, write)
        return self._upload_staged(version, digest, staged, position, shards, self._origin(state))

    def _upload_staged(self, version: int, digest: str, staged: Path, position: int | None,
                       shards: dict[str, int] | None, origin: dict[str, Any] | None):
        """The part of a staged release after the weights are written: the delta,
        the digests and the upload (on the worker thread under ``background``).
        Returns ``(release, delta event fields)``; the caller emits the event."""
        import shutil

        base = None if self.delta is None else self.delta.base()
        kept = False
        try:
            release = self.releases.release_staged(
                version, digest, staged, position=lambda: (position, shards),
                weights=FULL, origin=origin, delta_base=base,
                digest_full=self.delta is not None)
            kept = self.delta is not None and self.delta.advance(release, staged)
        finally:
            if not kept:
                shutil.rmtree(staged, ignore_errors=True)
        delta = None if release is None else release.delta
        event = dict(policy_version=int(version),
                     base_seq=None if delta is None else delta["base_seq"],
                     base_version=None if delta is None else delta["base_version"],
                     ratio=None if delta is None else delta["ratio"],
                     bytes=None if release is None else sum(int(v) for v in (release.files or {}).values()),
                     timings=dict(self.releases.last_timings))
        return release, event

    def _origin(self, state: Any) -> dict[str, Any] | None:
        # A release whose policy hash is an identity label (an unverified round
        # under --rl-verify-every N) says so in the release record itself, the one thing a
        # sampler or a later trainer reads; a verified release keeps the record unchanged.
        unverified = {"policy_hash": "unverified-identity"} if not bool(getattr(state, "verified", True)) else {}
        if self.lane is None:
            return ({"kind": "single", **unverified} if unverified else None)
        return {"kind": "merged", "learner_id": self.lane.index, "num_learners": self.lane.count,
                "local_version": int(state.policy_version), **unverified}

    def publish(self, state: Any) -> PublicationResult:
        isl = self.island
        version = int(state.policy_version)
        param_version = int(isl.call("yeto_param_version"))
        isl.vmap.record(param_version, version)  # raises on contradiction / regression
        if isl.consume_map is not isl.vmap:
            # the version the samplers stamp on the groups of this release is the
            # outer version itself; idempotent when it is already seeded
            isl.consume_map.record(version, version)
        digest = state.policy_tensor_hash()
        weights = (self.weights or lora_release_weights(self.base_dir, self.merge))(state)
        started = self.clock()
        merge: dict[str, Any] = {}
        release = None
        position = self.feed.read_cursor()
        shards = self.feed.read_shards()
        released = self._release_version(state)
        releasing = released is not None and not weights.is_base(released)
        if releasing:
            self.flush()  # the release in flight first: the store and the delta chain are serial
            def write(directory: Path) -> None:
                merge.update(weights.write(directory))

            if self.staging is not None and self.background:
                staged = self._write_staged(released, write)
                written = round(self.clock() - started, 3)
                record = self._record(state, version, param_version, digest, position, shards,
                                      None, merge, started)
                record.update(released="pending", write_seconds=written)
                result: dict[str, Any] = {}
                origin = self._origin(state)  # on this thread: the worker never reads ``state``

                def upload() -> None:
                    result["release"], result["event"] = self._upload_staged(
                        released, digest, staged, position, shards, origin)

                self.releaser.submit(upload)
                self._pending = (record, started, result)
                isl.emit("rl_policy_release_background", policy_version=int(released),
                         write_seconds=written)
                return self._finish_publication(version, param_version, digest, None, weights)
            elif self.staging is not None:
                release, event = self._release_from_staging(released, digest, write, position, shards, state)
                isl.emit("rl_policy_release_delta", **event)
            else:
                release = self.releases.release(released, digest, write, buffer_selected=position,
                                                buffer_shards=shards, weights=FULL,
                                                origin=self._origin(state))
        record = self._record(state, version, param_version, digest, position, shards, release,
                              merge, started)
        if self._pending is not None:  # after the release in flight: the log stays in version order
            self._deferred.append((record, release, merge))
        else:
            self._commit_record(record, release, merge)
        return self._finish_publication(version, param_version, digest, record.get("path"), weights,
                                        record.get("payload_bytes"))

    # -- background releases ------------------------------------------------
    def _write_staged(self, version: int, write: Callable[[Path], None]) -> Path:
        import shutil
        import tempfile

        self.staging.mkdir(parents=True, exist_ok=True)
        staged = Path(tempfile.mkdtemp(prefix=f"v{int(version)}-", dir=self.staging))
        try:
            write(staged)
        except BaseException:
            shutil.rmtree(staged, ignore_errors=True)
            raise
        return staged

    def flush(self) -> None:
        """Wait for the background release in flight (if any) and put its record,
        events and heartbeat on the tape; its error, if it failed, raises here."""
        if self._pending is None:
            return
        record, started, result = self._pending
        self._pending = None
        waited = self.releaser.wait()
        release = result["release"]
        self.island.emit("rl_policy_release_delta", **result["event"])
        # seconds: publication start to upload end; wait_seconds: how long the
        # training thread blocked on it (0 when the upload hid behind training)
        record.update(released=release is not None, seconds=round(self.clock() - started, 3),
                      wait_seconds=round(waited, 3))
        self._commit_record(record, release, record.get("merge") or {})
        deferred, self._deferred = self._deferred, []
        for args in deferred:
            self._commit_record(*args)

    def close(self) -> None:
        """The end of the run: the release in flight goes out before the island reports done."""
        self.flush()

    # -- records ------------------------------------------------------------
    def _record(self, state: Any, version: int, param_version: int, digest: str,
                position: int | None, shards: dict[str, int] | None, release: Any,
                merge: dict[str, Any], started: float) -> dict[str, Any]:
        """The publication log line (``RELEASE_LOG``) before the release fields."""
        isl = self.island
        # Under --rl-verify-every N a round the driver does not verify carries
        # an identity label (policy_digest.unverified_identity), not a content hash. Say
        # which one this is, so nothing downstream takes the label for the weights' hash.
        verified = bool(getattr(state, "verified", True))
        record = {"policy_version": version, "param_version": param_version, "policy_hash": digest,
                  "policy_hash_verified": verified,
                  "released": release is not None, "buffer_selected": position, "buffer_shards": shards,
                  "seconds": round(self.clock() - started, 3), "merge": merge or None,
                  "version_map": isl.vmap.to_dict()}
        if self.lane is not None:
            record.update(merged_version=merged_version(self.driver, state),
                          merged_versions=self.feed.merged.to_dict())
        return record

    def _commit_record(self, record: dict[str, Any], release: Any, merge: dict[str, Any]) -> None:
        """The release event, the publication log line and the heartbeat release."""
        isl = self.island
        version, digest = int(record["policy_version"]), record["policy_hash"]
        position = record["buffer_selected"]
        verified = bool(record["policy_hash_verified"])
        if release is not None:
            record.update(seq=release.seq, path=release.path, release_version=release.version,
                          payload_bytes=sum(int(v) for v in (release.files or {}).values()))
            isl.emit(RELEASE_EVENT, seq=release.seq, policy_version=release.version,
                     local_version=version, origin=release.origin, policy_hash=digest,
                     buffer_selected=position, weights=FULL, sha256=release.sha256,
                     policy_hash_verified=verified,
                     engine_checksum=None, merged_modules=merge.get("merged_modules"),
                     max_relative_change=merge.get("max_relative_change"),
                     seconds=record["seconds"])
        isl.append_jsonl(RELEASE_LOG.format(learner=isl.learner_id), record)
        if release is not None:
            isl.heartbeat_release = {"seq": int(release.seq), "policy_hash": str(release.policy_hash)}

    def _finish_publication(self, version: int, param_version: int, digest: str,
                            path: str | None, weights: ReleaseWeights,
                            payload_bytes: int | None = None) -> PublicationResult:
        isl = self.island
        isl.emit("rl_verl_version_map", param_version=param_version, outer_version=version)
        isl.published = (version, digest)
        isl.hashes[version] = digest
        manifest = InferencePublicationManifest(
            publication_mode="full", base_policy_version=None, target_policy_version=version,
            target_policy_hash=digest, target_manifest_hash=_sha(f"{version}:{digest}"),
            payload_hash=_sha(json.dumps({"release": path, "hash": digest})),
            # the release's files, or (nothing released, or still uploading) the
            # weights' fallback size
            payload_bytes=int(payload_bytes or weights.fallback_bytes),
            complete=True)
        return PublicationResult(manifest, frozenset({MEMBER}))


class BufferPlacement:
    """The trainer GPUs only: the samplers run on other islands."""

    def describe(self) -> PlacementDescription:
        return PlacementDescription(FULLY_ASYNC_PLACEMENT, ("cuda:trainer-0",), ("store:sample-buffer",),
                                    extra={"engine": "verl-buffer-trainer"})


def install_buffer_ports(island: FullyAsyncIsland, *, buffer: Any, releases: PolicyReleaseStore,
                         shape: RowShape, base_dir: str | Path,
                         tokenize: Callable[[str], Sequence[int]] | None = None,
                         wait_s: float = 3600.0, poll_s: float = 2.0, cursor_log: Any = None,
                         weights: Callable[[Any], ReleaseWeights] | None = None,
                         staging: str | Path | None = None, delta_full_every: int = 0,
                         resume_versions: bool = False, no_release: bool = False,
                         release_only: Iterable[int] | None = None, release_background: bool = False,
                         **feed_kwargs: Any) -> tuple[FullyAsyncRolloutPool, BufferReleasePublisher,
                                                      BufferPlacement]:
    """Route the island's queue calls to a buffer feed and return the rollout
    port, the publisher and the placement of a buffer trainer (module doc); the
    feed is ``publisher.feed``. A lane reader ``buffer`` (``buffer.lane``) makes
    a synced trainer: ``cursor_log`` is its lane cursor log
    (``merged_release.lane_cursor_log``).

    ``delta_full_every`` > 0 (``--rl-release-delta-full-every``) publishes the
    releases from ``staging`` as bf16 deltas (module doc, delta releases).

    ``resume_versions``: this island counts its policy
    versions itself (no outer sync), so it continues the account above the
    releases its store already holds (``merged_release.ResumedReleases``) --
    its round counter, its version map and its published policies all start at
    the store's newest release version. A synced trainer takes its versions
    from the syncer and refuses this."""
    lane = getattr(buffer, "lane", None)
    if lane is not None and cursor_log is None:
        raise ValueError("a synced buffer trainer needs both a lane reader and its cursor log")
    if resume_versions and lane is not None:
        raise ValueError("a synced buffer trainer's versions come from the syncer, not from the store")
    actor_call = island.call
    log = CONSUMED_LOG.format(learner=island.learner_id)

    heartbeat_root = buffer_dir(buffer)
    # the newest release this trainer knows: the store's at start, then each one it
    # publishes (BufferReleasePublisher.publish); it ties the heartbeat to this run
    island.heartbeat_release = None
    if heartbeat_root is not None:
        try:
            newest = releases.latest()
        except Exception:  # noqa: BLE001 - a heartbeat without a release is still valid
            newest = None
        if newest is not None:
            island.heartbeat_release = {"seq": int(newest.seq), "policy_hash": str(newest.policy_hash)}

    def record(fields: dict[str, Any]) -> None:
        island.append_jsonl(log, fields)
        if heartbeat_root is not None:  # the read position, released or not
            try:
                write_read_position(heartbeat_root, buffer_selected=feed.read_cursor(),
                                    buffer_shards=feed.read_shards(),
                                    train_step=int(island.published[0]) if island.published else None,
                                    release=getattr(island, "heartbeat_release", None))
            except Exception as error:  # noqa: BLE001 - a heartbeat never stops training
                island.emit("rl_buffer_read_position_failed", error=str(error)[:300])

    # A buffered group's behavior version is a store release version (an outer
    # version), not a verl param version, so the round judges it with its own map.
    island.consume_map = VersionMap()
    resumed = resumed_releases(releases, island.limit) if resume_versions else None
    if resumed is not None:
        for version in resumed.known_versions():
            island.consume_map.record(version, version)
        island.hashes.update(resumed.window())
        island.emit("rl_buffer_resume_versions", base_version=resumed.base_version,
                    start_version=resumed.start_version, max_policy_age=resumed.limit,
                    known_versions=list(resumed.known_versions()),
                    restored_versions=sorted(resumed.window()))
    feed = BufferFeed(buffer, vmap=lambda: island.consume_map, shape=shape,
                      add_group=lambda rows: actor_call("yeto_add_group", rows),
                      remove_group=lambda uid: actor_call("yeto_remove_group", uid), record=record,
                      tokenize=tokenize, wait_s=wait_s, poll_s=poll_s, cursor_log=cursor_log,
                      merged=None if lane is None else MergedVersions(), **feed_kwargs)
    island.call = BufferCall(feed, actor_call, round_id=lambda: int(island.published[0]))
    island.recheck_kept = feed.recheck_kept  # the round judged again before assembly
    island.start_rollouter = None  # no rollouter: the samplers run elsewhere
    if int(delta_full_every) > 0 and staging is None:
        raise ValueError("--rl-release-delta-full-every needs a staging directory")
    delta = (delta_chain_for(releases, int(delta_full_every), emit=island.emit)
             if int(delta_full_every) > 0 else None)
    publisher = BufferReleasePublisher(island, releases, feed, base_dir=base_dir, lane=lane,
                                       weights=weights,
                                       staging=staging if int(delta_full_every) > 0 else None,
                                       delta=delta, resumed=resumed, no_release=no_release,
                                       release_only=release_only, background=release_background)
    return FullyAsyncRolloutPool(island, queue_end_is_buffer_end=True), publisher, BufferPlacement()

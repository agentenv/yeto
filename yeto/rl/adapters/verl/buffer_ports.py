"""verl buffer trainer: the ports that differ from the fully_async path (S25 C2).

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
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from yeto.rl.contracts import InferencePublicationManifest
from yeto.rl.engine.policy_release import FULL, PolicyReleaseStore
from yeto.rl.engine.ports import PlacementDescription, PublicationResult

from .buffer_feed import BufferCall, BufferFeed, RowShape
from .fully_async_ports import MEMBER, FullyAsyncIsland, FullyAsyncRolloutPool
from .fully_async_round import FULLY_ASYNC_PLACEMENT
from .merged_release import LORA_B, write_merged_checkpoint

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


class BufferReleasePublisher:
    """yeto publication as a store release of the merged policy (module doc).

    ``feed`` gives the buffer read position; ``base_dir`` is the base HF
    checkpoint the adapter trains over."""

    def __init__(self, island: FullyAsyncIsland, releases: PolicyReleaseStore, feed: BufferFeed,
                 *, base_dir: str | Path, clock: Callable[[], float] = time.monotonic,
                 merge: Callable[..., dict[str, Any]] = write_merged_checkpoint) -> None:
        self.island = island
        self.releases = releases
        self.feed = feed
        self.base_dir = Path(base_dir)
        self.clock = clock
        self.merge = merge

    def publish(self, state: Any) -> PublicationResult:
        isl = self.island
        version = int(state.policy_version)
        param_version = int(isl.call("yeto_param_version"))
        isl.vmap.record(param_version, version)  # raises on contradiction / regression
        digest = state.policy_tensor_hash()
        tensors = dict(state.to_lora().tensors)
        started = self.clock()
        merge: dict[str, Any] = {}
        release = None
        position = self.feed.read_cursor()
        if not is_base_adapter(version, tensors):
            def write(directory: Path) -> None:
                merge.update(self.merge(self.base_dir, tensors, directory))

            release = self.releases.release(version, digest, write, buffer_selected=position, weights=FULL)
        record = {"policy_version": version, "param_version": param_version, "policy_hash": digest,
                  "released": release is not None, "buffer_selected": position,
                  "seconds": round(self.clock() - started, 3), "merge": merge or None,
                  "version_map": isl.vmap.to_dict()}
        if release is not None:
            record.update(seq=release.seq, path=release.path,
                          payload_bytes=sum(int(v) for v in (release.files or {}).values()))
            isl.emit(RELEASE_EVENT, seq=release.seq, policy_version=version, policy_hash=digest,
                     buffer_selected=position, weights=FULL, sha256=release.sha256,
                     engine_checksum=None, merged_modules=merge.get("merged_modules"),
                     max_relative_change=merge.get("max_relative_change"),
                     seconds=record["seconds"])
        isl.append_jsonl(RELEASE_LOG.format(learner=isl.learner_id), record)
        isl.emit("rl_verl_version_map", param_version=param_version, outer_version=version)
        isl.published = (version, digest)
        isl.hashes[version] = digest
        manifest = InferencePublicationManifest(
            publication_mode="full", base_policy_version=None, target_policy_version=version,
            target_policy_hash=digest, target_manifest_hash=_sha(f"{version}:{digest}"),
            payload_hash=_sha(json.dumps({"release": record.get("path"), "hash": digest})),
            # the release's files, or (the base, not released) the adapter in bf16 as on the fully_async path
            payload_bytes=int(record.get("payload_bytes") or sum(int(t.numel()) * 2 for t in tensors.values())),
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
                         wait_s: float = 3600.0, poll_s: float = 2.0,
                         **feed_kwargs: Any) -> tuple[FullyAsyncRolloutPool, BufferReleasePublisher,
                                                      BufferPlacement]:
    """Route the island's queue calls to a buffer feed and return the rollout
    port, the publisher and the placement of a buffer trainer (module doc); the
    feed is ``publisher.feed``."""
    actor_call = island.call
    log = CONSUMED_LOG.format(learner=island.learner_id)

    def record(fields: dict[str, Any]) -> None:
        island.append_jsonl(log, fields)

    feed = BufferFeed(buffer, vmap=lambda: island.vmap, shape=shape,
                      add_group=lambda rows: actor_call("yeto_add_group", rows), record=record,
                      tokenize=tokenize, wait_s=wait_s, poll_s=poll_s, **feed_kwargs)
    island.call = BufferCall(feed, actor_call)
    island.start_rollouter = None  # no rollouter: the samplers run elsewhere
    publisher = BufferReleasePublisher(island, releases, feed, base_dir=base_dir)
    return FullyAsyncRolloutPool(island), publisher, BufferPlacement()

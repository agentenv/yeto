"""verl relay training island (S27 G-EV; SPOT-ROLLOUT.md §12.9 item 1).

A *relay trainer* is a synced full-parameter buffer trainer
(``--rl-sample-role trainer``, VERL-PARITY #14) whose groups come from spot
sampler islands through the head's sample relay (``--rl-sample-relay``,
:mod:`yeto.rl.spot_rollout`), not from a store buffer that its own releases
feed. The engine-neutral rules are G-EM's (``yeto.rl.engine.external_policy``):
the samplers serve merged cuts the syncer broadcast to its receive-only
subscribers, so a group names the *merged version* of its cut, and its age is
counted in merged versions against the trainer's merged version at the
publication the round trains from (negative ages, a cut the trainer applies at
its next boundary, are accepted). The verl ports of that rule:

* the buffer is a local one that :func:`~yeto.rl.spot_rollout.relay_buffer.
  maybe_start_relay_pump` fills from the island's relay channel
  (:func:`relay_buffer`); the trainer reads all of it (one lane);
* :class:`RelayPublisher` releases nothing (the samplers follow the syncer);
  it records the param-version pairing as the other verl publishers and sets
  the driver's ``external_policy`` when it is bound, so the driver checks every
  batch with ``ExternalMergedCuts`` (``IslandDriver._accept_external_batch``);
* :class:`RelayRolloutPool` collects a round in merged versions
  (:class:`RelayRoundCollector`: the same keep/discard bookkeeping as the
  fully_async collector, ages in merged versions) and names each group by the
  merged versions of its samples (``policy_versions``), the unit
  ``external_policy.group_merged_version`` reads.

The verl rows carry the merged version as ``min_global_steps`` /
``max_global_steps``; verl's trainer reads them only for its staleness metrics
(fork ``detach_utils.py:151``). verl writes no trajectory evidence files, so
there is no schema-5 branch to add; the driver is the only identity check
before training (verl's trainer has none of its own).
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from yeto.rl.engine.ports import PublicationResult

from .buffer_feed import BufferCall, BufferFeed, RowShape
from .buffer_ports import CONSUMED_LOG, BufferPlacement
from .fully_async_ports import FullyAsyncIsland, FullyAsyncRolloutPool
from .fully_async_round import GroupVerdict, RoundCollector, SampleMeta
from .fully_async_translate import VerlTranslationError

RELAY_SECTION_KEY = "relay"  # in the plan's ``sample_buffer`` section (island_entry)
RELAY_BUFFER_DIR = "yeto-relay-buffer"


class IdentityVersions:
    """The version map of a relay trainer's feed and collector: the samples' versions
    are merged versions already, and they stay merged versions."""

    @staticmethod
    def local(version: int) -> int:
        return int(version)

    @staticmethod
    def outer(version: int) -> int:
        return int(version)


@dataclass
class RelayRoundCollector(RoundCollector):
    """:class:`RoundCollector` in merged versions: ``current_outer`` is the trainer's
    merged version at the round's publication; a group is kept when its oldest merged
    version is at most ``limit`` behind it (a newer cut is kept, as the driver keeps
    it); ``older`` counts the tokens of versions behind ``current_outer``."""

    def offer(self, meta: SampleMeta) -> bool:
        if self.full:
            raise VerlTranslationError("round already full")
        versions = sorted({int(v) for i in range(meta.n) for v in (meta.min_steps[i], meta.max_steps[i])})
        if not versions:
            verdict = GroupVerdict(meta, (), "unknown version: no merged version", True, 0, 0, 0)
        else:
            age = int(self.current_outer) - versions[0]
            older = sum(length for i, length in enumerate(meta.lengths)
                        if int(meta.min_steps[i]) < int(self.current_outer))
            reason = None if age <= int(self.limit) else (
                f"merged version {versions[0]} is {age} behind {self.current_outer} (max_policy_age={self.limit})")
            verdict = GroupVerdict(meta, tuple(versions), reason, False, older, 0, 0)
        (self.discarded if verdict.reason else self.kept).append(verdict)
        return verdict.reason is None


def relay_group_metadata(collector: RelayRoundCollector, rollout_id: int):
    """Kept groups as :class:`GroupMetadata` named by their merged versions: the token is
    the oldest merged version (a plain version number, as a spot sampler stamps its
    engine) and ``policy_versions`` lists every merged version of the group."""
    from yeto.rl.engine.ports import GroupMetadata

    out = []
    for verdict in collector.kept:
        meta = verdict.meta
        rewards = [r for r in meta.rewards if not math.isnan(r)]
        mean = sum(rewards) / len(rewards) if rewards else float("nan")
        std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards)) if rewards else float("nan")
        out.append(GroupMetadata(
            group_id=f"r{rollout_id}-{meta.uid}",
            sample_ids=tuple(f"r{rollout_id}-{meta.uid}-{k}" for k in range(meta.n)),
            policy_token=str(min(verdict.versions)), reward_mean=mean, reward_std=std,
            token_count=sum(meta.lengths), policy_versions=tuple(verdict.versions)))
    return tuple(out)


class RelayRolloutPool(FullyAsyncRolloutPool):
    """The fully_async rollout port over the relay buffer (module doc)."""

    def __init__(self, island: FullyAsyncIsland, cuts: Callable[[], Any]):
        # S29 spot run (defect D10): no cap on discarded pulls. A spot sampler that
        # restarted on older weights, or that loads cuts slowly over a long WAN, sends
        # groups past the policy-age limit for a while; they are discarded (and
        # recorded) and the round waits for newer ones. With verl's 8 x required cap
        # the whole trainer left INFRA_FAILED after one such stretch.
        super().__init__(island, queue_end_is_buffer_end=True, max_pulls_factor=0)
        self.cuts = cuts  # the driver's ExternalMergedCuts, once the publisher is bound

    def _merged_now(self, rollout_id: int) -> int:
        cuts = self.cuts()
        merged = None if cuts is None else cuts.merged_at(rollout_id)
        if merged is None:
            raise RuntimeError(f"rollout {rollout_id}: no merged version recorded for publication v{rollout_id} "
                               "(the relay publisher is bound to the driver before its first publication)")
        return int(merged)

    def _collector(self, rollout_id: int) -> RoundCollector:
        isl = self.island
        return RelayRoundCollector(IdentityVersions(), self._merged_now(rollout_id), isl.limit, isl.required)

    def _group_metadata(self, collector: RoundCollector, rollout_id: int):
        return relay_group_metadata(collector, rollout_id)


class RelayPublisher:
    """A relay trainer's publication: no release (the spot samplers subscribe to the
    syncer); the param-version pairing and the published hash as on the other verl
    publishers. :meth:`bind` names the driver's groups as external merged cuts."""

    def __init__(self, island: FullyAsyncIsland, feed: BufferFeed) -> None:
        self.island = island
        self.feed = feed
        self.driver: Any = None

    def bind(self, driver: Any) -> None:
        from yeto.rl.engine.external_policy import ExternalMergedCuts

        self.driver = driver
        driver.external_policy = ExternalMergedCuts()
        self.island.emit("rl_policy_release_role", learner_id=self.island.learner_id, publisher=False,
                         relay=os.environ.get("YETO_SAMPLE_RELAY_ADDR"))

    def publish(self, state: Any) -> PublicationResult:
        import hashlib

        from yeto.rl.contracts import InferencePublicationManifest
        from yeto.rl.engine.merged_release import merged_version

        from .fully_async_ports import MEMBER

        if self.driver is None:
            raise RuntimeError("relay trainer: publish before bind(driver)")
        isl = self.island
        version = int(state.policy_version)
        check = getattr(state, "check_holds", None)
        if callable(check):
            check()
        param_version = int(isl.call("yeto_param_version"))
        isl.vmap.record(param_version, version)  # raises on contradiction / regression
        digest = state.policy_tensor_hash()
        payload_hash, payload_bytes = state.payload_digest
        record = {"policy_version": version, "param_version": param_version, "policy_hash": digest,
                  "released": False, "relay": True, "merged_version": merged_version(self.driver, state),
                  "buffer_selected": self.feed.read_cursor(), "version_map": isl.vmap.to_dict()}
        isl.append_jsonl(f"verl-relay-publish-{isl.learner_id}.jsonl", record)
        isl.emit("rl_verl_version_map", param_version=param_version, outer_version=version)
        isl.published = (version, digest)
        isl.hashes[version] = digest
        manifest = InferencePublicationManifest(
            publication_mode="full", base_policy_version=None, target_policy_version=version,
            target_policy_hash=digest,
            target_manifest_hash=hashlib.sha256(f"{version}:{digest}".encode()).hexdigest(),
            payload_hash=payload_hash, payload_bytes=int(payload_bytes), complete=True)
        return PublicationResult(manifest, frozenset({MEMBER}))


def install_relay_ports(island: FullyAsyncIsland, *, buffer: Any, shape: RowShape,
                        tokenize: Callable[[str], Sequence[int]] | None = None, wait_s: float = 3600.0,
                        poll_s: float = 2.0) -> tuple[RelayRolloutPool, RelayPublisher, BufferPlacement]:
    """The rollout port, publisher and placement of a relay trainer over ``buffer`` (the
    local buffer the relay pump fills); the island's queue calls go to the feed."""
    actor_call = island.call
    log = CONSUMED_LOG.format(learner=island.learner_id)
    feed = BufferFeed(buffer, vmap=IdentityVersions, shape=shape,
                      add_group=lambda rows: actor_call("yeto_add_group", rows),
                      record=lambda fields: island.append_jsonl(log, fields),
                      tokenize=tokenize, wait_s=wait_s, poll_s=poll_s)
    island.call = BufferCall(feed, actor_call, round_id=lambda: int(island.published[0]))
    island.start_rollouter = None  # no rollouter: the spot samplers generate
    publisher = RelayPublisher(island, feed)
    pool = RelayRolloutPool(island, lambda: getattr(publisher.driver, "external_policy", None))
    return pool, publisher, BufferPlacement()


def relay_buffer(learner_id: int, *, tmp_root: str | os.PathLike[str] | None = None,
                 environ: Mapping[str, str] | None = None) -> tuple[Any, Any]:
    """``(buffer, pump)``: the trainer's local buffer and the relay pump that fills it
    (``maybe_start_relay_pump``, reading the relay variables of ``environ``)."""
    import tempfile

    from yeto.rl.engine.resume import CheckpointStore
    from yeto.rl.engine.sample_buffer import FileSampleBuffer
    from yeto.rl.spot_rollout.relay_buffer import RELAY_ADDR_ENV, maybe_start_relay_pump

    env = os.environ if environ is None else environ
    if not env.get(RELAY_ADDR_ENV):
        raise RuntimeError(f"a relay trainer needs {RELAY_ADDR_ENV} (the island entry sets it)")
    root = Path(tmp_root or tempfile.gettempdir()) / RELAY_BUFFER_DIR / f"island{int(learner_id)}"
    root.mkdir(parents=True, exist_ok=True)
    buffer = FileSampleBuffer(CheckpointStore(root), batch_size=1, create_index=True)
    return buffer, maybe_start_relay_pump(buffer, environ=env)

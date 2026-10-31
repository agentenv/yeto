"""The spot sampler island (S27 G-EA) on the engine sampler island's code.

A spot sampler is the engine sampler island (:mod:`yeto.rl.engine.engine_sampler`,
:mod:`yeto.rl.engine.engine_sampler_entry`): ``SamplerLoop`` admits rounds
by backlog, ``PolicyFollower`` loads new policies through ``PortReleaseLoader``
with the post-load checks, ``EngineRound`` generates, scores and writes each
group with ``GroupWriter``, ``SamplerMembership`` and the reclaim marker
handle replacement. Only its two channels differ:

* weights: :class:`~.release_store.SubscriberReleaseStore` (a syncer
  subscriber, no store) instead of the volume's ``PolicyReleaseStore``;
* samples: :class:`~.relay_buffer.RelaySampleBuffer` (the head's relay)
  instead of the volume's ``FileSampleBuffer``.

It stops when the syncer says the run finished, when the syncer link stays
down for ``syncer_lost_s``, when the relay did not answer for ``relay_gone_s``,
or on the reclaim marker (PREEMPTED). Every relay item names the cut that
generated each of its versions (S27 G-EM); the base weights served before the
first cut are named by the digest of ``base_dir``.

``python -m yeto.rl.spot_rollout.sampler --spec SPEC.json``: the engine
sampler spec (``engine``, ``sampling``, ``prompts``, ``reward``, ``checks``,
``event_tape``, ``max_prompt_tokens``, ``max_backlog_groups``, ``base_dir``)
plus ``syncer`` (host:port of the actor syncer that serves the subscribers and
the relay), ``channel`` (the training island it feeds), ``base_model`` (HF
directory with config and tokenizer; default ``base_dir``), ``work_dir``;
optional ``num_streams`` (4), ``reload_every`` (1 merged version),
``converter`` (``module:callable``; default Miles' ``megatron_to_hf``),
``model_name``, ``layout_timeout_s`` (1800), ``syncer_lost_s`` (1800), ``relay_gone_s`` (900),
``num_rollouts`` (no limit), ``store`` (local membership records; default
``<work_dir>/store``), ``telemetry`` (false: no device telemetry). The subscriber id is
:func:`~.release_store.subscriber_id_for` of ``sampler_id`` (the task
dispenser's sampler holder id); the HMAC key comes from ``YETO_ISLAND_HMAC_KEY``.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

NO_LIMIT_ROUNDS = 1 << 40


class SpotChannels:
    """The subscriber and relay connections of one spot sampler process."""

    def __init__(self, spec: Mapping[str, Any], environ: Mapping[str, str] | None = None, *,
                 incarnation: str, emit: Callable[..., None]) -> None:
        from yeto.protocol import island_hmac_key_from_env, layout_fingerprint
        from yeto.rl.engine.task_dispense import member_incarnation

        from . import layout_manifest
        from .materialize import converter_from_spec, materialize
        from .relay import RelayBoard, RelayProducer
        from .relay_buffer import RelaySampleBuffer, parse_addr
        from .release_store import SubscriberReleaseStore, subscriber_id_for
        from .subscriber import SubscriberClient, WeightFollower

        self.addr = parse_addr(str(spec["syncer"]))
        key = island_hmac_key_from_env(environ)
        self.subscriber_id = subscriber_id_for(str(spec["sampler_id"]))
        board = RelayBoard(self.addr, layout_manifest.LAYOUT_BOARD, self.subscriber_id, key=key)
        try:
            self.layout, self.meta = layout_manifest.fetch(
                board, timeout_s=float(spec.get("layout_timeout_s", 1800.0)))
        finally:
            board.close()
        # One member incarnation for the process's subscriber, relay producer and
        # task dispenser connections (engine_sampler_entry.run derives it from the
        # process incarnation); a newer process of this sampler supersedes all three.
        self.incarnation = member_incarnation(environ)
        self.client = SubscriberClient(self.addr, self.subscriber_id, self.layout,
                                       num_streams=int(spec.get("num_streams", 4)), hmac_key=key,
                                       incarnation=self.incarnation)
        self.client.start()
        self.follower = WeightFollower(self.client, self.layout.num_fragments)
        # The producer's stream is random per connection object: a restarted writer
        # never reuses the sequence numbers of an earlier one.
        self.producer = RelayProducer(self.addr, int(spec["channel"]), self.subscriber_id, key=key,
                                      incarnation=self.incarnation,
                                      gone_s=float(spec.get("relay_gone_s", 900.0)))
        self.syncer_lost_s = float(spec.get("syncer_lost_s", 1800.0))
        self._lost_since: float | None = None
        convert = converter_from_spec(spec)
        work = Path(spec["work_dir"]).expanduser()
        work.mkdir(parents=True, exist_ok=True)
        base_model = spec.get("base_model") or spec.get("base_dir")
        self.releases = SubscriberReleaseStore(
            self.follower,
            lambda cut: materialize(self.layout, cut, base_model=base_model, out_root=work / "weights",
                                    convert=convert, model_name=spec.get("model_name")),
            read_position=lambda: self.buffer.read_position(),
            reload_every=int(spec.get("reload_every", 1)), finished=self.client.run_finished,
            layout_fingerprint=layout_fingerprint(self.layout).hex(),
            base_version=int(spec.get("base_version", 0)), base_dir=spec.get("base_dir") or base_model)
        from .version_board import VERSION_BOARD, VersionReader

        # S29 D9/D4: the run's merged version, read at every round admission
        self.version = VersionReader(lambda: RelayBoard(self.addr, VERSION_BOARD, self.subscriber_id, key=key))
        if spec.get("version_board", True):  # False: drills that need stale groups (S29 D10)
            self.releases.version_reader = self.version
        self.buffer = RelaySampleBuffer(self.producer, fragment_versions=self.releases.fragment_versions,
                                        should_stop=self.stopping, cut_of=self.releases.cut_identity)
        emit("rl_spot_sampler_channels", syncer=f"{self.addr[0]}:{self.addr[1]}",
             subscriber_id=self.subscriber_id, channel=int(spec["channel"]),
             member_incarnation=f"{self.incarnation:016x}",
             fragments=self.layout.num_fragments, layout_meta=self.meta)

    def syncer_lost(self) -> bool:
        if self.client._connected.is_set():
            self._lost_since = None
            return False
        now = time.monotonic()
        self._lost_since = now if self._lost_since is None else self._lost_since
        return now - self._lost_since >= self.syncer_lost_s

    def stopping(self) -> bool:
        # fenced: a newer process of this sampler took its id on the head
        # version.regressed: the board went back below a version this sampler read,
        # i.e. a fresh syncer replaced the run it joined (S29 D4)
        return (self.client.run_finished() or self.client.fenced() or self.syncer_lost()
                or self.buffer.gone or self.version.regressed)

    def stop_reason(self) -> tuple[Any, dict[str, Any]] | None:
        """(LeaveReason, detail) of a fence stop, None otherwise (the island
        then leaves FINISHED as before). S29 D4: a sampler that left because a
        fresh syncer replaced its run leaves FENCED, fenced="other run"."""
        from yeto.rl.pool_client import LeaveReason

        if self.version.regressed:
            return LeaveReason.FENCED, {"fenced": "other run", "run_merged_version": self.version.merged_version,
                                        "highest_merged_version": self.version.highest}
        if self.client.fenced():
            return LeaveReason.FENCED, {"fenced": "superseded"}
        return None

    def close(self) -> None:
        try:
            self.client.close()
        finally:
            self.producer.close()
            self.version.close()


def run_spec(spec: Mapping[str, Any], *, environ: Mapping[str, str] | None = None,
             incarnation: str | None = None, resource_sampler: Any = None) -> tuple[int, SpotChannels | None]:
    """Run the engine sampler island of ``spec`` on the spot channels; returns
    (rounds, channels)."""
    from yeto.rl.engine import engine_sampler_entry as entry
    from yeto.rl.engine.sampler_resume import new_incarnation

    spec = dict(spec)
    spec.setdefault("num_rollouts", NO_LIMIT_ROUNDS)
    spec.setdefault("store", str(Path(spec["work_dir"]).expanduser() / "store"))
    incarnation = incarnation or new_incarnation()
    made: dict[str, SpotChannels] = {}

    def stores(inc: str, emit: Callable[..., None]) -> tuple[Any, Any]:
        made["channels"] = SpotChannels(spec, environ, incarnation=inc, emit=emit)
        return made["channels"].releases, made["channels"].buffer

    if resource_sampler is None and spec.get("telemetry") is False:
        resource_sampler = lambda emit: None  # noqa: E731 - CPU tests: no device telemetry
    try:
        rounds = entry.run(spec, environ=environ, incarnation=incarnation, stores=stores,
                           stop=lambda: "channels" in made and made["channels"].stopping(),
                           stop_reason=lambda: made["channels"].stop_reason() if "channels" in made else None,
                           resource_sampler=resource_sampler)
    finally:
        if "channels" in made:
            made["channels"].close()
    return rounds, made.get("channels")


def main(argv: Sequence[str] | None = None) -> int:
    from yeto.rl.engine.engine_sampler_entry import SamplerSpecError, hold_after_reclaim, load_spec

    parser = argparse.ArgumentParser(prog="python -m yeto.rl.spot_rollout.sampler",
                                     description="run a spot sampler island from a spec file")
    parser.add_argument("--spec", required=True, help="the sampler spec JSON (module doc)")
    args = parser.parse_args(argv)
    try:
        rounds, channels = run_spec(load_spec(args.spec))
    except SamplerSpecError as error:
        print(f"[spot-sampler] spec rejected: {error}", file=sys.stderr)
        return 2
    finished = channels is not None and channels.client.run_finished()
    fence = None if channels is None else channels.stop_reason()
    if fence is not None:
        print(f"[spot-sampler] left {fence[0].name}: fenced ({fence[1]['fenced']})", flush=True)
    print(f"[spot-sampler] done: {rounds} round(s); the run "
          f"{'finished' if finished else 'did not finish'}", flush=True)
    hold_after_reclaim()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

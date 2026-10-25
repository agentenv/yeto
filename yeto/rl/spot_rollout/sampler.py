"""The spot sampler island (S27 G-EA): a rollout island on any cloud.

It follows the global weights as a receive-only syncer subscriber, serves
them with its own engine, and sends its groups back through the head's
sample relay. It never pushes, never counts toward quorum, and holds no store
credentials. When the cloud reclaims it, the training islands keep merging;
a new incarnation subscribes again and gets every fragment at its current
version.

Loop (:class:`SpotSampler`):

1. take the fragments the subscriber received (:class:`~.subscriber.WeightFollower`);
2. when every fragment arrived and the merged version
   (:func:`~yeto.rl.engine.full_parameter_sync.merged_version`) is at least
   ``reload_every`` above the served one: write a HF directory
   (:mod:`.materialize`) and load it into the engine with the post-load checks
   (:class:`~yeto.rl.engine.sampler_checks.PortReleaseLoader`); the engine then
   serves that merged version, and every sample of the following rounds is
   stamped with it (``metadata.weight_version``; the per-token
   ``rollout_log_probs`` come from exactly these weights);
3. when the trainer has fewer than ``max_backlog_groups`` groups waiting in
   the relay channel: run one :class:`~yeto.rl.engine.engine_sampler.EngineRound`
   into a :class:`~.relay_buffer.RelayGroupWriter`; else wait.

Stops when: the reclaim marker is set (PREEMPTED), the syncer says the run
finished, ``max_rounds`` is reached, or the syncer link stays down for
``syncer_lost_s``.

Process entry: ``python -m yeto.rl.spot_rollout.sampler --spec SPEC.json``
(:func:`run_spec`; the spec keys are those of
:mod:`yeto.rl.engine.engine_sampler_entry` for ``engine``, ``sampling``,
``prompts``, ``reward``, ``checks``, ``event_tape``, ``max_prompt_tokens``,
``max_backlog_groups``, plus ``syncer`` (host:port), ``subscriber_id``,
``channel``, ``base_model``, ``work_dir``; optional ``num_streams`` (4),
``reload_every`` (1), ``max_rounds``, ``converter`` (``module:callable``;
default Miles' ``megatron_to_hf``), ``model_name``, ``layout_timeout_s``
(1800), ``syncer_lost_s`` (1800), ``poll_s`` (1.0)). The HMAC key comes from
``YETO_ISLAND_HMAC_KEY``.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from yeto.rl.engine.policy_release import PolicyRelease, ReleaseNotLoaded

from .subscriber import WeightCut, WeightFollower

LOADED_EVENT = "rl_spot_weights_loaded"
WAIT_EVENT = "rl_spot_backlog_wait"
START_EVENT = "rl_spot_sampler_start"
STOP_EVENT = "rl_spot_sampler_stop"


class SpotSampler:
    """The loop of the module doc; every part is injected (tests use fakes)."""

    def __init__(self, *, client: Any, follower: WeightFollower, loader: Any,
                 materialize: Callable[[WeightCut], Path], round: Callable[[int], None],
                 writer: Any, emit: Callable[..., None], max_backlog_groups: int,
                 reload_every: int = 1, poll_s: float = 1.0,
                 should_stop: Callable[[], bool] = lambda: False,
                 syncer_lost_s: float = 1800.0, max_rounds: int | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        if int(reload_every) < 1:
            raise ValueError("reload_every must be >= 1")
        self.client = client
        self.follower = follower
        self.loader = loader
        self.materialize = materialize
        self.round = round
        self.writer = writer
        self.emit = emit
        self.max_backlog_groups = int(max_backlog_groups)
        self.reload_every = int(reload_every)
        self.poll_s = float(poll_s)
        self.should_stop = should_stop
        self.syncer_lost_s = float(syncer_lost_s)
        self.max_rounds = max_rounds
        self.clock = clock
        self.sleep = sleep
        self.served_merged = 0
        self.served_versions: tuple[int, ...] = ()
        self.loads = 0
        self.rounds = 0
        self.stop_reason: str | None = None
        self._lost_since: float | None = None

    def fragment_versions(self) -> Sequence[int]:
        return self.served_versions

    def refresh_weights(self) -> bool:
        """Load the newest complete cut when it is new enough (step 2)."""
        self.follower.drain()
        cut = self.follower.cut()
        if cut is None or cut.merged_version < self.served_merged + self.reload_every:
            return False
        started = self.clock()
        directory = self.materialize(cut)
        made = self.clock()
        tag = hashlib.sha256(repr(cut.versions).encode()).hexdigest()
        release = PolicyRelease(seq=self.loads + 1, version=int(cut.merged_version), policy_hash=tag,
                                path=Path(directory).name)
        try:
            self.loader(release, Path(directory))
        except ReleaseNotLoaded as error:
            self.emit(LOADED_EVENT, policy_version=cut.merged_version, loaded=False,
                      error=str(error)[:300])
            return False
        self.loads += 1
        self.served_merged = int(cut.merged_version)
        self.served_versions = tuple(cut.versions)
        self.emit(LOADED_EVENT, policy_version=self.served_merged, loaded=True,
                  fragment_versions_min=min(cut.versions), fragment_versions_max=max(cut.versions),
                  materialize_s=round(made - started, 3), load_s=round(self.clock() - made, 3),
                  received_fragments=self.follower.received, received_bytes=self.follower.received_bytes)
        return True

    def _stopping(self) -> str | None:
        if self.should_stop():
            return "stopped"
        if callable(getattr(self.client, "run_finished", None)) and self.client.run_finished():
            return "run_finished"
        if self.max_rounds is not None and self.rounds >= int(self.max_rounds):
            return "max_rounds"
        connected = getattr(self.client, "_connected", None)
        if connected is not None and not connected.is_set():
            now = self.clock()
            self._lost_since = now if self._lost_since is None else self._lost_since
            if now - self._lost_since >= self.syncer_lost_s:
                return "syncer_lost"
        else:
            self._lost_since = None
        return None

    def _health(self) -> None:
        try:
            self.client.check_health()
        except Exception:
            if callable(getattr(self.client, "run_finished", None)) and self.client.run_finished():
                return
            raise

    def run(self, start: int = 0) -> int:
        rollout_id = int(start)
        waiting = False
        while True:
            self._health()
            reason = self._stopping()
            if reason is not None:
                self.stop_reason = reason
                break
            self.refresh_weights()
            backlog = self.writer.backlog()
            if backlog >= self.max_backlog_groups:
                if not waiting:
                    self.emit(WAIT_EVENT, backlog_groups=backlog, max_backlog_groups=self.max_backlog_groups)
                waiting = True
                self.sleep(self.poll_s)
                refresh = getattr(getattr(self.writer, "producer", None), "depth", None)
                if callable(refresh):
                    refresh()
                continue
            waiting = False
            self.round(rollout_id)
            rollout_id += 1
            self.rounds += 1
        self.emit(STOP_EVENT, reason=self.stop_reason, rounds=self.rounds, loads=self.loads,
                  policy_version=self.served_merged)
        return self.rounds


def run_spec(spec: Mapping[str, Any], *, environ: Mapping[str, str] | None = None,
             incarnation: str | None = None, stop: Callable[[], bool] | None = None) -> SpotSampler:
    """Build the island from a spec (module doc), run it, stop the engine."""
    import os

    from yeto.protocol import island_hmac_key_from_env
    from yeto.rl.engine.engine_sampler import EngineRound
    from yeto.rl.engine.engine_sampler_entry import (
        apply_rollout_quant,
        build_engine,
        build_prompts,
        build_reward,
        incarnation_tape,
        tape_writer,
    )
    from yeto.rl.engine.sampler_checks import PortReleaseLoader, PostLoadChecks, check_sampling_fits
    from yeto.rl.engine.sampler_engine import Provenance, SamplingParams
    from yeto.rl.engine.sampler_resume import (
        RECLAIM_MARKER_ENV,
        install_marker_signals,
        new_incarnation,
        reclaim_marker,
        reclaim_requested,
    )

    from . import layout_manifest
    from .materialize import converter_from_spec, materialize
    from .relay import RelayBoard, RelayProducer
    from .relay_buffer import RelayGroupWriter, parse_addr
    from .subscriber import SubscriberClient

    env = os.environ if environ is None else environ
    spec = apply_rollout_quant(spec)
    sampler_id = str(spec["sampler_id"])
    incarnation = incarnation or new_incarnation()
    emit = tape_writer(incarnation_tape(spec["event_tape"], incarnation), sampler_id, spec.get("island_id"))
    addr = parse_addr(str(spec["syncer"]))
    key = island_hmac_key_from_env(env)
    reclaim_marker(env).unlink(missing_ok=True)
    if env.get(RECLAIM_MARKER_ENV):
        install_marker_signals(env)
    reclaimed = lambda: reclaim_requested(env)  # noqa: E731

    board = RelayBoard(addr, layout_manifest.LAYOUT_BOARD, int(spec["subscriber_id"]), key=key)
    layout, meta = layout_manifest.fetch(board, timeout_s=float(spec.get("layout_timeout_s", 1800.0)))
    board.close()
    sampling_doc = dict(spec["sampling"])
    sampling_doc["stop"] = tuple(sampling_doc.get("stop") or ())
    sampling = SamplingParams(**sampling_doc).check()
    engine = build_engine(spec)
    check_sampling_fits(sampling, engine, max_prompt_tokens=int(spec["max_prompt_tokens"]))
    provenance = Provenance.of(engine.identity, sampler_id=sampler_id, sampling=sampling)
    emit(START_EVENT, sampler_id=sampler_id, incarnation=incarnation, syncer=f"{addr[0]}:{addr[1]}",
         subscriber_id=int(spec["subscriber_id"]), channel=int(spec["channel"]),
         fragments=layout.num_fragments, layout_meta=meta, sampling=sampling.to_dict(),
         provenance=provenance.to_record())
    loader = PortReleaseLoader(engine, PostLoadChecks(**dict(spec.get("checks") or {})), emit)
    loader.start(None, 0)  # the base weights: version 0 (the trainers' initial cut)
    client = SubscriberClient(addr, int(spec["subscriber_id"]), layout,
                              num_streams=int(spec.get("num_streams", 4)), hmac_key=key)
    client.start()
    follower = WeightFollower(client, layout.num_fragments)
    producer = RelayProducer(addr, int(spec["channel"]), int(spec["subscriber_id"]), key=key)
    stopping = lambda: reclaimed() or (stop is not None and stop())  # noqa: E731
    sampler: SpotSampler
    writer = RelayGroupWriter(producer, sampler_id, fragment_versions=lambda: sampler.fragment_versions(),
                              should_stop=lambda: stopping() or client.run_finished())
    prompts, reward = build_prompts(spec), build_reward(spec)
    engine_round = EngineRound(engine, writer, prompts=prompts, reward=reward, sampling=sampling,
                               provenance=provenance, emit=emit, incarnation=incarnation,
                               reclaimed=stopping)
    convert = converter_from_spec(spec)
    work = Path(spec["work_dir"]).expanduser()
    work.mkdir(parents=True, exist_ok=True)
    sampler = SpotSampler(
        client=client, follower=follower, loader=loader,
        materialize=lambda cut: materialize(layout, cut, base_model=spec["base_model"], out_root=work,
                                            convert=convert, model_name=spec.get("model_name")),
        round=engine_round, writer=writer, emit=emit,
        max_backlog_groups=int(spec["max_backlog_groups"]), reload_every=int(spec.get("reload_every", 1)),
        poll_s=float(spec.get("poll_s", 1.0)), should_stop=stopping,
        syncer_lost_s=float(spec.get("syncer_lost_s", 1800.0)), max_rounds=spec.get("max_rounds"))
    try:
        sampler.run()
    finally:
        try:
            client.close()
            producer.close()
        finally:
            engine.stop()
    return sampler


def main(argv: Sequence[str] | None = None) -> int:
    from yeto.rl.engine.engine_sampler_entry import SamplerSpecError, hold_after_reclaim, load_spec

    parser = argparse.ArgumentParser(prog="python -m yeto.rl.spot_rollout.sampler",
                                     description="run a spot sampler island from a spec file")
    parser.add_argument("--spec", required=True, help="the sampler spec JSON (module doc)")
    args = parser.parse_args(argv)
    try:
        sampler = run_spec(load_spec(args.spec))
    except SamplerSpecError as error:
        print(f"[spot-sampler] spec rejected: {error}", file=sys.stderr)
        return 2
    print(f"[spot-sampler] done: {sampler.rounds} round(s), {sampler.loads} load(s), "
          f"stop {sampler.stop_reason}", flush=True)
    hold_after_reclaim()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

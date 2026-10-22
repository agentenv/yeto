"""Process entry of the engine sampler island (``python -m
yeto.rl.engine.engine_sampler --spec SPEC.json``; S25 stage 1, gap 3).

:func:`~.engine_sampler.build_engine_sampler` wires the island from objects;
this module builds those objects from one JSON spec file that the launch
layer writes (the stage-1 plan writes it with ``yeto-stage1 plan``), then runs
the island to its data end. It adds no sampling logic of its own.

The spec (:class:`SamplerSpec`, every key below is required unless marked):

* ``sampler_id``: the sampler's buffer shard and sample id prefix;
* ``engine``: ``{"factory": "module:callable", ...}``; the factory (adapter
  layer, e.g. ``yeto.rl.adapters.sampler_engines.from_spec:build``) builds the
  engine from this section;
* ``sampling``: :class:`~.sampler_engine.SamplingParams` fields;
* ``num_rollouts``, ``max_backlog_groups``, ``max_prompt_tokens``;
  optional ``base_version`` (0), ``base_dir`` (the engine's base weights),
  ``min_round_s``, ``poll_s``, ``live_poll_s``, ``max_consecutive_load_failures``;
* ``store``: the run's checkpoint store (``store_for``; an ``oss://`` store
  needs the store environment of :mod:`yeto.cloud.oss_store`);
  ``release_dir``: the release directory the sampler follows (optional,
  default ``policy-releases``); ``buffer_root``: the store holding
  ``sample-buffer/`` (optional, default ``store``);
* ``prompts``: ``{"path": JSONL, "input_key", "label_key", "groups_per_round",
  "format": "text" | "chat"}`` -- round ``r`` takes the
  ``groups_per_round`` rows after row ``r * groups_per_round`` (wrapping);
  ``chat`` applies the tokenizer's chat template of ``tokenizer`` (default
  ``base_dir``) to a message list (``trust_remote_code``: optional, false);
* ``reward``: ``"module:function"`` called as ``function(item, completion)``
  -> float, or a Miles reward (``async score(args, sample)``) when
  ``reward_kind`` is ``"miles"``;
* ``checks``: :class:`~.sampler_checks.PostLoadChecks` keyword overrides
  (optional; e.g. ``expected_expert_matrices``);
* ``event_tape``: the JSONL tape file of the island's events; each
  incarnation writes its own file next to it (``events.<incarnation>.jsonl``),
  so a replacement container never overwrites an earlier one's tape.

Replacement (S25 G1C): the process is one incarnation of the sampler
(``sampler_resume.new_incarnation``). It reads where the earlier incarnations
got to from the buffer (``sampler_resume.resume_point``), starts at the next
round, and writes ``rl_sampler_join`` / ``rl_sampler_leave`` plus one member
record per incarnation in the buffer store. With ``YETO_SPOT_RECLAIM_MARKER``
set, SIGINT/SIGTERM write the reclaim marker; the island then starts no new
round, writes no more groups and leaves as PREEMPTED.

Nothing here was run on a PPU or NPU node.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import json
import sys
import time
import types
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .engine_sampler import EngineSamplerConfig, PromptItem, Reward, build_engine_sampler
from .sampler_checks import PostLoadChecks
from .sampler_engine import Completion, SamplerEngine, SamplingParams

PROMPT_FORMATS = ("text", "chat")
REWARD_KINDS = ("neutral", "miles")


class SamplerSpecError(ValueError):
    """The sampler spec file is incomplete or contradicts itself."""


def _require(doc: Mapping[str, Any], key: str, where: str = "spec") -> Any:
    if key not in doc:
        raise SamplerSpecError(f"{where}.{key} is required")
    return doc[key]


def _import(target: str) -> Any:
    module, _, attr = str(target).partition(":")
    if not module or not attr:
        raise SamplerSpecError(f"{target!r} is not 'module:attribute'")
    return getattr(importlib.import_module(module), attr)


# ------------------------------------------------------------------- engine

def build_engine(spec: Mapping[str, Any]) -> SamplerEngine:
    """The engine adapter of ``spec["engine"]``: ``factory`` (``module:callable``,
    an adapter-layer function) builds it from the engine section. The neutral
    core names no adapter (import boundaries, ``yeto/rl/engine/README.md``)."""
    engine = _require(spec, "engine")
    factory = _import(_require(engine, "factory", "engine"))
    try:
        return factory(engine)
    except (KeyError, TypeError, ValueError) as error:
        raise SamplerSpecError(f"engine: {error}") from None


# ------------------------------------------------------------------ prompts

class JsonlPrompts:
    """``prompts(rollout_id)`` over a local JSONL file (module doc)."""

    def __init__(self, path: str | Path, *, input_key: str, label_key: str | None, groups_per_round: int,
                 render: Callable[[Any], str] = str) -> None:
        self.path = Path(path)
        self.rows = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()
                     if line.strip()]
        if not self.rows:
            raise SamplerSpecError(f"prompts file {self.path} has no rows")
        if int(groups_per_round) < 1:
            raise SamplerSpecError("prompts.groups_per_round must be >= 1")
        self.input_key, self.label_key = input_key, label_key
        self.groups_per_round = int(groups_per_round)
        self.render = render

    def cursor(self, rollout_id: int) -> dict[str, int]:
        """The data position after round ``rollout_id`` drew its prompts (the
        ``sampler_round`` record; rounds draw by rollout id, so it only documents)."""
        return {"row_offset": (int(rollout_id) + 1) * self.groups_per_round}

    def __call__(self, rollout_id: int) -> list[PromptItem]:
        start = int(rollout_id) * self.groups_per_round
        items = []
        for offset in range(self.groups_per_round):
            index = (start + offset) % len(self.rows)
            row = self.rows[index]
            if self.input_key not in row:
                raise SamplerSpecError(f"{self.path} row {index} has no {self.input_key!r}")
            items.append(PromptItem(prompt_id=f"row{index}", prompt=self.render(row[self.input_key]),
                                    label=row.get(self.label_key) if self.label_key else None,
                                    metadata={"row": index}))
        return items


def chat_renderer(tokenizer_dir: str | Path, *, remote_code: bool = False) -> Callable[[Any], str]:
    from transformers import AutoTokenizer  # noqa: PLC0415 - only the chat format needs it

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), trust_remote_code=remote_code)

    def render(messages: Any) -> str:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return render


def build_prompts(spec: Mapping[str, Any]) -> JsonlPrompts:
    prompts = _require(spec, "prompts")
    fmt = str(prompts.get("format", "text"))
    if fmt not in PROMPT_FORMATS:
        raise SamplerSpecError(f"prompts.format {fmt!r} is not one of {PROMPT_FORMATS}")
    render: Callable[[Any], str] = str
    if fmt == "chat":
        tokenizer = prompts.get("tokenizer") or spec.get("base_dir") \
            or (spec.get("engine") or {}).get("server", {}).get("base_dir")
        if not tokenizer:
            raise SamplerSpecError("prompts.format chat needs prompts.tokenizer or a base_dir")
        render = chat_renderer(tokenizer, remote_code=bool(prompts.get("trust_remote_code", False)))
    return JsonlPrompts(_require(prompts, "path", "prompts"), input_key=str(_require(prompts, "input_key", "prompts")),
                        label_key=prompts.get("label_key"),
                        groups_per_round=int(_require(prompts, "groups_per_round", "prompts")), render=render)


# ------------------------------------------------------------------- reward

def miles_reward(score: Callable[..., Any]) -> Reward:
    """A Miles reward (``async score(args, sample)``) as an engine-sampler reward:
    the sample carries ``response``, ``label``, a ``metadata`` dict, ``tokens``
    (prompt ids when the engine gave them, then the response ids) and
    ``response_length``, as a Miles ``Sample`` does."""

    def reward(item: PromptItem, completion: Completion) -> float:
        response = list(getattr(completion, "token_ids", ()))
        prompt_ids = list(getattr(completion, "prompt_token_ids", ()))
        sample = types.SimpleNamespace(prompt=item.prompt, response=completion.text, label=item.label,
                                       metadata=dict(item.metadata), tokens=prompt_ids + response,
                                       response_length=len(response))
        value = score(None, sample)
        return float(asyncio.run(value) if inspect.isawaitable(value) else value)

    return reward


def build_reward(spec: Mapping[str, Any]) -> Reward:
    kind = str(spec.get("reward_kind", "neutral"))
    if kind not in REWARD_KINDS:
        raise SamplerSpecError(f"reward_kind {kind!r} is not one of {REWARD_KINDS}")
    function = _import(_require(spec, "reward"))
    return miles_reward(function) if kind == "miles" else function


# -------------------------------------------------------------------- store

def build_stores(spec: Mapping[str, Any], environ: Mapping[str, str] | None = None) -> tuple[Any, Any]:
    """(release store, sample buffer) of the spec."""
    from .policy_release import RELEASE_DIR, PolicyReleaseStore  # noqa: PLC0415
    from .resume import store_for  # noqa: PLC0415
    from .sample_buffer import INDEX, FileSampleBuffer, needs_shards, shard_index  # noqa: PLC0415

    store = str(_require(spec, "store"))
    releases = PolicyReleaseStore(store_for(store, environ=environ),
                                  release_dir=str(spec.get("release_dir") or RELEASE_DIR))
    buffer_store = store_for(str(spec.get("buffer_root") or store), environ=environ)
    sampler_id = str(_require(spec, "sampler_id"))
    index = shard_index(sampler_id) if needs_shards(buffer_store) else INDEX
    return releases, FileSampleBuffer(buffer_store, batch_size=1, index=index)  # as open_buffer(sampler=True)


def incarnation_tape(path: str | Path, incarnation: str) -> Path:
    """``events.jsonl`` -> ``events.<incarnation>.jsonl`` (one tape per incarnation)."""
    tape = Path(path).expanduser()
    return tape.with_name(f"{tape.stem}.{incarnation}{tape.suffix or '.jsonl'}")


def tape_writer(path: str | Path, sampler_id: str) -> Callable[..., None]:
    from yeto.rl.event_echo import append_record  # noqa: PLC0415

    tape = Path(path).expanduser()
    tape.parent.mkdir(parents=True, exist_ok=True)

    def emit(event: str, **fields: Any) -> None:
        append_record(tape, {"sampler_id": sampler_id, "time_unix": time.time(), "event": event, **fields})

    return emit


# ---------------------------------------------------------------------- run

def island_config(spec: Mapping[str, Any]) -> EngineSamplerConfig:
    sampling = dict(_require(spec, "sampling"))
    sampling["stop"] = tuple(sampling.get("stop") or ())
    base_dir = spec.get("base_dir")
    return EngineSamplerConfig(
        sampler_id=str(_require(spec, "sampler_id")), num_rollouts=int(_require(spec, "num_rollouts")),
        max_backlog_groups=int(_require(spec, "max_backlog_groups")),
        sampling=SamplingParams(**sampling), max_prompt_tokens=int(_require(spec, "max_prompt_tokens")),
        base_version=int(spec.get("base_version", 0)), base_dir=None if base_dir is None else Path(base_dir),
        min_round_s=spec.get("min_round_s"), poll_s=float(spec.get("poll_s", 5.0)),
        live_poll_s=None if spec.get("live_poll_s") is None else float(spec["live_poll_s"]),
        max_consecutive_load_failures=int(spec.get("max_consecutive_load_failures", 3)))


def run(spec: Mapping[str, Any], *, environ: Mapping[str, str] | None = None,
        resource_sampler: Callable[[Callable[..., None]], Any] | None = None,
        incarnation: str | None = None, stop: Callable[[], bool] | None = None,
        on_event: Callable[..., None] | None = None) -> int:
    """Build every part of the island from ``spec``, run it from the resume
    point, stop the engine; returns the rounds this incarnation ran.
    ``resource_sampler(emit)`` returns a started telemetry thread (default:
    ``telemetry.ResourceSampler`` of the island's device family; None from the
    factory = no telemetry). ``stop()``: the caller's own end condition (no new
    round starts; the island leaves FINISHED, not PREEMPTED). ``on_event(event,
    **fields)`` sees every tape record after it is written."""
    import os  # noqa: PLC0415

    from yeto.rl.elastic_client import LeaveReason  # noqa: PLC0415

    from .sampler_resume import (  # noqa: PLC0415
        RECLAIM_MARKER_ENV,
        ReclaimWatcher,
        SamplerMembership,
        install_marker_signals,
        new_incarnation,
        reclaim_marker,
        reclaim_requested,
        resume_point,
    )

    env = os.environ if environ is None else environ
    config = island_config(spec)
    incarnation = incarnation or new_incarnation()
    tape = incarnation_tape(_require(spec, "event_tape"), incarnation)
    write = tape_writer(tape, config.sampler_id)

    def emit(event: str, **fields: Any) -> None:
        write(event, **fields)
        if on_event is not None:
            on_event(event, **fields)

    releases, buffer = build_stores(spec, environ)
    prompts, reward = build_prompts(spec), build_reward(spec)
    checks = PostLoadChecks(**dict(spec.get("checks") or {}))
    reclaim_marker(env).unlink(missing_ok=True)  # a restarted machine keeps its /tmp
    if env.get(RECLAIM_MARKER_ENV):
        install_marker_signals(env)
    buffer.refresh()
    resume = resume_point(buffer.entries(), config.sampler_id)
    from .resume import store_for  # noqa: PLC0415

    member_store = store_for(str(spec.get("buffer_root") or spec["store"]), environ=environ)
    membership = SamplerMembership(member_store, config.sampler_id, incarnation, resume, emit)
    reclaimed = lambda: reclaim_requested(env)  # noqa: E731
    island = None

    def leave(reason: LeaveReason, **detail: Any) -> None:
        version = None if island is None else island.follower.version
        membership.leave(reason.name, policy_version=version, buffer_end=len(buffer), **detail)

    engine = build_engine(spec)
    if resource_sampler is None:
        from .telemetry import ResourceSampler  # noqa: PLC0415

        def resource_sampler(sink: Callable[..., None]) -> Any:
            return ResourceSampler(lambda event, **fields: sink(event, **fields)).start()

    telemetry = resource_sampler(emit)
    watcher = ReclaimWatcher(reclaimed, lambda: leave(LeaveReason.PREEMPTED)).start()
    try:
        island = build_engine_sampler(config, engine=engine, releases=releases, buffer=buffer, emit=emit,
                                      prompts=prompts, reward=reward, checks=checks,
                                      should_stop=lambda: reclaimed() or (stop is not None and stop()),
                                      reclaimed=reclaimed, incarnation=incarnation, resume=resume,
                                      data_cursor=getattr(prompts, "cursor", None))
        latest = releases.latest()
        membership.join(tape=str(tape), latest_release=None if latest is None else
                        {"seq": latest.seq, "policy_version": latest.version})
        rounds = 0 if buffer.data_ended else island.run()
        leave(LeaveReason.PREEMPTED if reclaimed() else LeaveReason.FINISHED, rounds=rounds,
              dropped_groups=island.round.dropped_groups)
        return rounds
    except BaseException as error:
        try:
            leave(LeaveReason.PREEMPTED if reclaimed() else LeaveReason.INFRA_FAILED,
                  error=f"{type(error).__name__}: {error}"[:300])
        except Exception:  # noqa: BLE001 - the original error is the one to report
            pass
        raise
    finally:
        watcher.stop()
        try:
            engine.stop()
        finally:
            if telemetry is not None:
                telemetry.stop()


def load_spec(path: str | Path) -> dict[str, Any]:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SamplerSpecError(f"cannot read the sampler spec {path}: {error}") from None
    if not isinstance(doc, dict):
        raise SamplerSpecError(f"{path}: the sampler spec must be a JSON object")
    return doc


RECLAIM_HOLD_S = 300.0  # after a reclaim LEAVE: wait this long for the cloud to stop the container


def hold_after_reclaim(environ: Mapping[str, str] | None = None, *, hold_s: float = RECLAIM_HOLD_S,
                       sleep: Callable[[float], None] = time.sleep) -> bool:
    """After a reclaim the process stays up until the cloud stops the container:
    a clean early exit could end the cloud's input as finished instead of letting
    the cloud run the island again (as the Miles sampler island does). Returns
    whether a reclaim was requested (False: returns at once)."""
    import os  # noqa: PLC0415

    from .sampler_resume import reclaim_requested  # noqa: PLC0415

    if not reclaim_requested(os.environ if environ is None else environ):
        return False
    print(f"[engine-sampler] reclaimed: waiting up to {hold_s:g} s for the container to stop", flush=True)
    sleep(hold_s)
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m yeto.rl.engine.engine_sampler",
                                     description="run an engine sampler island from a spec file")
    parser.add_argument("--spec", required=True, help="the sampler spec JSON (module doc)")
    args = parser.parse_args(argv)
    try:
        spec = load_spec(args.spec)
        rounds = run(spec)
    except SamplerSpecError as error:
        print(f"[engine-sampler] spec rejected: {error}", file=sys.stderr)
        return 2
    print(f"[engine-sampler] done: {rounds} round(s)", flush=True)
    hold_after_reclaim()
    return 0

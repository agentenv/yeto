"""Process entry of the engine sampler island (``python -m
yeto.rl.engine.engine_sampler --spec SPEC.json``).

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
* ``rollout_fields`` (optional): ``{"routed_experts": bool, "top_logprobs_num": int,
  "fp32_lm_head": bool}``, the per-token fields every sample must carry
  (R3 routed experts; the top-K log-probs of score_centering, K default 128;
  the fp32 lm-head the trainer's ``engine.version_fields`` check expects). The
  section is handed to the engine factory inside the engine section and checked
  against the built engine's capabilities (:func:`check_rollout_fields`);
* ``platform`` (optional): ``cuda`` (default) / ``rocm`` / ``ppu`` / ``npu``, the
  accelerator the engine runs on; handed to the engine factory the same way
  (``rocm`` turns on the SGLang ROCm environment preset and the AMD card check);
* ``num_rollouts``, ``max_backlog_groups``, ``max_prompt_tokens``;
  optional ``base_version`` (0), ``base_dir`` (the engine's base weights),
  ``start_on_latest`` (false: start on ``base_dir``; true: on the store's newest
  release, ``role_switch.sampler_spec_on_release``, so a restarted sampler does
  not generate its first round on the base weights),
  ``min_round_s``, ``poll_s``, ``live_poll_s``, ``max_consecutive_load_failures``;
  ``reload_max_defer_s`` (default 600; null: no bound): the longest a release
  waits on running agentic trials before it is loaded anyway;
* ``store``: the run's checkpoint store (``store_for``; an ``oss://`` store
  needs the store environment of :mod:`yeto.cloud.oss_store`);
  ``release_dir``: the release directory the sampler follows (optional,
  default ``policy-releases``); ``buffer_root``: the store holding
  ``sample-buffer/`` (optional, default ``store``); ``buffer_shards``: the
  sampler shards of a run with several samplers on one shared directory
  (optional; the sampler writes its own shard, the trainer merges them, as
  ``--rl-buffer-shards`` on the Miles sampler);
* ``prompts``: ``{"path": JSONL, "input_key", "label_key", "groups_per_round",
  "format": "text" | "chat"}`` -- round ``r`` takes the
  ``groups_per_round`` rows after row ``r * groups_per_round`` (wrapping);
  optional ``"data_slot": "I/N"``: the sampler draws only rows ``i % N == I``
  (``sample_roles.DataSlot``; several samplers of one run draw disjoint prompts);
  ``chat`` applies the tokenizer's chat template of ``tokenizer`` (default
  ``base_dir``) to a message list (``trust_remote_code``: optional, false);
* ``reward``: ``"module:function"`` called as ``function(item, completion)``
  -> float, or a Miles reward (``async score(args, sample)``) when
  ``reward_kind`` is ``"miles"``;
* ``checks``: :class:`~.sampler_checks.PostLoadChecks` keyword overrides
  (optional; e.g. ``expected_expert_matrices``);
* ``expert_selection_sha256`` (optional): the partial-experts selection the
  trainer runs with (Kimi-K3.1, INTERFACES 2.3); a release whose record's
  ``manifest.expert_selection_sha256`` differs is refused and the served
  weights are kept;
* ``expert_trainable_sha256`` (optional): the selection's ``trainable_sha256``; when
  set the gate compares ``manifest.partial_experts.trainable_sha256`` instead, so a
  trainer running the same trainable experts under another EP layout (another slot
  order, hence another ``expert_selection_sha256``) is accepted;
* ``rollout_quant``: optional ``{"tier", "apply"}``, applied first
  (:func:`apply_rollout_quant`);
* ``incarnation_lease``: optional (default on, in the buffer store): ``false``, or
  ``{"store", "poll_s", "write_check_s"}`` (:mod:`.incarnation_lease`):
  the incarnation claims the island's lease at join and leaves FENCED, writing no
  more groups, once another incarnation holds it;
* ``event_tape``: the JSONL tape file of the island's events; each
  incarnation writes its own file next to it (``events.<incarnation>.jsonl``),
  so a replacement container never overwrites an earlier one's tape.

Replacement: the process is one incarnation of the sampler
(``sampler_resume.new_incarnation``). It reads where the earlier incarnations
got to from the buffer (``sampler_resume.resume_point``), starts at the next
round, and writes ``rl_sampler_join`` / ``rl_sampler_leave`` plus one member
record per incarnation in the buffer store. With ``YETO_SPOT_RECLAIM_MARKER``
set, SIGINT/SIGTERM write the reclaim marker; the island then starts no new
round, writes no more groups and leaves as PREEMPTED. The process exits with
``RECLAIM_EXIT_CODE`` within ``RECLAIM_DEADLINE_S`` of the marker, also when a
trial or a boat sandbox call still blocks (``ReclaimDeadline``).

Nothing here was run on a PPU or NPU node.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib
import inspect
import json
import os
import sys
import threading
import time
import types
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any

from .engine_sampler import (DEFAULT_RELOAD_MAX_DEFER_S, EngineSamplerConfig, PromptItem, Reward,
                             build_engine_sampler)
from .sampler_checks import PostLoadChecks
from .data_split import DataSlot
from .sampler_engine import (
    Completion,
    SamplerEngine,
    SamplingParams,
    returns_routed_experts,
    top_logprobs_num,
    version_fields,
)

PROMPT_FORMATS = ("text", "chat", "agentic")
ROLLOUT_FIELDS = "rollout_fields"
ROLLOUT_FIELD_KEYS = ("routed_experts", "top_logprobs_num", "fp32_lm_head")
PLATFORM = "platform"
PLATFORMS = ("cuda", "rocm", "ppu", "npu")
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
    engine = dict(_require(spec, "engine"))
    fields = spec.get(ROLLOUT_FIELDS)
    if fields is not None:
        if not isinstance(fields, Mapping) or set(fields) - set(ROLLOUT_FIELD_KEYS):
            raise SamplerSpecError(f"{ROLLOUT_FIELDS} must be a mapping with keys from {ROLLOUT_FIELD_KEYS}")
        if ROLLOUT_FIELDS in engine and dict(engine[ROLLOUT_FIELDS]) != dict(fields):
            raise SamplerSpecError(f"{ROLLOUT_FIELDS} given twice (spec and engine section) and they differ")
        engine[ROLLOUT_FIELDS] = dict(fields)
    platform = spec.get(PLATFORM)
    if platform is not None:
        if platform not in PLATFORMS:
            raise SamplerSpecError(f"{PLATFORM} must be one of {PLATFORMS}, got {platform!r}")
        if PLATFORM in engine and engine[PLATFORM] != platform:
            raise SamplerSpecError(f"{PLATFORM} given twice (spec and engine section) and they differ")
        engine[PLATFORM] = platform
    factory = _import(_require(engine, "factory", "engine"))
    try:
        built = factory(engine)
    except (KeyError, TypeError, ValueError) as error:
        raise SamplerSpecError(f"engine: {error}") from None
    check_rollout_fields(spec, built)
    return built


def check_rollout_fields(spec: Mapping[str, Any], engine: Any) -> dict[str, Any]:
    """The built engine must return every field ``spec["rollout_fields"]`` asks
    for (the trainer refuses a buffer without them much later, after GPU time
    was spent). Returns what the engine will write: the R3 flag, K and the
    ``engine.version_fields`` of the spec's sampling."""
    fields = spec.get(ROLLOUT_FIELDS) or {}
    sampling = dict(spec.get("sampling") or {"max_tokens": 1})
    sampling["stop"] = tuple(sampling.get("stop") or ())
    have = {"routed_experts": returns_routed_experts(engine), "top_logprobs_num": top_logprobs_num(engine),
            "version_fields": version_fields(engine, SamplingParams(**sampling))}
    if fields.get("routed_experts") and not have["routed_experts"]:
        raise SamplerSpecError(f"{ROLLOUT_FIELDS}.routed_experts: the engine {type(engine).__name__} returns "
                               "no routed experts (R3)")
    want_k = fields.get("top_logprobs_num", 128 if fields else 0) or 0
    if int(want_k) > 0 and have["top_logprobs_num"] < int(want_k):
        raise SamplerSpecError(f"{ROLLOUT_FIELDS}.top_logprobs_num = {want_k}: the engine returns "
                               f"{have['top_logprobs_num']}")
    if fields.get("fp32_lm_head") and not have["version_fields"]["fp32_lm_head"]:
        raise SamplerSpecError(f"{ROLLOUT_FIELDS}.fp32_lm_head: the engine serves no fp32 lm-head")
    return have


def apply_rollout_quant(spec: Mapping[str, Any]) -> Mapping[str, Any]:
    """``spec["rollout_quant"] = {"tier", "apply": "module:function"}`` (written by
    the launch layer) fills the tier's engine settings on this node; the adapter-layer
    function returns the new spec. No section: the spec is used as written."""
    section = spec.get("rollout_quant")
    if not section:
        return spec
    try:
        return _import(_require(section, "apply", "rollout_quant"))(spec)
    except ValueError as error:
        raise SamplerSpecError(f"rollout_quant: {error}") from None


# ------------------------------------------------------------------ prompts

class JsonlPrompts:
    """``prompts(rollout_id)`` over a local JSONL file (module doc)."""

    def __init__(self, path: str | Path, *, input_key: str, label_key: str | None, groups_per_round: int,
                 render: Callable[[Any], str] = str, data_slot: str | None = None,
                 task_key: str | None = None) -> None:
        """``task_key``: the row field holding the agent task mapping (format
        ``agentic``); it goes into the item metadata as ``agent_task``."""
        self.path = Path(path)
        rows = list(enumerate(json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()
                              if line.strip()))
        if not rows:
            raise SamplerSpecError(f"prompts file {self.path} has no rows")
        try:
            self.slot = DataSlot.parse(data_slot)
            rows = self.slot.rows(rows)
        except ValueError as error:
            raise SamplerSpecError(f"prompts.data_slot: {error}") from None
        # the prompt ids keep the file's row numbers, so two slots never share one
        self.row_numbers = [i for i, _ in rows]
        self.rows = [row for _, row in rows]
        if int(groups_per_round) < 1:
            raise SamplerSpecError("prompts.groups_per_round must be >= 1")
        if int(groups_per_round) > len(self.rows):
            # rounds wrap over the file, but one round drawing a row twice would write two
            # groups under one key (the payload name carries the row): refuse before any GPU time
            raise SamplerSpecError(f"prompts.groups_per_round {int(groups_per_round)} exceeds the {len(self.rows)} "
                                   f"rows of {self.path}{'' if data_slot is None else f' in slot {data_slot}'}: "
                                   "a round would draw a prompt twice")
        self.input_key, self.label_key = input_key, label_key
        self.groups_per_round = int(groups_per_round)
        self.render = render
        self.task_key = task_key

    def cursor(self, rollout_id: int) -> dict[str, int]:
        """The data position after round ``rollout_id`` drew its prompts (the
        ``sampler_round`` record; rounds draw by rollout id, so it only documents)."""
        return {"row_offset": (int(rollout_id) + 1) * self.groups_per_round}

    def __call__(self, rollout_id: int) -> list[PromptItem]:
        start = int(rollout_id) * self.groups_per_round
        items = []
        for offset in range(self.groups_per_round):
            index = self.row_numbers[(start + offset) % len(self.rows)]
            row = self.rows[(start + offset) % len(self.rows)]
            if self.input_key not in row:
                raise SamplerSpecError(f"{self.path} row {index} has no {self.input_key!r}")
            metadata: dict[str, Any] = {"row": index}
            if self.task_key is not None:
                task = row.get(self.task_key)
                if not isinstance(task, Mapping):
                    raise SamplerSpecError(f"{self.path} row {index}: {self.task_key!r} is not an agent task mapping")
                metadata["agent_task"] = dict(task)
            items.append(PromptItem(prompt_id=f"row{index}", prompt=self.render(row[self.input_key]),
                                    label=row.get(self.label_key) if self.label_key else None,
                                    metadata=metadata))
        return items


def dispensed_prompts(path: str | Path, config: Mapping[str, Any], *, input_key: str,
                      label_key: str | None, groups_per_round: int,
                      render: Callable[[Any], str] = str, key: bytes | None = None,
                      environ: Any = None, task_key: str | None = None) -> "Any":
    """The engine sampler's prompts from the task dispenser: the
    sibling of :class:`JsonlPrompts` that a sampler island of any backend uses
    (``task_dispense.DispensedPrompts``). ``config``: ``prompts.dispense`` of the
    spec (address, island id, seed, prefetch blocks; the mix / label streams)."""
    from .task_dispense import DispensedPrompts

    return DispensedPrompts.open(path, config, input_key=input_key, label_key=label_key,
                                 groups_per_round=groups_per_round, render=render, key=key,
                                 environ=environ, task_key=task_key)


def store_dispensed_prompts(spec: Mapping[str, Any], *, render: Callable[[Any], str] = str,
                            task_key: str | None = None, environ: Any = None) -> "Any":
    """The engine sampler's prompts from the store ledger (``prompts.dispense``
    with ``backend: "store"``; ``store_dispense.StoreDispensedPrompts``): the
    ledger lives under ``namespace`` of the run's store (``dispense.store``, else
    ``buffer_root`` / ``store``), so samplers that share only the bucket share
    the tasks. The holder is the sampler id, the incarnation this process's;
    the incarnation leases of the other samplers (same store) free the claims of
    a fenced one at once."""
    from .incarnation_lease import read_lease  # noqa: PLC0415
    from .resume import store_for  # noqa: PLC0415
    from .store_dispense import StoreDispensedPrompts  # noqa: PLC0415
    from .task_dispense import SAMPLER_INCARNATION_ENV, DispenseError  # noqa: PLC0415

    prompts = spec["prompts"]
    config = dict(prompts["dispense"])
    env = os.environ if environ is None else environ
    incarnation = env.get(SAMPLER_INCARNATION_ENV)
    if not incarnation:
        raise SamplerSpecError("prompts.dispense backend store needs the sampler incarnation "
                               f"({SAMPLER_INCARNATION_ENV}; set by engine_sampler_entry.run)")
    root = str(config.get("store") or spec.get("buffer_root") or _require(spec, "store"))
    store = store_for(root, environ=environ)
    lease_store = store
    section = spec.get("incarnation_lease", True)
    if isinstance(section, Mapping) and section.get("store"):
        lease_store = store_for(str(section["store"]), environ=environ)
    lease_off = section is False or section is None

    def lease_of_holder(sampler: str) -> str | None:
        record = read_lease(lease_store, sampler)
        return None if not record or record.get("incarnation") is None else str(record["incarnation"])

    try:
        gate = None
        if isinstance(spec.get("agentic"), Mapping) and spec["agentic"].get("envs"):
            # sandbox readiness manifest + free capacity (yeto.sandbox.gate); a runtime filter, not in the plan
            from yeto.sandbox.gate import gate_from_spec  # noqa: PLC0415
            from yeto.sandbox.envs import EnvsError  # noqa: PLC0415

            from .store_dispense import ledger_objects  # noqa: PLC0415

            try:
                gate = gate_from_spec(spec, ledger_objects(store), environ=env)
            except EnvsError as error:
                raise SamplerSpecError(str(error)) from error
        return StoreDispensedPrompts.open(
            _require(prompts, "path", "prompts"), config, store=store, holder=str(_require(spec, "sampler_id")),
            incarnation=str(incarnation), input_key=str(_require(prompts, "input_key", "prompts")),
            label_key=prompts.get("label_key"),
            groups_per_round=int(_require(prompts, "groups_per_round", "prompts")), render=render,
            task_key=task_key, lease_of_holder=None if lease_off else lease_of_holder, gate=gate)
    except DispenseError as error:
        raise SamplerSpecError(str(error)) from error


def chat_renderer(tokenizer_dir: str | Path, *, remote_code: bool = False,
                  template_kwargs: Mapping[str, Any] | None = None, tokenizer: Any = None) -> Callable[[Any], str]:
    """``template_kwargs``: extra chat template variables (``prompts.chat_template_kwargs``,
    e.g. Kimi-K3.1's ``{"thinking_effort": "low"}``)."""
    if tokenizer is None:
        from transformers import AutoTokenizer  # noqa: PLC0415 - only the chat format needs it

        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), trust_remote_code=remote_code)
    extra = dict(template_kwargs or {})

    def render(messages: Any) -> str:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **extra)

    return render


def build_prompts(spec: Mapping[str, Any], environ: Any = None) -> JsonlPrompts:
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
        kwargs = prompts.get("chat_template_kwargs") or {}
        if not isinstance(kwargs, Mapping):
            raise SamplerSpecError("prompts.chat_template_kwargs must be an object")
        render = chat_renderer(tokenizer, remote_code=bool(prompts.get("trust_remote_code", False)),
                               template_kwargs=kwargs)
    if (fmt == "agentic") != bool(spec.get("agentic")):
        raise SamplerSpecError("prompts.format agentic and the spec's agentic section go together")
    task_key = str(prompts.get("task_key", "metadata")) if fmt == "agentic" else None
    if prompts.get("dispense"):  # Rows from the task dispenser, not a slot
        if prompts.get("data_slot"):
            raise SamplerSpecError("prompts.dispense and prompts.data_slot exclude each other")
        if str((prompts["dispense"] or {}).get("backend") or "tcp") == "store":
            return store_dispensed_prompts(spec, render=render, task_key=task_key, environ=environ)
        from yeto.protocol import island_hmac_key_from_env  # noqa: PLC0415

        return dispensed_prompts(_require(prompts, "path", "prompts"), prompts["dispense"],
                                 input_key=str(_require(prompts, "input_key", "prompts")),
                                 label_key=prompts.get("label_key"),
                                 groups_per_round=int(_require(prompts, "groups_per_round", "prompts")),
                                 render=render, key=island_hmac_key_from_env(), environ=environ,
                                 task_key=task_key)
    return JsonlPrompts(_require(prompts, "path", "prompts"), input_key=str(_require(prompts, "input_key", "prompts")),
                        label_key=prompts.get("label_key"),
                        groups_per_round=int(_require(prompts, "groups_per_round", "prompts")), render=render,
                        data_slot=prompts.get("data_slot"), task_key=task_key)


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
    if spec.get("agentic") and not spec.get("reward"):  # The Harbor verifier scores
        return lambda item, completion: 0.0
    kind = str(spec.get("reward_kind", "neutral"))
    if kind not in REWARD_KINDS:
        raise SamplerSpecError(f"reward_kind {kind!r} is not one of {REWARD_KINDS}")
    function = _import(_require(spec, "reward"))
    return miles_reward(function) if kind == "miles" else function


# -------------------------------------------------------------------- store

def build_stores(spec: Mapping[str, Any], environ: Mapping[str, str] | None = None) -> tuple[Any, Any]:
    """(release store, sample buffer) of the spec."""
    from .policy_release import RELEASE_DIR, PolicyReleaseStore  # noqa: PLC0415
    from .release_delta import default_mirror  # noqa: PLC0415
    from .resume import store_for  # noqa: PLC0415
    from .sample_buffer import INDEX, FileSampleBuffer, needs_shards, shard_index  # noqa: PLC0415

    store = str(_require(spec, "store"))
    releases = PolicyReleaseStore(store_for(store, environ=environ),
                                  release_dir=str(spec.get("release_dir") or RELEASE_DIR),
                                  mirror=default_mirror(environ, store=store))  # delta releases
    buffer_store = store_for(str(spec.get("buffer_root") or store), environ=environ)
    sampler_id = str(_require(spec, "sampler_id"))
    shards = [str(s) for s in spec.get("buffer_shards") or ()]
    if shards and sampler_id not in shards:
        raise SamplerSpecError(f"sampler {sampler_id!r} is not one of the buffer shards {shards}")
    index = shard_index(sampler_id) if needs_shards(buffer_store) or shards else INDEX
    return releases, FileSampleBuffer(buffer_store, batch_size=1, index=index)  # as open_buffer(sampler=True)


def incarnation_tape(path: str | Path, incarnation: str) -> Path:
    """``events.jsonl`` -> ``events.<incarnation>.jsonl`` (one tape per incarnation)."""
    tape = Path(path).expanduser()
    return tape.with_name(f"{tape.stem}.{incarnation}{tape.suffix or '.jsonl'}")


def tape_writer(path: str | Path, sampler_id: str, island_id: int | None = None) -> Callable[..., None]:
    """``island_id``: the launcher island this sampler runs as; its records then
    carry it, so the launcher accepts the file as the island's event tape
    (``yeto.rl.event_echo.REQUIRED_KEYS``)."""
    from yeto.rl.event_echo import append_record  # noqa: PLC0415

    tape = Path(path).expanduser()
    tape.parent.mkdir(parents=True, exist_ok=True)
    head = {"sampler_id": sampler_id} if island_id is None else {"island_id": int(island_id), "sampler_id": sampler_id}

    def emit(event: str, **fields: Any) -> None:
        append_record(tape, {**head, "time_unix": time.time(), "event": event, **fields})

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
        reload_max_defer_s=(None if "reload_max_defer_s" in spec and spec["reload_max_defer_s"] is None
                            else float(spec.get("reload_max_defer_s", DEFAULT_RELOAD_MAX_DEFER_S))),
        max_consecutive_load_failures=int(spec.get("max_consecutive_load_failures", 3)),
        expert_selection_sha256=(None if spec.get("expert_selection_sha256") is None
                                 else str(spec["expert_selection_sha256"])),
        expert_trainable_sha256=(None if spec.get("expert_trainable_sha256") is None
                                 else str(spec["expert_trainable_sha256"])),
        drop_zero_variance_groups=bool(spec.get("drop_zero_variance_groups", False)))


def run(spec: Mapping[str, Any], *, environ: Mapping[str, str] | None = None,
        resource_sampler: Callable[[Callable[..., None]], Any] | None = None,
        incarnation: str | None = None, stop: Callable[[], bool] | None = None,
        on_event: Callable[..., None] | None = None,
        stores: Callable[[str, Callable[..., None]], tuple[Any, Any]] | None = None,
        stop_reason: Callable[[], tuple[Any, Mapping[str, Any]] | None] | None = None,
        reclaim_deadline: "ReclaimDeadline | None" = None) -> int:
    """Build every part of the island from ``spec``, run it from the resume
    point, stop the engine; returns the rounds this incarnation ran.
    ``resource_sampler(emit)`` returns a started telemetry thread (default:
    ``telemetry.ResourceSampler`` of the island's device family; None from the
    factory = no telemetry). ``stop()``: the caller's own end condition (no new
    round starts; the island leaves FINISHED, not PREEMPTED). ``on_event(event,
    **fields)`` sees every tape record after it is written. ``stores(incarnation,
    emit)``: the (release store, sample buffer) pair instead of :func:`build_stores`
    (a spot sampler follows the syncer and writes to the relay).
    ``reclaim_deadline``: armed when the reclaim marker appears; it ends the
    process at its deadline even when a trial or a cleanup step still blocks.
    ``stop_reason()``: after the loop ends, (LeaveReason, detail) to leave with
    instead of FINISHED, or None (a fenced spot sampler leaves FENCED)."""
    import os  # noqa: PLC0415

    from .sampler_resume import new_incarnation  # noqa: PLC0415

    env = os.environ if environ is None else environ
    spec = apply_rollout_quant(spec)
    config = island_config(spec)
    incarnation = incarnation or new_incarnation()
    if env.get("YETO_BOAT_RUN_ID") and env.get("YETO_BOAT_INCARNATION") != incarnation:
        # An engine sampler (the release-store path, e.g. the 120B spot island) labels its
        # boat sandboxes with its incarnation too and sweeps its earlier incarnations' ones; before
        # this only the spot_rollout sampler did, so these sandboxes carried an empty incarnation
        # ("ytl1.<run>.<island>..") that no sweep selects
        from yeto.rl.spot_rollout.sampler import label_boat_sandboxes  # noqa: PLC0415

        label_boat_sandboxes(spec, incarnation, environ=env)
    with _member_incarnation_env(env, incarnation, shard=spec.get("sampler_id")):
        return _run(spec, env, config, incarnation, environ=environ, resource_sampler=resource_sampler,
                    stop=stop, on_event=on_event, stores=stores, stop_reason=stop_reason,
                    reclaim_deadline=reclaim_deadline)


@contextlib.contextmanager
def _member_incarnation_env(env: Any, incarnation: str, *, shard: Any = None):
    """The member incarnation of every channel this process opens to the
    head (task dispenser, spot subscriber, relay) derives from the process
    incarnation (``task_dispense.member_incarnation``), as on a Miles sampler
    island; the variable is restored afterwards (in-process callers, tests)."""
    from .task_dispense import SAMPLER_INCARNATION_ENV  # noqa: PLC0415

    if not isinstance(env, MutableMapping):
        yield
        return
    from yeto.sandbox.trial_labels import SHARD_ENV  # noqa: PLC0415

    # the sampler id labels this process's Harbor trial containers (ai.yeto.shard, with ai.yeto.inc)
    values = {SAMPLER_INCARNATION_ENV: incarnation}
    if shard:
        values[SHARD_ENV] = str(shard)
    previous = {k: env.get(k) for k in values}
    env.update(values)
    try:
        yield
    finally:
        for k, v in previous.items():
            if v is None:
                env.pop(k, None)
            else:
                env[k] = v


POOL_FENCE_PREFIX = "pool-fence:"  # yeto.launch.nc.elastic.FENCE_PREFIX: the lease a pool fence writes


def predecessor_fenced(member_store: Any, predecessor: str) -> bool:
    """Whether a pool fence holds ``predecessor``'s incarnation lease (its node is
    taken for gone: no incarnation of it writes again)."""
    from .incarnation_lease import read_lease  # noqa: PLC0415

    try:
        lease = read_lease(member_store, predecessor) or {}
    except Exception:  # noqa: BLE001 - unknown: not fenced (nothing is swept)
        return False
    return str(lease.get("incarnation") or "").startswith(POOL_FENCE_PREFIX)


def sweep_predecessor(prompts: Any, member_store: Any, predecessor: str, left: bool,
                      emit: Callable[..., None]) -> int:
    """A replacement (``continues_shard``) runs in a new shard id, so the stale sweep
    of its own shard never reaches the trial containers the predecessor left on the
    sandbox host (a node evicted or killed runs no teardown). Once the predecessor
    left or a pool fence holds its lease, remove them (``rl_sandbox_orphans_removed``
    with ``shard``). A predecessor still running unfenced is left alone."""
    sweep = getattr(getattr(prompts, "gate", None), "sweep_shard", None)
    if not callable(sweep):
        return 0
    if not (left or predecessor_fenced(member_store, predecessor)):
        return 0
    removed = int(sweep(predecessor) or 0)
    if removed:
        emit("rl_sandbox_orphans_removed", removed=removed, shard=predecessor,
             why="predecessor_left" if left else "predecessor_fenced")
    return removed


def _run(spec: Mapping[str, Any], env: Any, config: Any, incarnation: str, *,
         environ: Mapping[str, str] | None, resource_sampler: Callable[[Callable[..., None]], Any] | None,
         stop: Callable[[], bool] | None, on_event: Callable[..., None] | None,
         stores: Callable[[str, Callable[..., None]], tuple[Any, Any]] | None,
         stop_reason: Callable[[], tuple[Any, Mapping[str, Any]] | None] | None = None,
         reclaim_deadline: "ReclaimDeadline | None" = None) -> int:
    from yeto.rl.pool_client import LeaveReason  # noqa: PLC0415

    from .incarnation_lease import FENCED_EVENT, LEASE_EVENT  # noqa: PLC0415

    from .sampler_resume import (  # noqa: PLC0415
        RECLAIM_MARKER_ENV,
        ReclaimWatcher,
        SamplerMembership,
        install_marker_signals,
        reclaim_marker,
        reclaim_requested,
        resume_point,
    )

    tape = incarnation_tape(_require(spec, "event_tape"), incarnation)
    write = tape_writer(tape, config.sampler_id, spec.get("island_id"))

    def emit(event: str, **fields: Any) -> None:
        write(event, **fields)
        if on_event is not None:
            on_event(event, **fields)

    releases, buffer = build_stores(spec, environ) if stores is None else stores(incarnation, emit)
    if spec.get("start_on_latest"):
        # a restarted sampler starts its engine on the newest release (no first round
        # on the base weights; the follower then never reloads it)
        from .role_switch import sampler_spec_on_release  # noqa: PLC0415

        spec = sampler_spec_on_release(spec, releases)
        config = island_config(spec)
        emit("rl_sampler_start_on_latest", base_version=config.base_version,
             base_dir=None if config.base_dir is None else str(config.base_dir))
    prompts, reward = build_prompts(spec, env), build_reward(spec)
    checks = PostLoadChecks(**dict(spec.get("checks") or {}))
    reclaim_marker(env).unlink(missing_ok=True)  # a restarted machine keeps its /tmp
    if env.get(RECLAIM_MARKER_ENV):
        install_marker_signals(env)
    buffer.refresh()
    resume = resume_point(buffer.entries(), config.sampler_id)
    if callable(getattr(prompts, "recover", None)):  # store dispenser: settle what was written, free the rest
        if getattr(prompts, "emit", False) is None:
            prompts.emit = emit
        prompts.recover(buffer.entries())
    from .resume import store_for  # noqa: PLC0415

    member_store = store_for(str(spec.get("buffer_root") or spec["store"]), environ=environ)
    membership = SamplerMembership(member_store, config.sampler_id, incarnation, resume, emit)
    reclaimed = lambda: reclaim_requested(env)  # noqa: E731
    lease, lease_poll_s = incarnation_lease(spec, config.sampler_id, incarnation, member_store, environ)
    fenced = lambda: lease is not None and lease.fenced  # noqa: E731
    island = None
    from .sampler_resume import handoff_requested  # noqa: PLC0415

    handed_off = lambda: handoff_requested(member_store, config.sampler_id)  # noqa: E731
    predecessor = spec.get("continues_shard")
    if predecessor is not None and str(predecessor) == config.sampler_id:
        raise SamplerSpecError("continues_shard names this sampler's own shard")

    def leave(reason: LeaveReason, **detail: Any) -> None:
        version = None if island is None else island.follower.version
        membership.leave(reason.name, policy_version=version, buffer_end=len(buffer), **detail)

    engine = build_engine(spec)
    if resource_sampler is None:
        from .telemetry import ResourceSampler  # noqa: PLC0415

        def resource_sampler(sink: Callable[..., None]) -> Any:
            return ResourceSampler(lambda event, **fields: sink(event, **fields)).start()

    telemetry = resource_sampler(emit)
    def on_reclaim() -> None:
        if reclaim_deadline is not None:  # first: a blocking LEAVE must not delay the deadline
            reclaim_deadline.arm(reclaim_seen_at(env))
        leave(LeaveReason.PREEMPTED)

    if reclaim_deadline is not None:
        reclaim_deadline.add_cleanup("engine", engine.stop)
    watcher = ReclaimWatcher(reclaimed, on_reclaim).start()
    fence_watcher = None
    # the dashboard reads every island's tape from the store (tapes/<sampler>/<incarnation>/)
    from .tape_upload import TapeUploader, store_objects, upload_interval_s  # noqa: PLC0415

    upload_s = upload_interval_s(spec.get("tape_upload_s"), env)
    uploader = None
    if upload_s is not None:
        try:
            uploader = TapeUploader(store_objects(member_store), tape, island=config.sampler_id,
                                    incarnation=incarnation, interval_s=upload_s).start()
        except Exception as error:  # noqa: BLE001 - observation only
            print(f"[engine-sampler] tape upload off: {type(error).__name__}: {error}", flush=True)
    try:
        make_round = None
        if spec.get("agentic"):  # Harbor trials through the records proxy
            from .agentic_round import round_factory  # noqa: PLC0415

            make_round = round_factory(spec, engine, env)
            if reclaim_deadline is not None:
                from yeto.rl.harness.harbor.boat_pool import close_pools_now  # noqa: PLC0415

                reclaim_deadline.add_cleanup("boat_sandboxes", close_pools_now)
        island = build_engine_sampler(config, engine=engine, releases=releases, buffer=buffer, emit=emit,
                                      prompts=prompts, reward=reward, checks=checks,
                                      should_stop=lambda: (reclaimed() or fenced() or handed_off()
                                                           or (stop is not None and stop())),
                                      reclaimed=lambda: reclaimed() or fenced(), incarnation=incarnation,
                                      resume=resume,
                                      data_cursor=getattr(prompts, "cursor", None), make_round=make_round)
        latest = releases.latest()
        membership.join(tape=str(tape), latest_release=None if latest is None else
                        {"seq": latest.seq, "policy_version": latest.version})
        if lease is not None:  # this incarnation now owns the island's shard
            claimed = lease.claim(buffer.entries(), earlier=resume.incarnations)
            emit(LEASE_EVENT, incarnation=incarnation, generation=claimed["generation"],
                 superseded=claimed["superseded"], sealed=claimed["sealed"])
            island.round.write_fenced = lease.check_before_write
            if hasattr(prompts, "fenced"):  # store dispenser: a fenced incarnation renews and writes nothing
                prompts.fenced = lambda: lease.fenced

            def on_fenced() -> None:
                print(f"[engine-sampler] fenced: incarnation {lease.fenced_by} holds the lease of "
                      f"{config.sampler_id}; no group is written from now on", flush=True)
                emit(FENCED_EVENT, incarnation=incarnation, superseded_by=lease.fenced_by)

            fence_watcher = ReclaimWatcher(lease.poll, on_fenced, poll_s=lease_poll_s,
                                           name="sampler-lease-watcher").start()
        start = None
        if predecessor and not resume.resumed:  # a fresh replacement: continue the predecessor's prompts
            from .sample_buffer import FileSampleBuffer, shard_index  # noqa: PLC0415
            from .sampler_resume import HANDOFF_EVENT, handoff_start  # noqa: PLC0415

            def predecessor_entries() -> list[Any]:
                shard = FileSampleBuffer(buffer.store, batch_size=1, index=shard_index(str(predecessor)),
                                         create_index=False)
                return shard.entries()

            start, detail = handoff_start(member_store, str(predecessor), config.sampler_id, predecessor_entries,
                                          wait_s=float(spec.get("handoff_wait_s") or 900.0))
            emit(HANDOFF_EVENT, **detail)
            sweep_predecessor(prompts, member_store, str(predecessor), bool(detail.get("predecessor_left")), emit)
        rounds = 0 if buffer.data_ended else island.run(start)
        if handed_off() and not buffer.data_ended and not reclaimed():
            # a successor continues the prompts in its own shard: this shard gets no more
            # groups, so it ends (the trainer's merged index ends once every shard has)
            buffer.mark_data_end()
        preempted = reclaimed()
        fence = None if preempted else (LeaveReason.FENCED, lease.detail()) if fenced() \
            else None if stop_reason is None else stop_reason()
        if fence is not None:  # E.g. FENCED, fenced="other run"
            reason, detail = fence
            print(f"[engine-sampler] leaving {reason.name}: {dict(detail)}", flush=True)
            leave(reason, rounds=rounds, dropped_groups=island.round.dropped_groups, **dict(detail))
        else:
            leave(LeaveReason.PREEMPTED if preempted else LeaveReason.FINISHED, rounds=rounds,
                  dropped_groups=island.round.dropped_groups)
        if not preempted:
            from yeto.rl.event_echo import FINALIZED_EVENT  # noqa: PLC0415

            # a launcher island's complete tape ends with this record (yeto.launch.tapes);
            # a reclaimed incarnation's tape stays open: the next incarnation continues
            emit(FINALIZED_EVENT, rounds=rounds)
        return rounds
    except BaseException as error:
        try:
            leave(LeaveReason.PREEMPTED if reclaimed() else LeaveReason.FENCED if fenced()
                  else LeaveReason.INFRA_FAILED,
                  error=f"{type(error).__name__}: {error}"[:300])
        except Exception:  # noqa: BLE001 - the original error is the one to report
            pass
        raise
    finally:
        watcher.stop()
        if uploader is not None:
            uploader.stop()
        if fence_watcher is not None:
            fence_watcher.stop()
        closer = getattr(getattr(island, "round", None), "close", None)
        if callable(closer):  # An agentic round deletes its boat sandboxes
            try:
                closer()
            except Exception as error:  # noqa: BLE001 - the island's own error is the one to report
                emit("rl_agentic_close", error=f"{type(error).__name__}: {error}")
        close = getattr(prompts, "close", None)
        if callable(close):  # The rows this sampler holds go back to the dispenser
            try:
                emit("rl_dispense_close", **(close() or {}))
            except Exception as error:  # noqa: BLE001 - the lease returns them anyway
                emit("rl_dispense_close", error=f"{type(error).__name__}: {error}")
        try:
            engine.stop()
        finally:
            if telemetry is not None:
                telemetry.stop()


def incarnation_lease(spec: Mapping[str, Any], sampler_id: str, incarnation: str, default_store: Any,
                      environ: Mapping[str, str] | None = None) -> tuple[Any, float]:
    """``(IncarnationLease | None, poll seconds)`` of ``spec["incarnation_lease"]``:
    absent or ``true`` = on, in the buffer store; ``false`` = off; a mapping may name
    ``store`` (another store path every incarnation reads, e.g. a shared bucket mount),
    ``poll_s`` (watcher period, default 30) and ``write_check_s`` (the longest a group
    write trusts the last read, default 1)."""
    from .incarnation_lease import DEFAULT_POLL_S, DEFAULT_WRITE_CHECK_S, IncarnationLease  # noqa: PLC0415

    section = spec.get("incarnation_lease", True)
    if section is False or section is None:
        return None, DEFAULT_POLL_S
    section = {} if section is True else dict(section)
    store = default_store
    if section.get("store"):
        from .resume import store_for  # noqa: PLC0415

        store = store_for(str(section["store"]), environ=environ)
    lease = IncarnationLease(store, sampler_id, incarnation,
                             write_check_s=float(section.get("write_check_s", DEFAULT_WRITE_CHECK_S)))
    return lease, float(section.get("poll_s", DEFAULT_POLL_S))


def load_spec(path: str | Path) -> dict[str, Any]:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SamplerSpecError(f"cannot read the sampler spec {path}: {error}") from None
    if not isinstance(doc, dict):
        raise SamplerSpecError(f"{path}: the sampler spec must be a JSON object")
    return doc


RECLAIM_HOLD_S = 300.0  # after a reclaim LEAVE: wait this long for the cloud to stop the container
RECLAIM_DEADLINE_S = 300.0  # from the reclaim marker to the process exit, whatever still runs
RECLAIM_EXIT_MARGIN_S = 15.0  # the deadline fires this long before the limit: cleanup + exit
RECLAIM_CLEANUP_S = 10.0  # the longest one forced cleanup step may take at the deadline
# A reclaimed process exits with this code (EX_TEMPFAIL): a watchdog or patrol tells a
# reclaim exit (run the island again) from a finished run (0) and from a failure.
RECLAIM_EXIT_CODE = 75
RECLAIM_EXIT_LOG = "[engine-sampler] reclaim exit"


def reclaim_seen_at(environ: Mapping[str, str], *, clock: Callable[[], float] = time.time) -> float:
    """When the reclaim marker was written (its ``time`` field, else its mtime), never later than now."""
    from .sampler_resume import reclaim_marker  # noqa: PLC0415

    now = clock()
    path = reclaim_marker(environ)
    try:
        doc = json.loads(path.read_text(encoding="utf-8") or "{}")
        stamp = float(doc["time"]) if isinstance(doc, dict) and "time" in doc else path.stat().st_mtime
    except (OSError, ValueError, TypeError, KeyError):
        return now
    return min(stamp, now)


def _flush_std() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:  # noqa: BLE001 - exiting anyway
            pass


def reclaim_exit(reason: str, *, code: int = RECLAIM_EXIT_CODE,
                 exit_now: Callable[[int], Any] | None = None) -> None:
    """End a reclaimed process now with ``code``. ``os._exit``: the interpreter's
    shutdown joins every executor worker thread, and a trial blocked in a boat
    sandbox call (``asyncio.to_thread``) would hold the exit for as long as that
    call blocks."""
    print(f"{RECLAIM_EXIT_LOG} code={code} reason={reason}", flush=True)
    _flush_std()
    (os._exit if exit_now is None else exit_now)(code)


class ReclaimDeadline:
    """Once armed with the marker time, a daemon thread waits until ``deadline_s``
    minus ``margin_s`` after it; if the process is still up then, it runs each
    registered cleanup (bounded by ``cleanup_s``) and calls :func:`reclaim_exit`.
    The normal path (the island leaves, closes, holds, exits) ends earlier."""

    def __init__(self, *, deadline_s: float = RECLAIM_DEADLINE_S, margin_s: float = RECLAIM_EXIT_MARGIN_S,
                 cleanup_s: float = RECLAIM_CLEANUP_S, clock: Callable[[], float] = time.time,
                 exit_now: Callable[[int], Any] | None = None) -> None:
        self.deadline_s, self.margin_s, self.cleanup_s = float(deadline_s), float(margin_s), float(cleanup_s)
        self.clock, self.exit_now = clock, exit_now
        self.cleanups: list[tuple[str, Callable[[], Any]]] = []
        self.armed_at: float | None = None
        self.fired = threading.Event()
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def add_cleanup(self, name: str, fn: Callable[[], Any]) -> None:
        self.cleanups.append((name, fn))

    def exit_at(self) -> float | None:
        """The wall time the process must be gone by, minus the margin (None: not armed)."""
        return None if self.armed_at is None else self.armed_at + self.deadline_s - self.margin_s

    def remaining(self) -> float | None:
        at = self.exit_at()
        return None if at is None else max(0.0, at - self.clock())

    def arm(self, seen_at: float | None = None) -> "ReclaimDeadline":
        with self._lock:
            if self._thread is not None:
                return self
            self.armed_at = self.clock() if seen_at is None else float(seen_at)
            self._thread = threading.Thread(target=self._run, name="sampler-reclaim-deadline", daemon=True)
            self._thread.start()
        return self

    def cancel(self) -> None:
        self._cancel.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def _run(self) -> None:
        left = self.remaining() or 0.0
        if self._cancel.wait(left):
            return
        self.fired.set()
        print(f"[engine-sampler] reclaim deadline reached ({self.deadline_s:g} s after the marker); "
              "forcing the cleanup and the exit", flush=True)
        for name, fn in self.cleanups:
            worker = threading.Thread(target=self._cleanup, args=(name, fn), daemon=True,
                                      name=f"sampler-reclaim-cleanup-{name}")
            worker.start()
            worker.join(self.cleanup_s)
            if worker.is_alive():
                print(f"[engine-sampler] reclaim cleanup {name} still blocks after {self.cleanup_s:g} s",
                      flush=True)
        reclaim_exit("deadline", exit_now=self.exit_now)

    @staticmethod
    def _cleanup(name: str, fn: Callable[[], Any]) -> None:
        try:
            fn()
        except Exception as error:  # noqa: BLE001 - exiting anyway
            print(f"[engine-sampler] reclaim cleanup {name} failed: {type(error).__name__}: {error}",
                  flush=True)


def hold_after_reclaim(environ: Mapping[str, str] | None = None, *, hold_s: float = RECLAIM_HOLD_S,
                       sleep: Callable[[float], None] | None = None,
                       deadline: ReclaimDeadline | None = None) -> bool:
    """After a reclaim the process stays up until the cloud stops the container:
    a clean early exit could end the cloud's input as finished instead of letting
    the cloud run the island again (as the Miles sampler island does). With
    ``deadline`` the hold ends at its exit time (a reclaim no cloud follows, e.g.
    a manual marker, still ends the process in time). Returns whether a reclaim
    was requested (False: returns at once)."""
    from .sampler_resume import reclaim_requested  # noqa: PLC0415

    env = os.environ if environ is None else environ
    if not reclaim_requested(env):
        return False
    if deadline is not None:
        deadline.arm(reclaim_seen_at(env, clock=deadline.clock))
        hold_s = min(hold_s, deadline.remaining() or 0.0)
    print(f"[engine-sampler] reclaimed: waiting up to {hold_s:g} s for the container to stop", flush=True)
    (time.sleep if sleep is None else sleep)(hold_s)
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m yeto.rl.engine.engine_sampler",
                                     description="run an engine sampler island from a spec file")
    parser.add_argument("--spec", required=True, help="the sampler spec JSON (module doc)")
    args = parser.parse_args(argv)
    deadline = ReclaimDeadline()
    try:
        spec = load_spec(args.spec)
        rounds = run(spec, reclaim_deadline=deadline)
    except SamplerSpecError as error:
        print(f"[engine-sampler] spec rejected: {error}", file=sys.stderr)
        return 2
    except Exception as error:
        # after a reclaim, an ordinary error is most likely the teardown under way;
        # KeyboardInterrupt and SystemExit are not converted, and the original type
        # stays in the log line so a real failure can still be told apart
        if deadline.armed_at is None:
            raise
        import traceback  # noqa: PLC0415

        traceback.print_exc()
        reclaim_exit(f"error after the reclaim: {type(error).__module__}.{type(error).__qualname__}: "
                     f"{str(error)[:200]}")
        raise
    print(f"[engine-sampler] done: {rounds} round(s)", flush=True)
    if hold_after_reclaim(deadline=deadline):
        reclaim_exit("held")
    return 0

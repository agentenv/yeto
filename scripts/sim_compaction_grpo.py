"""CPU end-to-end simulation of async GRPO with context compaction (CompactionRL segments).

Real pieces: the records proxy (``RecordsProxy`` -> ``RecordingBackend`` with a
``CompactionConfig``), the gateway's Chat route, the outer agent layer
(``run_agent_task``), ``AgenticRound`` writing groups through ``GroupWriter``
into a local-disk sample buffer (``open_buffer`` over ``YETO_SAMPLE_BUFFER_ROOT``),
the trainer's ``BatchReader`` / ``miles_groups`` (``miles_sample_dict``), Miles'
``Sample`` (when Miles is importable) and Miles' GRPO reward normalization
(``reward_group_segments`` / ``rollout_segments`` from Miles when importable;
``_normalize_rewards_by_rollout`` itself needs Miles' train-data imports, so a
line-for-line port runs here, see :func:`normalize_rewards_by_rollout`).

Fake pieces:

* tokenizer / chat template: a word-level hashing tokenizer with a ChatML-like
  template (no HF download); tool calls are Qwen3 ``<tool_call>{json}</tool_call>``
  (parsed by ``parse_qwen3``);
* engine: :class:`SimEngine`, a :class:`FakeEngine` whose replies come from a
  scripted policy (tool calls until the environment says ``remaining=0``,
  then a final answer; a summary when the proxy asks for one), with
  log-probs, top-K rows (sampled token first) and small R3 rows, and an
  in-place weight reload after configured call counts (live update);
* agent: :class:`SimAgent`, a pi-like loop that sends the whole history to the
  records proxy's Chat Completions route every turn, runs each tool call in
  :class:`SimEnvironment` and returns the 0/1 verifier reward;
* environment: tool output lengths and tool run times from configured
  distributions; a trial may fail (ABORTED) after some calls.

Time is simulated (nothing sleeps): an LLM call costs
``call_overhead_s + prefill_s_per_token * uncached prompt tokens +
decode_s_per_token * completion tokens``; a tool call costs a draw from
``tool_time_s``. Two totals per trial: ``env_s`` (tool time only) and
``env_s + 0.1 * llm_s``. The real CPU wall clock of each phase is reported too
(yeto's own overhead; the fake engine's own time is subtracted per call).

Paths:

* ``agentic``: ``AgenticRound`` + ``ProxyTokenRecorder`` (HTTP on 127.0.0.1) ->
  buffer -> trainer read -> normalization. Without compaction this runs on
  main. With compaction it needs the sampler-side segment samples (C1); on main
  ``ProxyTokenRecorder`` refuses a compacting proxy.
* ``records``: the agent drives the records proxy in process (no AgenticRound);
  each recorded segment is joined on its own and turned into a buffer sample
  with the real ``agentic_sample`` plus ``metadata.compaction`` as the
  compaction-1118 contract fixes it (a stand-in until C1 lands); the groups go
  through the buffer and the trainer read (segment groups need C3: on main the
  reader refuses a group with more samples than ``n``) and through the
  normalization directly.

Run: ``python scripts/sim_compaction_grpo.py --scenario all`` (``--json out.json``,
``--path records|agentic|auto``). Add the Miles checkout to ``PYTHONPATH``
(or ``--miles-root``) to use Miles' own ``Sample`` and segment helpers.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import hashlib
import inspect
import json
import math
import random
import re
import statistics
import sys
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy

from yeto.rl.adapters.sampler_engines.fake import FakeEngine
from yeto.rl.compaction import SUMMARY_PROMPT, CompactionConfig
from yeto.rl.engine.sampler_engine import FINISH_LENGTH, FINISH_STOP, Completion, Provenance, SamplingParams
from yeto.rl.harness.chat_templates import ChatTemplate, TemplateSpec, parse_qwen3
from yeto.rl.harness.gateway.proxy import CURRENT_TRAJECTORY

SIGNED_MODEL = "policy"
GEN_PROMPT = "<|im_start|>assistant\n"
STOP = "<|im_end|>"
TOOLS = [{"type": "function", "function": {
    "name": "shell", "description": "run a shell command in the task sandbox",
    "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}}]
SYSTEM = ("You are a coding agent working in a sandbox. Call the shell tool to inspect and change files; "
          "answer without a tool call when the task is done.")
WORDS = ("file", "line", "test", "error", "patch", "build", "check", "value", "module", "config", "path", "read",
         "write", "parse", "fix", "run", "output", "diff", "commit", "import", "class", "method", "list", "index",
         "cache", "token", "shape", "batch", "trace", "stack", "branch", "merge", "status", "result", "number")
COMPACTION_SCHEMA = 1


# ============================================================================ tokenizer and template
class SimTokenizer:
    """Word-level hashing tokenizer: special tokens, `` ?word``, single whitespace,
    single punctuation. Ids are stable hashes (no vocabulary file); encoding a
    rendering piece by piece at turn boundaries gives the same ids as encoding it
    whole, so the records proxy sees no render drift."""

    SPECIAL = {"<|im_start|>": 1, STOP: 2}
    PATTERN = re.compile(r"<\|im_start\|>|<\|im_end\|>| ?[A-Za-z0-9_]+|\s|[^A-Za-z0-9_\s]")

    def __init__(self) -> None:
        self._text: dict[int, str] = {v: k for k, v in self.SPECIAL.items()}
        self._lock = threading.Lock()

    def _id(self, piece: str) -> int:
        special = self.SPECIAL.get(piece)
        if special is not None:
            return special
        token = 16 + int(hashlib.blake2b(piece.encode(), digest_size=4).hexdigest(), 16) % (2**31 - 16)
        known = self._text.get(token)
        if known is None:
            with self._lock:
                self._text.setdefault(token, piece)
                known = self._text[token]
        if known != piece:
            raise ValueError(f"sim tokenizer hash collision: {piece!r} / {known!r}")
        return token

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self._id(m.group(0)) for m in self.PATTERN.finditer(text)]

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        return "".join(self._text[int(t)] for t in ids)

    @staticmethod
    def apply_chat_template(messages: Sequence[Mapping[str, Any]], tools: Sequence[Any] | None = None,
                            tokenize: bool = False, add_generation_prompt: bool = False, **_: Any) -> str:
        out = []
        if tools:
            out.append("<|im_start|>tools\n" + json.dumps(list(tools), sort_keys=True) + STOP + "\n")
        for message in messages:
            body = str(message.get("content") or "")
            if message.get("role") == "assistant":
                for call in message.get("tool_calls") or ():
                    function = call["function"]
                    body += "<tool_call>" + json.dumps({"name": function["name"], "arguments": function["arguments"]},
                                                       ensure_ascii=False) + "</tool_call>"
            out.append(f"<|im_start|>{message['role']}\n{body}{STOP}\n")
        if add_generation_prompt:
            out.append(GEN_PROMPT)
        return "".join(out)


SIM_TEMPLATE = TemplateSpec(name="sim_chatml", tokenizer="sim", parser=parse_qwen3, stop_tokens=(STOP,),
                            keeps_history_reasoning=True, verified="sim word tokenizer (CPU simulation)")


def sim_template() -> ChatTemplate:
    return ChatTemplate(SIM_TEMPLATE, SimTokenizer())


# ============================================================================ scenario
@dataclasses.dataclass(frozen=True)
class Dist:
    """``const`` (a), ``uniform`` [a, b], ``int`` uniform integer [a, b] or
    ``lognormal`` (median a, sigma b)."""

    kind: str
    a: float
    b: float = 0.0

    def sample(self, rng: random.Random) -> float:
        if self.kind == "const":
            return float(self.a)
        if self.kind == "uniform":
            return rng.uniform(self.a, self.b)
        if self.kind == "int":
            return float(rng.randint(int(self.a), int(self.b)))
        if self.kind == "lognormal":
            return float(self.a) * math.exp(rng.gauss(0.0, self.b))
        raise ValueError(f"unknown distribution {self.kind!r}")


@dataclasses.dataclass(frozen=True)
class Scenario:
    name: str
    description: str = ""
    tasks: int = 4
    n: int = 8
    compaction: Mapping[str, int] | None = None  # spec agentic.compaction (None: off)
    max_model_len: int = 4096
    turn_max_tokens: int = 256
    max_response_len: int = 4096
    steps: Dist = Dist("int", 2, 4)  # tool calls a trial needs before it can answer
    action_tokens: Dist = Dist("int", 30, 90)  # text before the tool call
    final_tokens: Dist = Dist("int", 10, 30)
    summary_tokens: Dist = Dist("int", 60, 120)
    summary_ok_p: float = 0.9  # the summary closes its <summary> tag
    tool_tokens: Dist = Dist("int", 120, 400)  # tool output length
    task_tokens: Dist = Dist("int", 40, 80)
    tool_time_s: Dist = Dist("lognormal", 2.0, 0.8)
    success_p: Mapping[int, float] | float = 0.5  # reward 1 chance of a finished trial (per task index)
    abort_p: float = 0.0  # a trial fails (ABORTED) after 1-3 model calls
    reload_after_calls: tuple[int, ...] = ()  # engine call counts that load a new version in place
    prefill_s_per_token: float = 0.0002
    decode_s_per_token: float = 0.02
    call_overhead_s: float = 0.05
    llm_weight: float = 0.1  # the second timing total: env + llm_weight * llm
    rollout_batch_size: int = 2  # groups per trainer step
    max_policy_age: int = 4
    r3: tuple[int, int, int] = (16, 2, 2)  # (experts, MoE layers, top_k)
    top_k: int = 4
    seed: int = 1118
    max_agent_turns: int = 200

    def compaction_config(self) -> CompactionConfig | None:
        return None if self.compaction is None else CompactionConfig(**dict(self.compaction))

    def success(self, task: int) -> float:
        if isinstance(self.success_p, Mapping):
            return float(self.success_p.get(task, 0.5))
        return float(self.success_p)


SMALL = {"context_budget": 4096, "t_comp": 512, "keep_recent_steps": 2, "max_compactions": 3}
# t_comp 512 leaves room for one step (action <= ~90 tokens, tool output <= ~125) plus the
# summary request (~140) and the summary (<= ~135); ``tcomp_overflow`` breaks that on purpose.
SHORT_STEP = {"action_tokens": Dist("int", 20, 60), "tool_tokens": Dist("int", 60, 110),
              "summary_tokens": Dist("int", 40, 90)}
MIXED = {0: 0.5, 1: 0.25, 2: 0.75, 3: 0.5}
SCENARIOS: dict[str, Scenario] = {s.name: s for s in (
    Scenario("no_compaction", "compaction off, short trials (main format)", compaction=None, max_model_len=16384,
             max_response_len=16384, steps=Dist("int", 2, 5), success_p=MIXED),
    Scenario("no_trigger", "compaction on (4096 / 512), trials too short to trigger it", compaction=SMALL,
             steps=Dist("int", 3, 8), success_p=MIXED, **SHORT_STEP),
    Scenario("compact_once", "compaction on (4096 / 512), about one compaction per trial", compaction=SMALL,
             steps=Dist("int", 20, 28), success_p=MIXED, **SHORT_STEP),
    Scenario("compact3_truncated", "trials that never finish: 3 compactions, then the context fills (truncated)",
             compaction=SMALL, steps=Dist("int", 500, 500), success_p=1.0, **SHORT_STEP),
    Scenario("mixed_reload", "short to long trials, two in-place weight reloads during the round (live)",
             compaction=SMALL, steps=Dist("int", 3, 40), reload_after_calls=(300, 600), success_p=MIXED,
             **SHORT_STEP),
    Scenario("partial_abort", "a quarter of the trials fail after 1-3 calls (ABORTED, left out of the group)",
             compaction=SMALL, steps=Dist("int", 10, 30), abort_p=0.25, success_p=MIXED, **SHORT_STEP),
    Scenario("tcomp_overflow", "steps longer than t_comp allows: the summary request itself overflows the context",
             compaction=SMALL, steps=Dist("int", 9, 12), tool_tokens=Dist("int", 300, 450), success_p=MIXED),
)}


def _rng(*parts: Any) -> random.Random:
    return random.Random(int(hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).hexdigest(), 16))


def words(rng: random.Random, n: int) -> str:
    """``n`` tokens of filler text (one token per word under :class:`SimTokenizer`)."""
    picked = [rng.choice(WORDS) for _ in range(max(1, int(n)))]
    return picked[0] + "".join(" " + w for w in picked[1:])


# ============================================================================ engine
@dataclasses.dataclass
class CallRecord:
    trajectory: str | None
    call: int
    kind: str  # action | final | summary
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    version: int
    finish: str
    engine_cpu_s: float  # this call's CPU time in the fake engine (thread time)

    def llm_s(self, scenario: Scenario) -> float:
        return (scenario.call_overhead_s + scenario.prefill_s_per_token * (self.prompt_tokens - self.cached_tokens)
                + scenario.decode_s_per_token * self.completion_tokens)


class SimPolicy:
    """The scripted policy: tool calls while the last tool output says
    ``remaining=R > 0``, a final answer at ``remaining=0``, a summary when the
    prompt ends with the proxy's summary request."""

    SUMMARY_TAIL = SUMMARY_PROMPT + STOP + "\n" + GEN_PROMPT
    REMAINING = re.compile(r"remaining=(\d+)")

    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario

    def reply(self, prompt_text: str, rng: random.Random) -> tuple[str, str]:
        s = self.scenario
        found = self.REMAINING.findall(prompt_text)
        remaining = int(found[-1]) if found else None
        if prompt_text.endswith(self.SUMMARY_TAIL):
            state = f"state remaining={remaining} " if remaining is not None else ""
            text = (f"<analysis>{words(rng, s.summary_tokens.sample(rng) // 3)}</analysis>\n<summary>{state}"
                    f"{words(rng, s.summary_tokens.sample(rng))}")
            return "summary", text + ("</summary>" if rng.random() < s.summary_ok_p else "")
        if remaining == 0:
            return "final", words(rng, s.final_tokens.sample(rng)) + " DONE"
        call = json.dumps({"name": "shell", "arguments": {"cmd": f"step {rng.randrange(1000)}"}}, ensure_ascii=False)
        return "action", f"{words(rng, s.action_tokens.sample(rng))}<tool_call>{call}</tool_call>"


class SimEngine(FakeEngine):
    """:class:`FakeEngine` with the scripted policy (module doc). ``live``: the
    engine updates its weights during generation (``updates_during_generation``,
    each completion carries its version); ``reload_after_calls`` then loads a new
    version in place before those engine calls."""

    def __init__(self, scenario: Scenario, template: ChatTemplate, *, live: bool | None = None) -> None:
        super().__init__(max_model_len=scenario.max_model_len, routed_experts=scenario.r3,
                         top_logprobs_num=scenario.top_k, sampler_id="sim-0")
        self.scenario = scenario
        self.template = template
        self.policy = SimPolicy(scenario)
        live = bool(scenario.reload_after_calls) if live is None else live
        if live:
            self.updates_during_generation = True
        self.live = live
        self.calls = 0
        self.ledger: list[CallRecord] = []
        self.reloads: list[tuple[int, int]] = []  # (engine call, new version)
        self._per_trajectory: dict[str | None, tuple[int, tuple[int, ...]]] = {}
        self._lock = threading.Lock()

    def generate_tokens(self, prompts: Sequence[Sequence[int]], sampling: SamplingParams, *,
                        stop_token_ids: Sequence[int] = ()) -> list[list[Completion]]:
        from yeto.rl.engine.sampler_engine import SamplerEngineError

        if self._version is None or self.stopped:
            raise SamplerEngineError("sim engine is not serving")
        started = time.thread_time()
        trajectory = CURRENT_TRAJECTORY.get()
        stops = {int(t) for t in stop_token_ids}
        out = []
        for raw in prompts:
            prompt = tuple(int(t) for t in raw)
            with self._lock:
                self.calls += 1
                if self.calls in self.scenario.reload_after_calls:
                    new = int(self._version) + 1
                    self.load(Path(f"sim-release-v{new}"), new)
                    self.reloads.append((self.calls, new))
                version = int(self._version)
                number, last = self._per_trajectory.get(trajectory, (0, ()))
            rng = _rng(self.scenario.seed, trajectory, number, len(prompt))
            kind, text = self.policy.reply(self.template.decode(prompt), rng)
            reply = self.template.encode(text) + [self.template.stop_token_ids[0]]
            ids: list[int] = []
            for token in reply[: sampling.max_tokens]:
                ids.append(int(token))
                if token in stops:
                    break
            finish = FINISH_STOP if ids and ids[-1] in stops else FINISH_LENGTH
            logprobs = tuple(-0.02 - 0.9 * _rng(trajectory, number, k, version).random() for k in range(len(ids)))
            arrays = self._sim_arrays(rng, ids, logprobs, len(prompt))
            cached = len(last) if prompt[: len(last)] == last else 0
            with self._lock:
                self._per_trajectory[trajectory] = (number + 1, prompt + tuple(ids))
            out.append([Completion(token_ids=tuple(ids), logprobs=logprobs, text="", finish_reason=finish,
                                   prompt_token_ids=prompt,
                                   weight_version=version if self.live else None,
                                   final_weight_version=version if self.live else None, **arrays)])
            self.ledger.append(CallRecord(trajectory, number, kind, len(prompt), cached, len(ids), version, finish,
                                          time.thread_time() - started))
            started = time.thread_time()
        return out

    def _sim_arrays(self, rng: random.Random, ids: Sequence[int], logprobs: Sequence[float],
                    prompt_tokens: int) -> dict[str, Any]:
        """Top-K rows (the sampled token and its log-prob first, then lower ones) and R3
        rows ``[prompt + completion - 1, layers, top_k]`` of experts below ``experts``."""
        experts, layers, top_k = self.scenario.r3
        k = self.scenario.top_k
        top_ids = numpy.zeros((len(ids), k), dtype=numpy.int32)
        top_vals = numpy.zeros((len(ids), k), dtype=numpy.float32)
        for t, (token, lp) in enumerate(zip(ids, logprobs)):
            top_ids[t] = [int(token)] + [(int(token) + c) % 4093 + 3 for c in range(1, k)]
            top_vals[t] = [lp - 0.5 * c for c in range(k)]
        gen = numpy.random.default_rng(rng.randrange(2**31))
        rows = max(prompt_tokens + len(ids) - 1, 0)
        routed = gen.integers(0, experts, size=(rows, layers, top_k), dtype=numpy.uint16)
        return {"top_logprobs_ids": top_ids, "top_logprobs": top_vals, "routed_experts": routed}


# ============================================================================ environment and agent
class SimEnvironment:
    """One trial's sandbox: ``steps`` tool calls finish the task; each tool call
    returns ``[env remaining=R]`` plus filler and takes a simulated time."""

    def __init__(self, scenario: Scenario, rng: random.Random, *, task: int) -> None:
        self.scenario = scenario
        self.rng = rng
        self.steps = int(scenario.steps.sample(rng))
        self.done = 0
        self.success = rng.random() < scenario.success(task)
        self.abort_after = (rng.randint(1, 3) if rng.random() < scenario.abort_p else None)
        self.tool_s: list[float] = []

    def execute(self, call: Mapping[str, Any]) -> str:
        self.done += 1
        remaining = max(0, self.steps - self.done)
        self.tool_s.append(max(0.0, self.scenario.tool_time_s.sample(self.rng)))
        return f"[env remaining={remaining}] {words(self.rng, self.scenario.tool_tokens.sample(self.rng))}"

    def reward(self, answered: bool) -> float:
        return 1.0 if answered and self.done >= self.steps and self.success else 0.0


class InProcess:
    """The records proxy's Chat route without HTTP (``RecordsProxy.handle`` with the
    route's error mapping)."""

    def __init__(self, proxy: Any) -> None:
        self.proxy = proxy

    async def chat(self, trajectory_id: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        from yeto.rl.harness.chat_templates import TemplateError
        from yeto.rl.harness.gateway.core import GatewayError

        try:
            return 200, await self.proxy.handle("chat", body, trajectory_id=trajectory_id)
        except TemplateError as exc:
            return 400, {"error": {"type": type(exc).__name__, "message": str(exc)}}
        except GatewayError as exc:
            return 409, {"error": {"type": type(exc).__name__, "message": str(exc)}}


class SimAgent:
    """pi-like agent (``AgentHarness``): every turn it sends the whole history and
    the tool schemas to ``<endpoint>/chat/completions``; it runs every tool call and
    answers when the model replies without one. ``inproc://<trajectory>``
    endpoints go to :class:`InProcess` instead of HTTP. A refused call (the
    context is full) ends the agent with a non-zero exit, as pi does."""

    def __init__(self, scenario: Scenario, *, inproc: InProcess | None = None) -> None:
        self.scenario = scenario
        self.inproc = inproc
        self.trials: dict[str, dict[str, Any]] = {}
        self._http: Any = None

    async def _post(self, endpoint: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        if endpoint.startswith("inproc://"):
            assert self.inproc is not None
            return await self.inproc.chat(endpoint[len("inproc://"):], body)
        import aiohttp

        async with aiohttp.ClientSession() as http:
            async with http.post(f"{endpoint}/chat/completions", json=body) as reply:
                return reply.status, await reply.json()

    async def run(self, task: Any, endpoint: str) -> Any:
        from yeto.rl.agentic import TrialVerdict

        tid = str(task.trajectory_id)
        index = int(task.task_id.rsplit("-", 1)[-1]) if task.task_id and task.task_id[-1].isdigit() else 0
        rng = _rng(self.scenario.seed, "trial", tid)
        env = SimEnvironment(self.scenario, rng, task=index)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Task {task.task_id}: {words(rng, self.scenario.task_tokens.sample(rng))}"}]
        info = {"task": task.task_id, "env": env, "calls": 0, "answered": False, "exit": None,
                "call_wall_s": [], "statuses": Counter()}
        self.trials[tid] = info
        while info["calls"] < self.scenario.max_agent_turns:
            began = time.perf_counter()
            status, reply = await self._post(endpoint, {"model": SIGNED_MODEL, "messages": messages, "tools": TOOLS})
            info["call_wall_s"].append(time.perf_counter() - began)
            info["statuses"][status] += 1
            if status != 200:
                info["exit"] = f"{status}: {str((reply.get('error') or {}).get('message'))[:120]}"
                return TrialVerdict(rewards={"reward": 0.0}, exception_type="NonZeroAgentExitCodeError",
                                    exception_message=info["exit"], timings_s={"env_s": sum(env.tool_s)})
            info["calls"] += 1
            message = dict(reply["choices"][0]["message"])
            messages.append({k: message[k] for k in ("role", "content", "tool_calls") if k in message})
            if env.abort_after is not None and info["calls"] >= env.abort_after:
                info["exit"] = "sandbox lost"
                return TrialVerdict(rewards=None, exception_type="SandboxLostError",
                                    exception_message="simulated sandbox loss")
            calls = message.get("tool_calls") or []
            if not calls:
                info["answered"] = reply["choices"][0].get("finish_reason") != "length"
                break
            for call in calls:
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": env.execute(call)})
        reward = env.reward(info["answered"])
        return TrialVerdict(rewards={"reward": reward}, timings_s={"env_s": sum(env.tool_s)},
                            agent_metrics={"tool_calls": env.done, "steps_needed": env.steps})


# ============================================================================ capabilities (integration probes)
def capabilities() -> dict[str, bool]:
    """Which compaction-1118 parts are in this tree (the simulation adapts; tests skip):

    * ``sampler_segments`` (C1): ``ProxyTokenRecorder`` accepts a compacting proxy;
    * ``spec_compaction`` (C2): the spec's ``agentic.compaction`` reaches the proxy
      (``round_factory`` reads it);
    * ``trainer_segments`` (C3): the trainer keeps a group of segment samples
      (more samples than ``n``, siblings sharing ``index``) with its trial identity."""
    from yeto.rl.agentic import HarnessConfigError
    from yeto.rl.engine import agentic_round
    from yeto.rl.harness.gateway.proxy import RecordsProxy
    from yeto.rl.harness.gateway.recorder import ProxyTokenRecorder

    out = {}
    template = sim_template()
    engine = FakeEngine(max_model_len=4096)
    try:
        ProxyTokenRecorder(RecordsProxy(engine, template, model=SIGNED_MODEL, signed_sampling={"max_tokens": 8},
                                        compaction=CompactionConfig(context_budget=4096, t_comp=512)))
        out["sampler_segments"] = True
    except HarnessConfigError:
        out["sampler_segments"] = False
    out["spec_compaction"] = "compaction" in inspect.getsource(agentic_round.round_factory)
    out["trainer_segments"] = _trainer_keeps_segments()
    return out


def _trainer_keeps_segments() -> bool:
    from yeto.rl.adapters.miles.buffer_rollout import renumber_group
    from yeto.rl.engine.buffer_payload import SampleBufferRolloutError

    meta = lambda i, k: {"trajectory_id": f"t{i}", "compaction": {"schema": 1, "trajectory_id": f"t{i}",  # noqa: E731
                                                                  "segment_index": k, "num_segments": 2}}
    group = [{"group_index": 0, "index": i, "metadata": meta(i, k)} for i in range(2) for k in range(2)]
    try:
        out = renumber_group(group, position=5, samples_per_group=2)
    except SampleBufferRolloutError:
        return False
    keys = [(d.get("rollout_id"), d["index"]) for d in out]
    return keys[0] == keys[1] and keys[2] == keys[3] and keys[0] != keys[2]


# ============================================================================ Miles normalization
def miles_imports(miles_root: str | None = None) -> SimpleNamespace:
    """Miles' ``Sample`` and segment helpers when importable (None fields otherwise)."""
    if miles_root and miles_root not in sys.path:
        sys.path.insert(0, miles_root)
    out = SimpleNamespace(Sample=None, reward_group_segments=None, rollout_segments=None,
                          normalize=None, source="reference")
    try:
        from miles.rollout.reward_shaping import reward_group_segments, rollout_segments
        from miles.utils.types import Sample
    except ImportError:
        return out
    out.Sample, out.reward_group_segments, out.rollout_segments = Sample, reward_group_segments, rollout_segments
    out.source = "miles reward_shaping + reference _normalize_rewards_by_rollout"
    try:
        from miles.ray.rollout.train_data_conversion import _normalize_rewards_by_rollout
    except ImportError:
        return out
    out.normalize, out.source = _normalize_rewards_by_rollout, "miles"
    return out


def _reward_group_segments(args: Any, samples: Sequence[Any], prompt_group_sizes: list[int] | None = None
                           ) -> list[list[int]]:
    """Port of miles/rollout/reward_shaping.py::reward_group_segments (miles-k31 778794a8, lines 34-59)."""
    if prompt_group_sizes is not None:
        assert sum(prompt_group_sizes) == len(samples)
        groups, start = [], 0
        for size in prompt_group_sizes:
            if size > 0:
                groups.append(list(range(start, start + size)))
            start += size
        return groups
    if all(sample.group_index is not None for sample in samples):
        by_prompt: dict[int, list[int]] = {}
        for row, sample in enumerate(samples):
            by_prompt.setdefault(int(sample.group_index), []).append(row)
        return list(by_prompt.values())
    if len(samples) == args.n_samples_per_prompt * args.rollout_batch_size:
        return [list(range(start, start + args.n_samples_per_prompt))
                for start in range(0, len(samples), args.n_samples_per_prompt)]
    return [list(range(len(samples)))]


def _rollout_segments(samples: Sequence[Any], rows: list[int]) -> list[list[int]]:
    """Port of miles/rollout/reward_shaping.py::rollout_segments (miles-k31 778794a8, lines 62-69)."""
    by_rollout: dict[Any, list[int]] = {}
    for row in rows:
        sample = samples[row]
        key = sample.rollout_id if sample.rollout_id is not None else sample.index
        by_rollout.setdefault(key if key is not None else ("row", row), []).append(row)
    return list(by_rollout.values())


def normalize_rewards_by_rollout(miles: SimpleNamespace, args: Any, samples: Sequence[Any],
                                 raw_rewards: list[float], prompt_group_sizes: list[int] | None = None, *,
                                 unshaped_rewards: list[float] | None = None) -> list[float]:
    """Miles' ``_normalize_rewards_by_rollout`` (miles/ray/rollout/train_data_conversion.py,
    miles-k31 778794a8, lines 323-376): the Miles function itself when that module
    imports (it needs Miles' train-data dependencies), else this line-for-line port:
    torch as in Miles when torch imports, numpy float32 with the unbiased std
    (``torch.std``) otherwise. The segment helpers are Miles' own when
    ``miles.rollout.reward_shaping`` imports, else the ports above."""
    if miles.normalize is not None:
        return miles.normalize(args, list(samples), raw_rewards, prompt_group_sizes,
                               unshaped_rewards=unshaped_rewards)
    try:
        import torch
    except ImportError:
        torch = None
    group_segments = miles.reward_group_segments or _reward_group_segments
    segments_of = miles.rollout_segments or _rollout_segments
    if not samples:
        return []
    normalized = [0.0] * len(raw_rewards)
    for prompt_segments in group_segments(args, list(samples), prompt_group_sizes):
        groups = segments_of(list(samples), prompt_segments)
        shared = []
        for rows in groups:
            values = [raw_rewards[r] for r in rows]
            if any(v != values[0] for v in values[1:]):
                raise ValueError(f"all samples in a rollout must share one reward; rows {rows} have rewards {values}")
            shared.append(values[0])
        length_only = unshaped_rewards is not None and all(unshaped_rewards[r] == 1.0 for r in prompt_segments)
        scale = (args.advantage_estimator in ["grpo", "gspo"] and args.grpo_std_normalization and len(shared) > 1
                 and not length_only)
        if torch is not None:
            rewards = torch.tensor(shared, dtype=torch.float)
            centered = rewards - rewards.mean()
            if scale:
                std = rewards.std()
                if std > 0:
                    centered = centered / (std + 1e-6)
        else:
            rewards = numpy.asarray(shared, dtype=numpy.float32)
            centered = rewards - rewards.mean(dtype=numpy.float32)
            if scale:
                std = numpy.float32(rewards.std(ddof=1, dtype=numpy.float32))
                if std > 0:
                    centered = centered / (std + numpy.float32(1e-6))
        for rows, value in zip(groups, centered.tolist(), strict=True):
            for r in rows:
                normalized[r] = float(value)
    return normalized


def miles_args(scenario: Scenario) -> SimpleNamespace:
    return SimpleNamespace(advantage_estimator="grpo", rewards_normalization=True, grpo_std_normalization=True,
                           n_samples_per_prompt=scenario.n, rollout_batch_size=scenario.rollout_batch_size,
                           reward_key=None, length_penalty_coeff=0.0)


def to_miles_samples(miles: SimpleNamespace, dicts: Sequence[Mapping[str, Any]]) -> list[Any]:
    """Miles ``Sample`` objects (``Sample.from_dict``) or, without Miles, namespaces with
    the fields the normalization reads (``group_index``, ``index``, ``rollout_id``, ``reward``)."""
    if miles.Sample is not None:
        return [miles.Sample.from_dict(dict(d)) for d in dicts]
    return [SimpleNamespace(group_index=d.get("group_index"), index=d.get("index"), rollout_id=d.get("rollout_id"),
                            reward=d.get("reward"), metadata=d.get("metadata") or {}) for d in dicts]


def trainer_advantages(miles: SimpleNamespace, scenario: Scenario, samples: Sequence[Any]) -> list[float]:
    rewards = [float(s.reward) for s in samples]
    return normalize_rewards_by_rollout(miles, miles_args(scenario), samples, rewards, unshaped_rewards=rewards)


# ============================================================================ records -> segment samples
def trial_key(sample: Mapping[str, Any]) -> str:
    """The trial a buffer sample belongs to: ``metadata.compaction.trajectory_id``, else
    ``metadata.trajectory_id``."""
    meta = sample.get("metadata") or {}
    comp = meta.get("compaction")
    return str(comp["trajectory_id"]) if isinstance(comp, Mapping) else str(meta.get("trajectory_id"))


def records_segment_samples(item: Any, trajectory: Any, records: Mapping[str, Any], *, group_index: int,
                            index: int, cfg: CompactionConfig | None, max_response_len: int,
                            provenance: Mapping[str, str], live: bool) -> list[dict[str, Any]]:
    """One buffer sample per recorded segment, in the compaction-1118 contract form
    (stand-in for C1): each segment joined on its own (``turns_from_records`` +
    ``join_turns``), written by the real ``agentic_sample``; a trial of several
    segments gets ``metadata.compaction`` on each, only its last segment may be
    ``truncated``; a single-segment trial is the main sample."""
    from yeto.rl.agentic.finish import FinishedSample
    from yeto.rl.agentic.tokens import join_turns
    from yeto.rl.engine.agentic_round import agentic_sample
    from yeto.rl.harness.gateway.recorder import turns_from_records

    segments = [seg for session in records["sessions"] for seg in session["samples"]]
    joined = []
    for seg in segments:
        turns = turns_from_records({"sessions": [{"samples": [seg]}]})
        joined.append((seg, turns, join_turns(turns, max_response_len=max_response_len, rebase=True)))
    trained = [sum(j.response_mask) for _, _, j in joined]
    out = []
    for k, (seg, turns, tokens) in enumerate(joined):
        versions = [int(t.policy_version) for t in turns if t.policy_version is not None]
        finished = FinishedSample(trajectory, tokens, turns)
        sample = agentic_sample(item, finished, version=min(versions), provenance=provenance,
                                group_index=group_index, index=index,
                                versions=(min(versions), max(versions)) if live else None)
        if len(joined) > 1:
            last = k == len(joined) - 1
            context_full = any(str(t.policy_error or "").startswith("context_full") for t in turns)
            if not last:
                sample["status"] = "completed"
                sample["metadata"]["finish_reason"] = FINISH_STOP
            elif context_full:
                sample["status"] = "truncated"
            sample["metadata"]["compaction"] = {
                "schema": COMPACTION_SCHEMA, "trajectory_id": sample["metadata"]["trajectory_id"],
                "segment_index": k, "num_segments": len(joined),
                "context_budget": None if cfg is None else cfg.context_budget,
                "summary_ok": seg.get("summary_ok"), "kept_steps": seg.get("kept_steps"),
                "segment_tokens": trained[k], "tokens_after": sum(trained[k + 1:])}
        out.append(sample)
    return out


def check_sample_alignment(sample: Mapping[str, Any], scenario: Scenario) -> list[str]:
    """Token / loss mask / log-prob / top-K / R3 alignment of one buffer sample
    (empty: aligned)."""
    from yeto.rl.engine.buffer_payload import ROLLOUT_ROUTED_EXPERTS, ROLLOUT_TOP_LOGPROBS, ROLLOUT_TOP_LOGPROBS_IDS

    problems = []
    n = int(sample["response_length"])
    tokens = list(sample["tokens"])
    mask = numpy.asarray(sample["loss_mask"], dtype=int)
    lps = numpy.asarray(sample["rollout_log_probs"], dtype=float)
    if not len(mask) == len(lps) == n or n > len(tokens):
        problems.append(f"lengths: response {n}, mask {len(mask)}, logprobs {len(lps)}, tokens {len(tokens)}")
        return problems
    response = numpy.asarray(tokens[len(tokens) - n:])
    if (lps[mask == 0] != 0.0).any():
        problems.append("mask-0 position with a non-zero log-prob")
    if (lps[mask == 1] >= 0.0).any():
        problems.append("mask-1 position without a log-prob")
    top_ids, top_vals = sample.get(ROLLOUT_TOP_LOGPROBS_IDS), sample.get(ROLLOUT_TOP_LOGPROBS)
    if top_ids is None or top_vals is None:
        problems.append("no top-K rows")
    else:
        top_ids, top_vals = numpy.asarray(top_ids), numpy.asarray(top_vals)
        if top_ids.shape != (n, scenario.top_k) or top_vals.shape != (n, scenario.top_k):
            problems.append(f"top-K shape {top_ids.shape} for {n} response tokens")
        else:
            on = mask == 1
            if not (top_ids[on, 0] == response[on]).all():
                problems.append("top-K first id is not the sampled token")
            if not numpy.allclose(top_vals[on, 0], lps[on], atol=1e-6):
                problems.append("top-K first log-prob is not the sampled token's")
            if not (top_ids[~on] == -1).all():
                problems.append("mask-0 top-K rows are not the -1 fill")
    routed = sample.get(ROLLOUT_ROUTED_EXPERTS)
    if routed is None:
        problems.append("no R3 rows")
    else:
        routed = numpy.asarray(routed)
        _, layers, top_k = scenario.r3
        if routed.shape != (len(tokens) - 1, layers, top_k):
            problems.append(f"R3 shape {routed.shape}; tokens {len(tokens)} -> {len(tokens) - 1} rows")
    return problems


# ============================================================================ runs
@dataclasses.dataclass
class TrialResult:
    group: int
    trial: int
    trajectory_id: str
    outcome: str
    reward: float
    samples: list[dict[str, Any]]
    records: Mapping[str, Any] | None
    env_s: float
    llm_s: float
    calls: int
    summary_calls: int
    exit: str | None = None  # why the agent stopped early (None: it answered or ran out of turns)


class SimRun:
    """One scenario through one path (module doc); :meth:`metrics` summarizes it."""

    def __init__(self, scenario: Scenario, *, path: str = "auto", workdir: Path | None = None,
                 miles_root: str | None = None) -> None:
        self.scenario = scenario
        self.caps = capabilities()
        if path == "auto":
            path = "agentic" if scenario.compaction is None or self.caps["sampler_segments"] else "records"
        if path not in ("agentic", "records"):
            raise ValueError(f"path must be agentic, records or auto, got {path!r}")
        self.path = path
        self.workdir = Path(workdir or tempfile.mkdtemp(prefix="sim-compaction-"))
        self.miles = miles_imports(miles_root)
        self.template = sim_template()
        self.engine = SimEngine(scenario, self.template)
        self.engine.start(None, 0)
        self.trials: list[TrialResult] = []
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.steps: list[dict[str, Any]] = []
        self.wall: dict[str, float] = {}
        self.direct_advantages: dict[str, float] = {}
        self.compaction_attempts: Counter[str] = Counter()
        self.backend: Any = None
        self.agent: SimAgent | None = None
        self.groups_written = 0

    # ------------------------------------------------------------------ shared
    def _proxy(self) -> Any:
        from yeto.rl.harness.gateway.proxy import RecordsProxy

        signed = {"max_tokens": self.scenario.turn_max_tokens, "temperature": 1.0, "top_p": 1.0}
        proxy = RecordsProxy(self.engine, self.template, model=SIGNED_MODEL, signed_sampling=signed,
                             sampler_id="sim-0", compaction=self.scenario.compaction_config(),
                             r3_response_len=self.scenario.max_response_len)
        backend, compact = proxy.backend, proxy.backend._compact

        async def counted(session: Any, *args: Any, **kwargs: Any) -> Any:
            # instrumentation only: a summary request that overflows the context is an
            # attempt the backend's ``compactions`` counter does not see
            self.compaction_attempts[session.trajectory_id] += 1
            return await compact(session, *args, **kwargs)

        backend._compact = counted
        self.backend = backend
        return proxy

    def _items(self) -> list[Any]:
        from yeto.rl.engine.engine_sampler import PromptItem

        return [PromptItem(f"task-{g}", f"simulated task {g}",
                           metadata={"agent_task": {"instance_id": f"task-{g}", "agent_name": "pi-sim"}})
                for g in range(self.scenario.tasks)]

    def _buffer_env(self) -> dict[str, str]:
        from yeto.rl.adapters.miles.buffer_rollout import BUFFER_ROOT_ENV

        return {BUFFER_ROOT_ENV: str(self.workdir / "store")}

    def _provenance(self) -> Provenance:
        sampling = SamplingParams(max_tokens=self.scenario.turn_max_tokens, n=self.scenario.n)
        return Provenance.of(self.engine.identity, sampler_id="sim-0", sampling=sampling, routed_experts=True)

    def _timing(self, trajectory_id: str) -> tuple[float, int, int]:
        calls = [c for c in self.engine.ledger if c.trajectory == trajectory_id]
        return (sum(c.llm_s(self.scenario) for c in calls), len(calls), sum(c.kind == "summary" for c in calls))

    # ------------------------------------------------------------------ paths
    def run(self) -> "SimRun":
        began, cpu = time.perf_counter(), time.process_time()
        if self.path == "agentic":
            self._run_agentic()
        else:
            self._run_records()
        self.wall["sampler_s"], self.wall["sampler_cpu_s"] = time.perf_counter() - began, time.process_time() - cpu
        began, cpu = time.perf_counter(), time.process_time()
        self._train_read()
        self.wall["trainer_s"], self.wall["trainer_cpu_s"] = time.perf_counter() - began, time.process_time() - cpu
        return self

    def _run_agentic(self) -> None:
        from yeto.rl.adapters.miles.buffer_rollout import open_buffer
        from yeto.rl.engine.agentic_round import AgenticRound
        from yeto.rl.engine.buffer_payload import GroupWriter
        from yeto.rl.harness.gateway.recorder import ProxyTokenRecorder

        s = self.scenario
        recorder = ProxyTokenRecorder(self._proxy())
        agent = self.agent = SimAgent(s)
        buffer = open_buffer(self._buffer_env(), sampler=True)
        items = self._items()
        sampling = SamplingParams(max_tokens=s.turn_max_tokens, n=s.n)
        rnd = AgenticRound(self.engine, GroupWriter(buffer, "sim-0"), recorder=recorder, harness=agent,
                           max_in_flight=s.tasks * s.n, max_response_len=s.max_response_len, min_samples=2,
                           task_defaults={"provider": "docker", "model_name": SIGNED_MODEL},
                           prompts=lambda rollout_id: items if rollout_id == 0 else [], sampling=sampling,
                           provenance=self._provenance(), emit=lambda name, **f: self.events.append((name, f)),
                           weight_switch="live" if self.engine.live else "drain", release_poll_s=0.0,
                           wait_s=0.005, infra_retries=0)
        try:
            rnd(0)
            rnd.flush()
        finally:
            with contextlib.suppress(Exception):
                rnd.loop.run(recorder.aclose())
            rnd.close()
        reader = open_buffer(self._buffer_env(), reader=True)
        reader.refresh()
        from yeto.rl.engine.buffer_payload import decode_group

        written: dict[str, list[dict[str, Any]]] = {}
        for selected in reader.select("sim-audit", 10_000):
            self.groups_written += 1
            for sample in decode_group(reader.read_payload(selected.entry.payload_ref)):
                written.setdefault(trial_key(sample), []).append(sample)
        for tid, info in agent.trials.items():
            samples = written.get(tid, [])
            llm_s, calls, summaries = self._timing(tid)
            outcome = "written" if samples else f"not written ({info['exit'] or 'lost'})"
            self.trials.append(TrialResult(int(str(info["task"]).rsplit("-", 1)[-1]), int(tid.rsplit("-", 1)[-1]),
                                           tid, outcome, float(samples[0]["reward"]) if samples else 0.0, samples,
                                           None, sum(info["env"].tool_s), llm_s, calls, summaries, info["exit"]))

    def _run_records(self) -> None:
        from yeto.rl.adapters.miles.buffer_rollout import open_buffer
        from yeto.rl.agentic import AdmissionGate, AgentTask, Outcome, run_agent_task
        from yeto.rl.agentic.outcome import policy_output_failure
        from yeto.rl.engine.buffer_payload import GroupWriter

        s = self.scenario
        proxy = self._proxy()
        agent = self.agent = SimAgent(s, inproc=InProcess(proxy))
        items = self._items()
        provenance = self._provenance().to_record()

        async def trial(g: int, j: int, gate: Any) -> tuple[int, int, Any, Mapping[str, Any]]:
            tid = f"sim-r0-task-{g}-{j}"
            proxy.open(tid)
            task = AgentTask.from_mapping({**items[g].metadata["agent_task"], "trajectory_id": tid},
                                          defaults={"provider": "docker", "model_name": SIGNED_MODEL})
            trajectory = await run_agent_task(task, agent, f"inproc://{tid}", gate)
            records = proxy.records(tid)
            proxy.close(tid)
            proxy.forget(tid)
            errors = [t.get("parse_error") for sess in records["sessions"] for seg in sess["samples"]
                      for t in seg["turns"] if t.get("parse_error")]
            return g, j, policy_output_failure(trajectory, errors), records

        async def everything() -> list[Any]:
            gate = AdmissionGate(s.tasks * s.n, policy_version=self.engine.served_version)
            return await asyncio.gather(*(trial(g, j, gate) for g in range(s.tasks) for j in range(s.n)))

        results = asyncio.run(everything())
        writer = GroupWriter(open_buffer(self._buffer_env(), sampler=True), "sim-0")
        cfg = s.compaction_config()
        for g in range(s.tasks):
            kept: list[dict[str, Any]] = []
            for gg, j, trajectory, records in results:
                if gg != g:
                    continue
                tid = trajectory.task.trajectory_id
                llm_s, calls, summaries = self._timing(tid)
                env_s = sum(agent.trials[tid]["env"].tool_s)
                samples: list[dict[str, Any]] = []
                if trajectory.outcome is not Outcome.ABORTED and records["sessions"]:
                    samples = records_segment_samples(items[g], trajectory, records, group_index=g,
                                                      index=g * s.n + j, cfg=cfg,
                                                      max_response_len=s.max_response_len,
                                                      provenance=provenance, live=self.engine.live)
                kept += samples
                self.trials.append(TrialResult(g, j, tid, trajectory.outcome.value, float(trajectory.reward),
                                               samples, records, env_s, llm_s, calls, summaries,
                                               agent.trials[tid]["exit"]))
            if len({trial_key(x) for x in kept}) >= 2:  # min_samples 2 (trials)
                writer.write(kept, group_key=f"r0-task-{g}", provenance=provenance)
                self.groups_written += 1
        # Miles' normalization straight on the contract samples (no buffer renumbering)
        flat = [x for t in self.trials for x in t.samples]
        if flat:
            advantages = trainer_advantages(self.miles, s, to_miles_samples(self.miles, flat))
            for sample, adv in zip(flat, advantages):
                key = f"{trial_key(sample)}#{(sample['metadata'].get('compaction') or {}).get('segment_index', 0)}"
                self.direct_advantages[key] = adv

    def _train_read(self) -> None:
        """The trainer island's reader: ``BatchReader.take`` -> ``miles_groups`` -> Miles
        ``Sample`` -> GRPO normalization, ``rollout_batch_size`` groups per step."""
        from yeto.rl.adapters.miles.buffer_rollout import BatchReader, miles_groups, open_buffer
        from yeto.rl.engine.buffer_payload import SampleBufferRolloutError

        s = self.scenario
        buffer = open_buffer(self._buffer_env(), batch_size=s.rollout_batch_size, reader=True)
        reader = BatchReader(buffer, max_policy_age=s.max_policy_age, wait_s=0.0, poll_s=0.0,
                             sleep=lambda _s: None, read_workers=1, prefetch=False, incarnation_gate=None,
                             group_size=s.n)
        for step in range(-(-self.groups_written // s.rollout_batch_size)):
            try:
                groups, report = reader.take(step, s.rollout_batch_size,
                                             current_version=self.engine.served_version())
            except SampleBufferRolloutError as exc:
                self.steps.append({"step": step, "error": f"{type(exc).__name__}: {str(exc)[:160]}"})
                break
            converted, problems = miles_groups(groups, samples_per_group=s.n, r3_num_experts=s.r3[0])
            dicts = [d for _, group in converted for d in group]
            samples = to_miles_samples(self.miles, dicts)
            advantages = trainer_advantages(self.miles, s, samples) if samples else []
            self.steps.append({
                "step": step, "groups": len(groups), "converted_groups": len(converted),
                "unconvertible": problems, "samples": len(dicts),
                "trials": len({trial_key(d) for d in dicts}),
                "stale_groups": report.stale_groups, "incomplete_groups": report.incomplete_groups,
                "dropped_small_groups": report.dropped_small_groups,
                "carried_over_groups": report.carried_over_groups, "padded_samples": report.padded_samples,
                "rows": [{"trial": trial_key(d), "group_index": d.get("group_index"), "index": d.get("index"),
                          "rollout_id": getattr(x, "rollout_id", None), "reward": float(d["reward"]),
                          "segment": (d["metadata"].get("compaction") or {}).get("segment_index", 0),
                          "advantage": a} for d, x, a in zip(dicts, samples, advantages)]})

    # ------------------------------------------------------------------ metrics
    def metrics(self) -> dict[str, Any]:
        s = self.scenario
        by_group: dict[int, list[TrialResult]] = {}
        for t in self.trials:
            by_group.setdefault(t.group, []).append(t)
        samples = [x for t in self.trials for x in t.samples]
        kept = [t for t in self.trials if t.samples]
        compactions = Counter(len(t.samples) - 1 for t in kept)
        comp = [x["metadata"]["compaction"] for x in samples if "compaction" in x["metadata"]]
        summary = [c["summary_ok"] for c in comp if c["summary_ok"] is not None]
        trained = sum(int(sum(x["loss_mask"])) for x in samples)
        response = sum(int(x["response_length"]) for x in samples)
        total = sum(len(x["tokens"]) for x in samples)
        alignment = [p for x in samples for p in check_sample_alignment(x, s)]
        env = [t.env_s for t in self.trials]
        both = [t.env_s + s.llm_weight * t.llm_s for t in self.trials]
        by_segment: dict[int, list[int]] = {}
        for x in samples:
            by_segment.setdefault(int((x["metadata"].get("compaction") or {}).get("segment_index", 0)), []).append(
                len(x["tokens"]))
        calls = list(self.engine.ledger)
        engine_cpu = sum(c.engine_cpu_s for c in calls)
        sampler_cpu = self.wall.get("sampler_cpu_s", 0.0)
        return {
            "scenario": s.name, "description": s.description, "path": self.path, "capabilities": self.caps,
            "normalization": self.miles.source,
            "compaction": None if s.compaction is None else dict(s.compaction),
            "tasks": s.tasks, "n": s.n,
            "trials_per_task": {g: len(ts) for g, ts in sorted(by_group.items())},
            "kept_trials_per_task": {g: sum(1 for t in ts if t.samples) for g, ts in sorted(by_group.items())},
            "outcomes": dict(Counter(t.outcome for t in self.trials)),
            "reward_mean": round(statistics.fmean(t.reward for t in kept), 3) if kept else None,
            "segments_per_trial": dict(sorted(Counter(len(t.samples) for t in kept).items())),
            "compactions_per_trial": dict(sorted(compactions.items())),
            "compaction_attempts": sum(self.compaction_attempts.values()),
            "compactions": sum(k * v for k, v in compactions.items()),
            "context_full_exits": sum(1 for t in self.trials if "max_model_len" in str(t.exit or "")),
            "summary_ok_rate": round(sum(summary) / len(summary), 3) if summary else None,
            "kept_steps": dict(sorted(Counter(c["kept_steps"] for c in comp if c["kept_steps"] is not None).items())),
            "samples": len(samples), "trainable_tokens": trained,
            "trainable_of_response": round(trained / response, 3) if response else None,
            "trainable_of_tokens": round(trained / total, 3) if total else None,
            "segment_tokens": _stats([len(x["tokens"]) for x in samples]),
            "segment_tokens_by_index": {k: _stats(v) for k, v in sorted(by_segment.items())},
            "status": dict(Counter(x["status"] for x in samples)),
            "samples_with_compaction_meta": len(comp),
            "versions": sorted({x["metadata"]["weight_version"] for x in samples}),
            "multi_version_samples": sum(x["metadata"].get("final_weight_version", x["metadata"]["weight_version"])
                                         != x["metadata"]["weight_version"] for x in samples),
            "engine_reloads": list(self.engine.reloads),
            "alignment_problems": alignment[:10], "alignment_problem_count": len(alignment),
            "time_env_s": _stats(env), "time_env_plus_llm_s": _stats(both),
            "time_llm_s": _stats([t.llm_s for t in self.trials]),
            "group_time": {g: {"env_s": round(max(t.env_s for t in ts), 2),
                               "env_plus_llm_s": round(max(t.env_s + s.llm_weight * t.llm_s for t in ts), 2)}
                           for g, ts in sorted(by_group.items())},
            "round_time_env_s": round(max(env), 2) if env else None,
            "round_time_env_plus_llm_s": round(max(both), 2) if both else None,
            "engine_calls": len(calls), "summary_calls": sum(c.kind == "summary" for c in calls),
            "wall_sampler_s": round(self.wall.get("sampler_s", 0.0), 3),
            "cpu_sampler_s": round(sampler_cpu, 3), "cpu_fake_engine_s": round(engine_cpu, 3),
            "cpu_yeto_per_call_ms": round(1000 * (sampler_cpu - engine_cpu) / max(1, len(calls)), 2),
            "wall_trainer_s": round(self.wall.get("trainer_s", 0.0), 3),
            "groups_written": self.groups_written,
            "trainer_steps": [{k: v for k, v in step.items() if k != "rows"} for step in self.steps],
            "advantage_check": self.advantage_check(),
        }

    def advantage_check(self) -> dict[str, Any]:
        """Per trainer step: every trial's segments share one advantage, and it equals the
        GRPO normalization of the trial-level rewards of its group (computed here
        independently with ``statistics``)."""
        out = {"steps": [], "direct": None}
        for step in self.steps:
            rows = step.get("rows") or []
            out["steps"].append(_advantage_verdict(rows))
        if self.direct_advantages:
            rows = []
            for t in self.trials:
                for x in t.samples:
                    seg = (x["metadata"].get("compaction") or {}).get("segment_index", 0)
                    rows.append({"trial": t.trajectory_id, "group_index": x["group_index"], "reward": float(x["reward"]),
                                 "segment": seg, "advantage": self.direct_advantages[f"{t.trajectory_id}#{seg}"]})
            out["direct"] = _advantage_verdict(rows)
        return out


def _advantage_verdict(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_trial: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_trial.setdefault(str(row["trial"]), []).append(row)
    same = all(len({r["advantage"] for r in rs}) == 1 and len({r["reward"] for r in rs}) == 1
               for rs in by_trial.values())
    expected = {}
    for g in {r["group_index"] for r in rows}:
        trials = sorted({str(r["trial"]) for r in rows if r["group_index"] == g})
        rewards = [by_trial[t][0]["reward"] for t in trials]
        for t, a in zip(trials, grpo_reference(rewards)):
            expected[t] = a
    worst = max((abs(r["advantage"] - expected[str(r["trial"])]) for r in rows), default=0.0)
    return {"samples": len(rows), "trials": len(by_trial), "segments_share_advantage": same,
            "max_error_vs_trial_grpo": worst, "matches_trial_grpo": worst < 1e-5}


def grpo_reference(rewards: Sequence[float]) -> list[float]:
    """GRPO over trial-level rewards: ``(r - mean) / (std + 1e-6)`` (unbiased std; no
    division when the std is 0 or there is one trial)."""
    mean = statistics.fmean(rewards)
    centered = [r - mean for r in rewards]
    if len(rewards) > 1:
        std = statistics.stdev(rewards)
        if std > 0:
            centered = [c / (std + 1e-6) for c in centered]
    return centered


def _stats(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {"min": round(ordered[0], 2), "p50": round(statistics.median(ordered), 2),
            "mean": round(statistics.fmean(ordered), 2), "max": round(ordered[-1], 2)}


# ============================================================================ report
def _check(checks: Sequence[Mapping[str, Any] | None]) -> str:
    checks = [c for c in checks if c and c["samples"]]
    if not checks:
        return "-"
    share = all(c["segments_share_advantage"] for c in checks)
    match = all(c["matches_trial_grpo"] for c in checks)
    return f"{'yes' if share else 'NO'}/{'yes' if match else 'NO'}"


def markdown(all_metrics: Sequence[Mapping[str, Any]]) -> str:
    """Two tables: the sampler side and the trainer side of every scenario."""
    head = ["scenario", "path", "trials/task (kept)", "outcomes", "segments/trial", "compactions/trial",
            "summary_ok", "trainable/resp", "segment tokens p50 (max)", "env s p50/max",
            "env+0.1*llm s p50/max", "cpu ms/call"]
    rows = [head, ["---"] * len(head)]
    for m in all_metrics:
        seg = m["segment_tokens"] or {}
        kept = sorted(set(m["kept_trials_per_task"].values()))
        rows.append([m["scenario"], m["path"], f"{sorted(set(m['trials_per_task'].values()))} ({kept})",
                     json.dumps(m["outcomes"]), json.dumps(m["segments_per_trial"]),
                     json.dumps(m["compactions_per_trial"]), str(m["summary_ok_rate"]),
                     str(m["trainable_of_response"]), f"{seg.get('p50')} ({seg.get('max')})",
                     f"{m['time_env_s']['p50']}/{m['time_env_s']['max']}",
                     f"{m['time_env_plus_llm_s']['p50']}/{m['time_env_plus_llm_s']['max']}",
                     str(m["cpu_yeto_per_call_ms"])])
    head2 = ["scenario", "groups written", "trainer steps: samples/trials", "trainer: same adv/trial GRPO",
             "direct Miles norm: same adv/trial GRPO", "alignment problems", "status", "versions"]
    rows2 = [head2, ["---"] * len(head2)]
    for m in all_metrics:
        steps = ", ".join("error" if "error" in x else f"{x['samples']}/{x['trials']}" for x in m["trainer_steps"])
        rows2.append([m["scenario"], str(m["groups_written"]), steps or "-", _check(m["advantage_check"]["steps"]),
                      _check([m["advantage_check"]["direct"]]), str(m["alignment_problem_count"]),
                      json.dumps(m["status"]), json.dumps(m["versions"])])
    return "\n".join("| " + " | ".join(r) + " |" for r in rows) + "\n\n" + \
        "\n".join("| " + " | ".join(r) + " |" for r in rows2)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--scenario", default="all", help=f"one of {sorted(SCENARIOS)} or all")
    parser.add_argument("--path", default="auto", choices=("auto", "agentic", "records"))
    parser.add_argument("--json", help="write every scenario's metrics here")
    parser.add_argument("--miles-root", help="a Miles checkout (adds it to sys.path)")
    parser.add_argument("--workdir", help="buffer directory (default: a fresh temporary one per scenario)")
    args = parser.parse_args(argv)
    names = sorted(SCENARIOS) if args.scenario == "all" else [args.scenario]
    out = []
    for name in names:
        workdir = Path(args.workdir) / name if args.workdir else None
        began = time.perf_counter()
        metrics = SimRun(SCENARIOS[name], path=args.path, workdir=workdir, miles_root=args.miles_root).run().metrics()
        metrics["wall_total_s"] = round(time.perf_counter() - began, 3)
        out.append(metrics)
        print(f"[sim] {name}: path={metrics['path']} samples={metrics['samples']} "
              f"wall={metrics['wall_total_s']}s", file=sys.stderr, flush=True)
    print(markdown(out))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

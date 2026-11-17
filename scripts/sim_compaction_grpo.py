"""CPU end-to-end simulation of async GRPO with context compaction (CompactionRL segments).

Real pieces, end to end:

* the spec's ``agentic.compaction`` -> ``agentic_round.spec_compaction`` (``config_from_spec``
  and its launch checks) -> :class:`~yeto.rl.compaction.CompactionConfig`;
* the records proxy as ``round_factory`` builds it (``RecordsProxy(compaction=...,
  r3_response_len=max_response_len)``) behind ``ProxyTokenRecorder(allow_compaction=True)``
  on 127.0.0.1 HTTP, the gateway's Chat route and ``RecordingBackend``'s compaction;
* ``AgenticRound`` (``run_agent_task``, ``finish_sample(segmented=True)``,
  ``agentic_samples``: one buffer sample per compaction segment) writing through
  ``GroupWriter`` into a local-disk sample buffer (``open_buffer`` over
  ``YETO_SAMPLE_BUFFER_ROOT``);
* the trainer's round: ``BatchReader(group_size=n)`` and ``_BufferRound.__call__`` of
  ``buffer_rollout_miles`` (``miles_groups`` / ``renumber_group`` by trial,
  ``check_segmented_round``, padding by trial, the ``compaction_batch`` log line);
  only the Miles rollout base classes that module imports are stand-ins (as in
  ``tests/test_rl_buffer_rollout.py``), and Miles' own ``Sample`` is used when Miles
  is importable;
* Miles' GRPO normalization (``reward_group_segments`` / ``rollout_segments`` from
  Miles when importable; ``_normalize_rewards_by_rollout`` needs Miles' train-data
  imports, so a line-for-line port runs, see :func:`normalize_rewards_by_rollout`).

Fake pieces: a word-level hashing tokenizer with a ChatML-like template (Qwen3 tool
calls, ``parse_qwen3``); :class:`SimEngine` (a :class:`FakeEngine` with a scripted
policy, log-probs, top-K rows, small R3 rows and in-place reloads); :class:`SimAgent`
(pi-like: the whole history to the Chat route every turn, every tool call run in
:class:`SimEnvironment`, the 0/1 verifier reward); the environment's tool output
lengths and tool times come from configured distributions, and a trial may fail
(ABORTED) after some calls.

Time is simulated (nothing sleeps): an LLM call costs ``call_overhead_s +
prefill_s_per_token * uncached prompt tokens + decode_s_per_token * completion
tokens`` (the proxy's summary turns included); a tool call costs a draw from
``tool_time_s``. Two totals per trial: ``env_s`` (tool time only) and ``env_s + 0.1
* llm_s``. The real CPU per LLM call (the process minus the fake engine's thread
time) is yeto's own overhead.

Run: ``python scripts/sim_compaction_grpo.py --scenario all`` (``--json out.json``).
Add the Miles checkout to ``PYTHONPATH`` (or ``--miles-root``) to use Miles' own
``Sample`` and segment helpers.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import importlib.util
import io
import json
import math
import random
import re
import statistics
import sys
import tempfile
import threading
import time
import types
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
AGENT = "pi"  # the agent whose own compaction yeto turns off (pi_config.PROXY_COMPACTION_AGENTS)
BATCH_LINE = re.compile(r"compaction_batch: rollout=(\d+) trials=(\d+) segments=(\d+) per_trial=\{([^}]*)\}")


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
    spec_checked: bool = True  # False: build CompactionConfig directly (a config the spec refuses)
    max_model_len: int = 4096
    turn_max_tokens: int = 192
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

    def spec(self) -> dict[str, Any]:
        """The sampler spec's ``agentic`` section the round would be built from."""
        return {"sampler_id": "sim-0", "sampling": {"temperature": 1.0, "top_p": 1.0},
                "agentic": {"compaction": None if self.compaction is None else dict(self.compaction),
                            "turn_max_tokens": self.turn_max_tokens, "max_response_len": self.max_response_len,
                            "task_defaults": {"agent": AGENT}}}

    def spec_error(self) -> str | None:
        """Why the spec's checks refuse this scenario's compaction (None: accepted)."""
        from yeto.rl.compaction import config_from_spec

        try:
            config_from_spec(self.spec(), max_model_len=self.max_model_len)
        except ValueError as exc:
            return str(exc)
        return None

    def compaction_config(self, engine: Any = None) -> CompactionConfig | None:
        """The proxy's config as ``round_factory`` gets it (``agentic_round.spec_compaction``);
        ``spec_checked=False``: the dataclass itself (the launch checks skipped)."""
        if self.compaction is None:
            return None
        if not self.spec_checked:
            return CompactionConfig(**dict(self.compaction))
        from yeto.rl.engine.agentic_round import spec_compaction

        return spec_compaction(self.spec(), engine)

    def success(self, task: int) -> float:
        if isinstance(self.success_p, Mapping):
            return float(self.success_p.get(task, 0.5))
        return float(self.success_p)


# 4096 / 1024: t_comp holds the summary turn (summary_max_tokens 128 + the summary request's
# byte bound, 632) and the budget leaves the engine room for it after a full step
# (config_from_spec: check_summary_room / check_summary_headroom with turn_max_tokens 192).
# A t_comp of 512 is refused by the spec (``tcomp_overflow`` runs it with the checks off).
SMALL = {"context_budget": 4096, "t_comp": 1024, "keep_recent_steps": 2, "max_compactions": 3,
         "summary_max_tokens": 128}
SHORT_STEP = {"action_tokens": Dist("int", 20, 60), "tool_tokens": Dist("int", 60, 110),
              "summary_tokens": Dist("int", 30, 70)}
MIXED = {0: 0.5, 1: 0.25, 2: 0.75, 3: 0.5, 4: 0.5, 5: 0.25}
SCENARIOS: dict[str, Scenario] = {s.name: s for s in (
    Scenario("no_compaction", "compaction off, short trials (main format)", compaction=None, max_model_len=16384,
             max_response_len=16384, steps=Dist("int", 2, 5), success_p=MIXED),
    Scenario("no_trigger", "compaction on (4096 / 1024), trials too short to trigger it", compaction=SMALL,
             steps=Dist("int", 3, 8), success_p=MIXED, **SHORT_STEP),
    Scenario("compact_once", "compaction on, about one compaction per trial", compaction=SMALL,
             steps=Dist("int", 18, 24), success_p=MIXED, **SHORT_STEP),
    Scenario("compact3_truncated", "trials that never finish: 3 compactions, then the context fills (truncated)",
             compaction=SMALL, steps=Dist("int", 500, 500), success_p=1.0, **SHORT_STEP),
    Scenario("mixed_reload", "short to long trials, two in-place weight reloads during the round (live)",
             compaction=SMALL, steps=Dist("int", 3, 34), reload_after_calls=(250, 500), success_p=MIXED,
             **SHORT_STEP),
    Scenario("partial_abort", "a quarter of the trials fail after 1-3 calls (ABORTED, left out of the group)",
             compaction=SMALL, steps=Dist("int", 8, 24), abort_p=0.25, success_p=MIXED, **SHORT_STEP),
    Scenario("tcomp_overflow", "t_comp 512 (the spec refuses it; run with the checks off): steps of up to ~700 "
             "tokens, the summary request does not always fit", spec_checked=False,
             compaction={"context_budget": 4096, "t_comp": 512, "keep_recent_steps": 2, "max_compactions": 3},
             turn_max_tokens=256, steps=Dist("int", 9, 12), tool_tokens=Dist("int", 300, 450), success_p=MIXED),
    Scenario("r3_segment_window", "R3 rows cut to each segment's own training window: max_response_len 3250 "
             "< budget (the spec refuses it; checks off), one compaction, the last segment cut",
             spec_checked=False, compaction={**SMALL, "max_compactions": 1}, max_response_len=3250,
             steps=Dist("int", 500, 500), success_p=1.0, **SHORT_STEP),
    Scenario("batch_count", "6 tasks, 3 groups per trainer step, aborted trials: the global batch counts trials "
             "(rollout_batch_size x n distinct rollouts per step, padding by trial)", tasks=6,
             rollout_batch_size=3, compaction=SMALL, steps=Dist("int", 8, 24), abort_p=0.2, success_p=MIXED,
             **SHORT_STEP),
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


class SimAgent:
    """pi-like agent (``AgentHarness``): every turn it sends the whole history and
    the tool schemas to ``<endpoint>/chat/completions``; it runs every tool call and
    answers when the model replies without one. A refused call (the context is
    full) ends the agent with a non-zero exit, as pi does."""

    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario
        self.trials: dict[str, dict[str, Any]] = {}

    async def run(self, task: Any, endpoint: str) -> Any:
        import aiohttp

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
        async with aiohttp.ClientSession() as http:
            while info["calls"] < self.scenario.max_agent_turns:
                began = time.perf_counter()
                body = {"model": SIGNED_MODEL, "messages": messages, "tools": TOOLS}
                async with http.post(f"{endpoint}/chat/completions", json=body) as response:
                    status, reply = response.status, await response.json()
                info["call_wall_s"].append(time.perf_counter() - began)
                info["statuses"][status] += 1
                if status != 200:
                    info["exit"] = f"{status}: {str((reply.get('error') or {}).get('message'))[:160]}"
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
        return TrialVerdict(rewards={"reward": env.reward(info["answered"])}, timings_s={"env_s": sum(env.tool_s)},
                            agent_metrics={"tool_calls": env.done, "steps_needed": env.steps})


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


# ============================================================================ trainer round
def trial_key(sample: Any) -> str:
    """The trial a buffer sample (a dict or a Miles ``Sample``) belongs to:
    ``metadata.compaction.trajectory_id``, else ``metadata.trajectory_id``."""
    meta = (sample.get("metadata") if isinstance(sample, Mapping) else getattr(sample, "metadata", None)) or {}
    comp = meta.get("compaction")
    return str(comp["trajectory_id"]) if isinstance(comp, Mapping) else str(meta.get("trajectory_id"))


def _field(sample: Any, name: str) -> Any:
    return sample.get(name) if isinstance(sample, Mapping) else getattr(sample, name, None)


def buffer_round_module() -> types.ModuleType:
    """``yeto.rl.adapters.miles.buffer_rollout_miles`` loaded with stand-ins for the Miles
    rollout base classes it imports (as ``tests/test_rl_buffer_rollout.py`` does), under a
    private module name: the sys.modules entries of those Miles modules are restored, so a
    real Miles on the path is not shadowed afterwards."""
    from yeto.rl.adapters.miles import buffer_rollout

    class _Base:
        def __init__(self, input: Any) -> None:
            self.constructor_input = input

    stubs = {
        "miles.rollout.base_types": dict(BaseRolloutFn=_Base, RolloutFnConstructorInput=object,
                                         RolloutFnInput=object, RolloutFnTrainOutput=lambda **kw: kw),
        "miles.rollout.inference_rollout.inference_rollout_common": dict(InferenceRolloutFn=_Base),
        "miles.utils.function_registry": dict(load_function=lambda path: None),
        "miles.utils.types": dict(Sample=object),
    }
    parents = {"miles", "miles.rollout", "miles.rollout.inference_rollout", "miles.utils"}
    names = set(stubs) | parents
    saved = {name: sys.modules.get(name) for name in names}
    name = "yeto.rl.adapters.miles._sim_buffer_rollout_miles"
    try:
        for parent in parents:
            if parent not in sys.modules:
                sys.modules[parent] = types.ModuleType(parent)
        for module, attrs in stubs.items():
            sys.modules[module] = SimpleNamespace(**attrs)  # type: ignore[assignment]
        spec = importlib.util.spec_from_file_location(
            name, Path(buffer_rollout.__file__).with_name("buffer_rollout_miles.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
    return module


def trainer_round(plugin: types.ModuleType, reader: Any, scenario: Scenario, rollout_id: int, *, version: int,
                  sample_cls: Any) -> tuple[Any, list[dict[str, Any]], str]:
    """One trainer round: the real ``_BufferRound.__call__`` over ``reader`` (a real
    ``BatchReader``). Returns its output, the ``record_round_metadata`` counts, and what
    it printed (the ``compaction_batch`` line)."""
    counts: list[dict[str, Any]] = []
    plugin.record_round_metadata = lambda args, rid, **c: counts.append(dict(c))
    plugin.Sample = sample_cls if sample_cls is not None else SimpleNamespace(from_dict=lambda d: d)
    budget = (scenario.compaction or {}).get("context_budget", scenario.max_model_len)
    round_ = plugin._BufferRound.__new__(plugin._BufferRound)
    round_.args = SimpleNamespace(rollout_sample_filter_path=None, rollout_all_samples_process_path=None,
                                  use_dynamic_batch_size=True, seq_length=int(budget), grpo_std_normalization=True,
                                  hf_checkpoint=None, rollout_batch_size=scenario.rollout_batch_size,
                                  n_samples_per_prompt=scenario.n)
    round_.groups, round_.samples_per_group, round_.reader = scenario.rollout_batch_size, scenario.n, reader
    round_._r3_layers_cached = (None, scenario.r3[0])
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        out = round_(SimpleNamespace(evaluation=False, rollout_id=int(rollout_id), weight_version=int(version)))
    return out, counts, printed.getvalue()


def parse_batch_line(text: str) -> dict[str, Any] | None:
    """The ``compaction_batch`` line of a trainer round (None: not printed)."""
    m = BATCH_LINE.search(text)
    if m is None:
        return None
    per = {int(k): int(v) for k, v in (kv.split(":") for kv in m.group(4).split(",") if kv)}
    return {"rollout": int(m.group(1)), "trials": int(m.group(2)), "segments": int(m.group(3)), "per_trial": per}


# ============================================================================ runs
@dataclasses.dataclass
class TrialResult:
    group: int
    trial: int
    trajectory_id: str
    outcome: str
    reward: float
    samples: list[dict[str, Any]]
    env_s: float
    llm_s: float
    calls: int
    summary_calls: int
    exit: str | None = None  # why the agent stopped early (None: it answered or ran out of turns)
    counters: Mapping[str, int] = dataclasses.field(default_factory=dict)  # the proxy's, at close


class SimRun:
    """One scenario: sampler island (AgenticRound) -> buffer -> trainer rounds;
    :meth:`metrics` summarizes it."""

    def __init__(self, scenario: Scenario, *, workdir: Path | None = None, miles_root: str | None = None) -> None:
        self.scenario = scenario
        self.workdir = Path(workdir or tempfile.mkdtemp(prefix="sim-compaction-"))
        self.miles = miles_imports(miles_root)
        self.template = sim_template()
        self.engine = SimEngine(scenario, self.template)
        self.engine.start(None, 0)
        self.cfg = scenario.compaction_config(self.engine)
        self.trials: list[TrialResult] = []
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.steps: list[dict[str, Any]] = []
        self.wall: dict[str, float] = {}
        self.compaction_attempts: Counter[str] = Counter()
        self.closed_counters: dict[str, dict[str, int]] = {}
        self.r3_cut_turns: dict[str, int] = {}  # turns whose R3 rows the proxy cut to the training window
        self.agent: SimAgent | None = None
        self.groups_written = 0

    def _proxy(self) -> Any:
        """The records proxy as ``round_factory`` builds it, plus two instruments: the
        summary attempts (``_compact`` calls; one that finds no room is not a
        compaction) and each trajectory's counters when its records are read."""
        from yeto.rl.harness.gateway.proxy import RecordsProxy

        signed = {"max_tokens": self.scenario.turn_max_tokens, "temperature": 1.0, "top_p": 1.0}
        proxy = RecordsProxy(self.engine, self.template, model=SIGNED_MODEL, signed_sampling=signed,
                             sampler_id="sim-0", compaction=self.cfg, r3_response_len=self.scenario.max_response_len)
        backend, compact, records = proxy.backend, proxy.backend._compact, proxy.records

        async def counted(session: Any, *args: Any, **kwargs: Any) -> Any:
            self.compaction_attempts[session.trajectory_id] += 1
            return await compact(session, *args, **kwargs)

        def recorded(trajectory_id: str) -> dict[str, Any]:
            out = records(trajectory_id)
            self.closed_counters[trajectory_id] = dict(out.get("counters") or {})
            self.r3_cut_turns[trajectory_id] = sum(bool(turn.get("routed_experts_cut")) for session in out["sessions"]
                                                   for seg in session["samples"] for turn in seg["turns"])
            return out

        backend._compact, proxy.records = counted, recorded
        return proxy

    def _items(self) -> list[Any]:
        from yeto.rl.engine.engine_sampler import PromptItem

        return [PromptItem(f"task-{g}", f"simulated task {g}",
                           metadata={"agent_task": {"instance_id": f"task-{g}", "agent_name": AGENT}})
                for g in range(self.scenario.tasks)]

    def _buffer_env(self) -> dict[str, str]:
        from yeto.rl.adapters.miles.buffer_rollout import BUFFER_ROOT_ENV

        return {BUFFER_ROOT_ENV: str(self.workdir / "store")}

    def _timing(self, trajectory_id: str) -> tuple[float, int, int]:
        calls = [c for c in self.engine.ledger if c.trajectory == trajectory_id]
        return (sum(c.llm_s(self.scenario) for c in calls), len(calls), sum(c.kind == "summary" for c in calls))

    def run(self) -> "SimRun":
        began, cpu = time.perf_counter(), time.process_time()
        self._sample()
        self.wall["sampler_s"], self.wall["sampler_cpu_s"] = time.perf_counter() - began, time.process_time() - cpu
        began, cpu = time.perf_counter(), time.process_time()
        self._train()
        self.wall["trainer_s"], self.wall["trainer_cpu_s"] = time.perf_counter() - began, time.process_time() - cpu
        return self

    def _sample(self) -> None:
        """The sampler island's round: ``AgenticRound`` over ``ProxyTokenRecorder``."""
        from yeto.rl.adapters.miles.buffer_rollout import open_buffer
        from yeto.rl.engine.agentic_round import AgenticRound
        from yeto.rl.engine.buffer_payload import GroupWriter, decode_group
        from yeto.rl.harness.gateway.recorder import ProxyTokenRecorder

        s = self.scenario
        recorder = ProxyTokenRecorder(self._proxy(), allow_compaction=True)  # as round_factory
        agent = self.agent = SimAgent(s)
        items = self._items()
        sampling = SamplingParams(max_tokens=s.turn_max_tokens, n=s.n)
        provenance = Provenance.of(self.engine.identity, sampler_id="sim-0", sampling=sampling, routed_experts=True)
        buffer = open_buffer(self._buffer_env(), sampler=True)
        rnd = AgenticRound(self.engine, GroupWriter(buffer, "sim-0"),
                           recorder=recorder, harness=agent, max_in_flight=s.tasks * s.n,
                           max_response_len=s.max_response_len, min_samples=2,
                           task_defaults={"provider": "docker", "model_name": SIGNED_MODEL},
                           prompts=lambda rollout_id: items if rollout_id == 0 else [], sampling=sampling,
                           provenance=provenance, emit=lambda name, **f: self.events.append((name, f)),
                           weight_switch="live" if self.engine.live else "drain", release_poll_s=0.0,
                           wait_s=0.005, infra_retries=0)
        try:
            rnd(0)
            rnd.flush()
        finally:
            with contextlib.suppress(Exception):
                rnd.loop.run(recorder.aclose())
            rnd.close()
        buffer.mark_data_end()  # as the sampler loop after its last round: the trainer takes what is left
        reader = open_buffer(self._buffer_env(), reader=True)
        reader.refresh()
        written: dict[str, list[dict[str, Any]]] = {}
        for selected in reader.select("sim-audit", 10_000):
            self.groups_written += 1
            for sample in decode_group(reader.read_payload(selected.entry.payload_ref)):
                written.setdefault(trial_key(sample), []).append(sample)
        for tid, info in agent.trials.items():
            samples = written.get(tid, [])
            llm_s, calls, summaries = self._timing(tid)
            outcome = "written" if samples else f"not written ({(info['exit'] or 'lost').split(':', 1)[0]})"
            self.trials.append(TrialResult(int(str(info["task"]).rsplit("-", 1)[-1]), int(tid.rsplit("-", 1)[-1]),
                                           tid, outcome, float(samples[0]["reward"]) if samples else 0.0, samples,
                                           sum(info["env"].tool_s), llm_s, calls, summaries, info["exit"],
                                           self.closed_counters.get(tid, {})))

    def _train(self) -> None:
        """The trainer island's rounds: ``_BufferRound`` (real) over a ``BatchReader``
        (real) with ``rollout_batch_size`` groups of ``n`` trials, then Miles' GRPO
        normalization of the round's samples."""
        from yeto.rl.adapters.miles.buffer_rollout import BatchReader, is_padding, open_buffer
        from yeto.rl.engine.buffer_payload import SampleBufferRolloutError

        s = self.scenario
        plugin = buffer_round_module()
        buffer = open_buffer(self._buffer_env(), batch_size=s.rollout_batch_size, reader=True)
        reader = BatchReader(buffer, max_policy_age=s.max_policy_age, wait_s=0.0, poll_s=0.0,
                             sleep=lambda _s: None, read_workers=1, prefetch=False, incarnation_gate=None,
                             group_size=s.n)
        for step in range(-(-self.groups_written // s.rollout_batch_size)):
            try:
                out, counts, printed = trainer_round(plugin, reader, s, step, version=int(self.engine.served_version()),
                                                     sample_cls=self.miles.Sample)
            except SampleBufferRolloutError as exc:
                self.steps.append({"step": step, "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
                break
            flat = [x for group in out["samples"] for x in group]
            real = [x for x in flat if not is_padding(x)]
            samples = flat if self.miles.Sample is not None else to_miles_samples(self.miles, flat)
            advantages = trainer_advantages(self.miles, s, samples)
            rollouts = {(_field(x, "rollout_id") if _field(x, "rollout_id") is not None else ("index", _field(x, "index")))
                        for x in flat}
            meta = counts[0] if counts else {}
            self.steps.append({
                "step": step, "samples": len(real), "trials": len({trial_key(x) for x in real}),
                "padded_samples": len(flat) - len(real), "distinct_rollouts": len(rollouts),
                "expected_rollouts": s.rollout_batch_size * s.n,
                "compaction_batch": parse_batch_line(printed),
                "round_metadata": {k: v for k, v in meta.items() if k.startswith("buffer_")},
                "rows": [{"trial": trial_key(x), "group_index": _field(x, "group_index"), "index": _field(x, "index"),
                          "rollout_id": _field(x, "rollout_id"), "reward": float(_field(x, "reward")),
                          "segment": ((_field(x, "metadata") or {}).get("compaction") or {}).get("segment_index", 0),
                          "advantage": a}
                         for x, a in zip(flat, advantages) if not is_padding(x)]})

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
        counters: Counter[str] = Counter()
        for t in self.trials:
            counters.update(t.counters)
        calls = list(self.engine.ledger)
        engine_cpu = sum(c.engine_cpu_s for c in calls)
        sampler_cpu = self.wall.get("sampler_cpu_s", 0.0)
        rounds = [f for n, f in self.events if n == "rl_sampler_engine_round"]
        lost: Counter[str] = Counter()
        for f in rounds:
            lost.update((f.get("agentic") or {}).get("lost") or {})
        return {
            "scenario": s.name, "description": s.description, "normalization": self.miles.source,
            "compaction": None if s.compaction is None else dict(s.compaction),
            "spec_checked": s.spec_checked, "spec_error": None if s.compaction is None else s.spec_error(),
            "tasks": s.tasks, "n": s.n,
            "trials_per_task": {g: len(ts) for g, ts in sorted(by_group.items())},
            "kept_trials_per_task": {g: sum(1 for t in ts if t.samples) for g, ts in sorted(by_group.items())},
            "outcomes": dict(Counter(t.outcome for t in self.trials)), "lost": dict(lost),
            "compaction_no_room": sum(int((f.get("agentic") or {}).get("compaction_no_room", 0)) for f in rounds),
            "reward_mean": round(statistics.fmean(t.reward for t in kept), 3) if kept else None,
            "segments_per_trial": dict(sorted(Counter(len(t.samples) for t in kept).items())),
            "compactions_per_trial": dict(sorted(compactions.items())),
            "compaction_attempts": sum(self.compaction_attempts.values()),
            "compactions": int(counters.get("compactions", 0)),
            "proxy_counters": dict(sorted(counters.items())),
            "context_full_exits": sum(1 for t in self.trials if "max_model_len" in str(t.exit or "")
                                      or "context_budget" in str(t.exit or "") or "no_room" in str(t.exit or "")),
            "summary_ok_rate": round(sum(summary) / len(summary), 3) if summary else None,
            "kept_steps": dict(sorted(Counter(c["kept_steps"] for c in comp if c["kept_steps"] is not None).items())),
            "samples": len(samples), "trainable_tokens": trained,
            "trainable_of_response": round(trained / response, 3) if response else None,
            "trainable_of_tokens": round(trained / total, 3) if total else None,
            "segment_tokens": _stats([len(x["tokens"]) for x in samples]),
            "segment_tokens_by_index": {k: _stats(v) for k, v in sorted(by_segment.items())},
            "r3_cut_turns": sum(self.r3_cut_turns.values()),
            "response_at_max_response_len": sum(int(x["response_length"]) == s.max_response_len for x in samples),
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
            "advantage_check": [_advantage_verdict(step.get("rows") or []) for step in self.steps],
        }


def _advantage_verdict(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Every trial's segments share one advantage and reward, and the advantage equals the
    GRPO normalization of the trial-level rewards of its group (:func:`grpo_reference`)."""
    by_trial: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_trial.setdefault(str(row["trial"]), []).append(row)
    same = all(len({r["advantage"] for r in rs}) == 1 and len({r["reward"] for r in rs}) == 1
               for rs in by_trial.values())
    one_rollout = all(len({(r["rollout_id"], r["index"]) for r in rs}) == 1 for rs in by_trial.values())
    expected = {}
    for g in {r["group_index"] for r in rows}:
        trials = sorted({str(r["trial"]) for r in rows if r["group_index"] == g})
        rewards = [by_trial[t][0]["reward"] for t in trials]
        for t, a in zip(trials, grpo_reference(rewards)):
            expected[t] = a
    worst = max((abs(r["advantage"] - expected[str(r["trial"])]) for r in rows), default=0.0)
    return {"samples": len(rows), "trials": len(by_trial), "segments_share_advantage": same,
            "one_rollout_per_trial": one_rollout, "max_error_vs_trial_grpo": worst,
            "matches_trial_grpo": worst < 1e-5}


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
def _check(checks: Sequence[Mapping[str, Any]]) -> str:
    checks = [c for c in checks if c["samples"]]
    if not checks:
        return "-"
    share = all(c["segments_share_advantage"] and c["one_rollout_per_trial"] for c in checks)
    match = all(c["matches_trial_grpo"] for c in checks)
    return f"{'yes' if share else 'NO'}/{'yes' if match else 'NO'}"


def _table(rows: list[list[str]]) -> str:
    return "\n".join("| " + " | ".join(r) + " |" for r in [rows[0], ["---"] * len(rows[0]), *rows[1:]])


def markdown(all_metrics: Sequence[Mapping[str, Any]]) -> str:
    """Two tables: the sampler side and the trainer side of every scenario."""
    rows = [["scenario", "trials/task (kept)", "outcomes", "segments/trial", "compactions/trial", "summary_ok",
             "trainable/resp", "segment tokens p50 (max)", "env s p50/max", "env+0.1*llm s p50/max",
             "round env / env+0.1*llm s", "yeto cpu ms/call"]]
    for m in all_metrics:
        seg = m["segment_tokens"] or {}
        kept = sorted(set(m["kept_trials_per_task"].values()))
        rows.append([m["scenario"], f"{sorted(set(m['trials_per_task'].values()))} ({kept})",
                     json.dumps(m["outcomes"]), json.dumps(m["segments_per_trial"]),
                     json.dumps(m["compactions_per_trial"]), str(m["summary_ok_rate"]),
                     str(m["trainable_of_response"]), f"{seg.get('p50')} ({seg.get('max')})",
                     f"{m['time_env_s']['p50']}/{m['time_env_s']['max']}",
                     f"{m['time_env_plus_llm_s']['p50']}/{m['time_env_plus_llm_s']['max']}",
                     f"{m['round_time_env_s']} / {m['round_time_env_plus_llm_s']}", str(m["cpu_yeto_per_call_ms"])])
    rows2 = [["scenario", "groups", "per step: samples/trials/rollouts(expected)/padded", "compaction_batch lines",
              "same adv per trial / == trial GRPO", "alignment problems", "status", "versions"]]
    for m in all_metrics:
        steps = ", ".join("error" if "error" in x else
                          f"{x['samples']}/{x['trials']}/{x['distinct_rollouts']}({x['expected_rollouts']})"
                          f"/{x['padded_samples']}" for x in m["trainer_steps"])
        lines = ", ".join("-" if not x.get("compaction_batch") else
                          f"trials={x['compaction_batch']['trials']} segments={x['compaction_batch']['segments']}"
                          for x in m["trainer_steps"] if "error" not in x)
        rows2.append([m["scenario"], str(m["groups_written"]), steps or "-", lines or "-", _check(m["advantage_check"]),
                      str(m["alignment_problem_count"]), json.dumps(m["status"]), json.dumps(m["versions"])])
    return _table(rows) + "\n\n" + _table(rows2)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--scenario", default="all", help=f"one of {sorted(SCENARIOS)} or all")
    parser.add_argument("--json", help="write every scenario's metrics here")
    parser.add_argument("--miles-root", help="a Miles checkout (adds it to sys.path)")
    parser.add_argument("--workdir", help="buffer directory (default: a fresh temporary one per scenario)")
    args = parser.parse_args(argv)
    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    out = []
    for name in names:
        workdir = Path(args.workdir) / name if args.workdir else None
        began = time.perf_counter()
        metrics = SimRun(SCENARIOS[name], workdir=workdir, miles_root=args.miles_root).run().metrics()
        metrics["wall_total_s"] = round(time.perf_counter() - began, 3)
        out.append(metrics)
        print(f"[sim] {name}: samples={metrics['samples']} wall={metrics['wall_total_s']}s", file=sys.stderr,
              flush=True)
    print(markdown(out))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

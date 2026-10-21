"""Miles Harbor path with the records proxy as the token recorder (S25 N10.9).

Miles' own Harbor example records tokens through Miles' session server, which
speaks Chat Completions only; Harbor's codex agent (0.145.0) speaks Responses
only (T-G2).  On this path the yeto records proxy (N10.5) is the recorder, as
on verl:

* ``--custom-generate-function-path yeto.rl.adapters.miles.records_generate.generate``:
  one Harbor trial per sample under the yeto outer layer; the agent calls the
  records proxy, the proxy sends token ids to Miles' SGLang router
  (:class:`SGLangRouterEngine`), and the recorded calls become the Miles
  sample (``join_turns``: tokens, loss mask, rollout log-probs).  The reward
  is the Harbor verifier's and is set here, so Miles' reward hook is not called.
* ``--custom-reward-post-process-path yeto.rl.adapters.miles.records_generate.reward_post_process``:
  GRPO normalisation over each prompt group without its ABORTED members; an
  ABORTED sample has advantage 0 and ``remove_sample`` (loss mask 0).  Groups
  are kept (no dynamic filter), so an ABORTED sample never triggers a new
  sandbox.

A malformed model reply that ends the agent is a trained 0 (POLICY_LIMIT),
not ABORTED (:func:`~yeto.rl.agentic.outcome.policy_output_failure`).

Rollout-process environment (besides the Harbor/boat variables of
``harbor_agent``):
  YETO_RECORDS_PROXY            ``yeto.rl.adapters.miles.records_generate:records_proxy_from_env``
  YETO_RECORDS_PROXY_HOST/PORT/PUBLIC_BASE/SECRET_FILE   (``gateway.recorder``)
  YETO_RECORDS_TEMPLATE, YETO_RECORDS_TOKENIZER, YETO_RECORDS_MODEL,
  YETO_RECORDS_SAMPLING, YETO_RECORDS_MAX_MODEL_LEN, YETO_RECORDS_CARD
                                (``gateway.records_env``)
  YETO_SGLANG_ROUTER_URL        ``http://<ip>:<port>`` of Miles' router (default: from
                                ``--sglang-router-ip`` / ``--sglang-router-port``)

Policy version: Miles' synchronous loop updates the engines between
rollouts, never during one; the proxy records version None and the trainer's
rollout id is the version record (as on verl).
"""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Sequence
from functools import partial
from typing import Any

from yeto.rl.agentic import (
    OUTCOME_KEY,
    AgentTrajectory,
    HarnessConfigError,
    Outcome,
    TokenAlignmentError,
    join_turns,
    run_agent_task,
)
from yeto.rl.agentic.loop_local import LoopLocal
from yeto.rl.agentic.outcome import policy_output_failure
from yeto.rl.engine.sampler_engine import (
    FINISH_LENGTH,
    FINISH_STOP,
    UNKNOWN,
    Completion,
    EngineIdentity,
    LoadReport,
    SamplerEngineError,
    SamplingParams,
)
from yeto.rl.harness.harbor import HarborHarness, HarborSettings, build_trial_config
from yeto.rl.harness.harbor.trial import CUSTOM_PROVIDERS

from .harbor_agent import gate, task_from_metadata

ROUTER_URL_ENV = "YETO_SGLANG_ROUTER_URL"
RECORDER_ENV = "YETO_AGENTIC_RECORDER"
DEFAULT_RECORDER = "yeto.rl.harness.gateway.recorder:recorder_from_env"
GENERATE_FUNCTION = "yeto.rl.adapters.miles.records_generate.generate"
REWARD_POST_PROCESS = "yeto.rl.adapters.miles.records_generate.reward_post_process"
HTTP_TIMEOUT_S = 1800.0


# ---------------------------------------------------------------------- engine
class SGLangRouterEngine:
    """``TokenSamplerEngine`` over an SGLang router (or server) someone else runs (Miles)."""

    def __init__(self, url: str, *, max_model_len: int, identity: EngineIdentity, http: Any = None) -> None:
        self.url = url.rstrip("/")
        self._max_model_len = int(max_model_len)
        self._identity = identity
        self._http = http

    @property
    def identity(self) -> EngineIdentity:
        return self._identity

    @property
    def max_model_len(self) -> int:
        return self._max_model_len

    def served_version(self) -> int | None:
        return None

    def _post(self, body: dict[str, Any]) -> Any:
        if self._http is not None:
            return self._http(f"{self.url}/generate", body)
        import httpx

        reply = httpx.post(f"{self.url}/generate", json=body, timeout=HTTP_TIMEOUT_S)
        reply.raise_for_status()
        return reply.json()

    def generate_tokens(self, prompts: Sequence[Sequence[int]], sampling: SamplingParams, *,
                        stop_token_ids: Sequence[int] = ()) -> list[list[Completion]]:
        """Called from a worker thread (the records backend uses ``asyncio.to_thread``)."""
        params: dict[str, Any] = {
            "temperature": float(sampling.temperature), "top_p": float(sampling.top_p),
            "top_k": int(sampling.top_k), "repetition_penalty": float(sampling.repetition_penalty),
            "max_new_tokens": int(sampling.max_tokens), "skip_special_tokens": False,
            "no_stop_trim": True,
        }
        if stop_token_ids:
            params["stop_token_ids"] = [int(t) for t in stop_token_ids]
        out: list[list[Completion]] = []
        for prompt in prompts:
            group = []
            for _ in range(int(sampling.n)):
                body = self._post({"input_ids": [int(t) for t in prompt], "sampling_params": params,
                                   "return_logprob": True})
                body = body[0] if isinstance(body, list) else body
                meta = body.get("meta_info") or {}
                pairs = meta.get("output_token_logprobs") or []
                tokens = tuple(int(p[1]) for p in pairs)
                finish = (meta.get("finish_reason") or {}).get("type")
                if finish == "abort":
                    raise SamplerEngineError(f"sglang aborted the request: {meta.get('finish_reason')}")
                group.append(Completion(token_ids=tokens, logprobs=tuple(float(p[0]) for p in pairs),
                                        text=str(body.get("text", "")),
                                        finish_reason=FINISH_LENGTH if finish == "length" else FINISH_STOP,
                                        prompt_token_ids=tuple(int(t) for t in prompt)))
            out.append(group)
        return out

    # Miles starts, updates and stops its engines: the proxy never calls these.
    def start(self, release_dir: Any, version: int) -> LoadReport:
        raise SamplerEngineError("Miles owns its rollout engines")

    def load(self, release_dir: Any, version: int) -> LoadReport:
        raise SamplerEngineError("Miles owns its rollout engines")

    def generate(self, prompts: Sequence[str], sampling: SamplingParams) -> list[list[Completion]]:
        raise SamplerEngineError("text prompts are not used; the records proxy sends token ids")

    def score(self, prompt: str, continuation: str) -> list[float]:
        raise SamplerEngineError("scoring is not used on the Miles rollout")

    def loaded_expert_matrices(self) -> int | None:
        return None

    def weight_gib_per_card(self) -> float | None:
        return None

    def stop(self) -> None:
        return None


def records_proxy_from_env() -> Any:
    """``YETO_RECORDS_PROXY`` callable: a RecordsProxy over Miles' router (module doc)."""
    from yeto.rl.harness.gateway.records_env import (
        CARD_ENV,
        MAX_MODEL_LEN_ENV,
        MODEL_ENV,
        SAMPLING_ENV,
        TEMPLATE_ENV,
        TOKENIZER_ENV,
    )
    from yeto.rl.harness.chat_templates import ChatTemplate
    from yeto.rl.harness.gateway.proxy import RecordsProxy

    try:
        url = os.environ[ROUTER_URL_ENV]
        template = ChatTemplate.load(os.environ[TEMPLATE_ENV], tokenizer_path=os.environ[TOKENIZER_ENV])
        signed = json.loads(os.environ[SAMPLING_ENV])
        engine = SGLangRouterEngine(url, max_model_len=int(os.environ[MAX_MODEL_LEN_ENV]), identity=EngineIdentity(
            engine="miles-sglang-router", engine_version=UNKNOWN, device_family="cuda",
            card_model=os.environ.get(CARD_ENV, UNKNOWN), precision="bf16", quant_scheme="bf16",
            topology=UNKNOWN))
        model = os.environ[MODEL_ENV]
    except (KeyError, ValueError) as exc:
        raise HarnessConfigError(f"records proxy over Miles' router: bad or missing setting {exc}") from exc
    return RecordsProxy(engine, template, model=model, signed_sampling=signed, sampler_id="miles-records")


# ---------------------------------------------------------------------- recorder
def _recorder_from_env() -> Any:
    import importlib

    spec = os.environ.get(RECORDER_ENV, DEFAULT_RECORDER)
    module, _, attr = spec.partition(":")
    return getattr(importlib.import_module(module), attr)()


_recorders: LoopLocal[Any] = LoopLocal(_recorder_from_env)


def _harness(task: Any) -> HarborHarness:
    if task.provider not in CUSTOM_PROVIDERS:
        raise HarnessConfigError(f"records_generate runs yeto's TrialConfig builder; provider {task.provider!r} "
                                 f"is not one of {sorted(CUSTOM_PROVIDERS)} (use Miles' example path)")
    settings = HarborSettings.from_env()
    return HarborHarness(partial(build_trial_config, settings=settings), setup_retries=settings.setup_retries)


# ---------------------------------------------------------------------- generate
async def generate(input: Any) -> Any:
    """Miles ``--custom-generate-function-path`` (module doc)."""
    from miles.rollout.base_types import GenerateFnOutput

    args = input.args
    # Miles' router address is an argument, not an env variable; the proxy is built from env once per loop.
    os.environ.setdefault(ROUTER_URL_ENV, f"http://{args.sglang_router_ip}:{args.sglang_router_port}")
    sample = await run_sample(input.sample, input.sampling_params, args, recorder=_recorders.get())
    return GenerateFnOutput(samples=sample)


async def run_sample(sample: Any, sampling_params: dict[str, Any], args: Any, *, recorder: Any,
                     harness: Any = None) -> Any:
    """One Harbor trial -> the filled Miles sample (separate from ``generate`` for CPU tests)."""
    task = task_from_metadata(dict(sample.metadata or {}))
    session = f"miles-{task.task_id}-{os.urandom(6).hex()}"
    endpoint = await recorder.open(session, _verl_style_sampling(sampling_params))
    try:
        trajectory = await run_agent_task(task, harness or _harness(task), endpoint, gate())
    except BaseException:
        with contextlib.suppress(Exception):  # the trial's error is the one to report
            await recorder.close(session)
        raise
    turns: Sequence[Any] = ()
    try:
        turns = await recorder.close(session)
    except TokenAlignmentError as exc:
        trajectory = _reclassify(trajectory, Outcome.ABORTED, f"token_alignment: {exc}")
    else:
        updated = policy_output_failure(trajectory, [t.policy_error for t in turns if t.policy_error])
        if updated.outcome is not trajectory.outcome:
            gate().metrics.reclassify(trajectory.outcome, updated.outcome)
        trajectory = updated
    segments = None
    if trajectory.outcome is not Outcome.ABORTED:
        try:
            segments = join_turns(turns, max_response_len=int(args.rollout_max_response_len))
        except TokenAlignmentError as exc:
            trajectory = _reclassify(trajectory, Outcome.ABORTED, f"token_alignment: {exc}")
    return fill_sample(sample, trajectory, segments, turns)


def _reclassify(trajectory: AgentTrajectory, outcome: Outcome, reason: str) -> AgentTrajectory:
    if trajectory.outcome is not outcome:
        gate().metrics.reclassify(trajectory.outcome, outcome)
    return AgentTrajectory(task=trajectory.task, outcome=outcome, reward=0.0 if outcome is Outcome.ABORTED
                           else trajectory.reward, reason=reason, verdict=trajectory.verdict,
                           policy_version=trajectory.policy_version)


def _verl_style_sampling(sampling_params: dict[str, Any]) -> dict[str, Any]:
    """Miles' SGLang sampling keys -> the recorder's check keys (temperature, top_p, top_k)."""
    return {k: sampling_params[k] for k in ("temperature", "top_p", "top_k", "repetition_penalty")
            if k in sampling_params}


def fill_sample(sample: Any, trajectory: AgentTrajectory, segments: Any, turns: Sequence[Any]) -> Any:
    status = _status()
    metadata = {**(sample.metadata or {}), **trajectory.to_metadata(),
                "length_capped_turns": sum(t.finish_reason == FINISH_LENGTH for t in turns),
                "model_calls": len(turns)}
    if segments is None:  # ABORTED: one masked placeholder token, never trained
        sample.tokens, sample.response_length = [0, 0], 1
        sample.loss_mask, sample.rollout_log_probs = [0], [0.0]
        sample.remove_sample = True
        sample.status = status.ABORTED
        sample.reward = 0.0
    else:
        sample.tokens = [*segments.prompt_ids, *segments.response_ids]
        sample.response_length = len(segments.response_ids)
        sample.loss_mask = list(segments.response_mask)
        sample.rollout_log_probs = list(segments.response_logprobs)
        sample.status = status.TRUNCATED if segments.truncated else status.COMPLETED
        sample.reward = float(trajectory.reward)
        metadata["truncated"] = bool(segments.truncated)
    sample.metadata = metadata
    return sample


def _status() -> Any:
    try:
        from miles.utils.types import Sample

        return Sample.Status
    except ImportError:  # CPU tests without Miles
        from enum import Enum

        return Enum("Status", "PENDING COMPLETED TRUNCATED ABORTED FAILED")


# ---------------------------------------------------------------------- rewards
def is_aborted(sample: Any) -> bool:
    return (sample.metadata or {}).get(OUTCOME_KEY) == Outcome.ABORTED.value


def reward_post_process(args: Any, samples: Sequence[Any]) -> tuple[list[float], list[float]]:
    """GRPO over each group of ``n_samples_per_prompt`` without its ABORTED members (module doc)."""
    n = int(args.n_samples_per_prompt)
    if len(samples) % n:
        raise ValueError(f"{len(samples)} samples are not whole groups of {n}")
    by_std = bool(getattr(args, "grpo_std_normalization", True))
    raw = [0.0 if is_aborted(s) else float(s.reward) for s in samples]
    out = [0.0] * len(samples)
    for start in range(0, len(samples), n):
        members = [i for i in range(start, start + n) if not is_aborted(samples[i])]
        if not members:
            continue
        values = [raw[i] for i in members]
        mean = sum(values) / len(values)
        std = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5 if len(values) > 1 else 0.0
        for i in members:
            out[i] = (raw[i] - mean) / (std + 1e-6) if by_std else raw[i] - mean
    for sample in samples:
        if is_aborted(sample):
            sample.remove_sample = True
    return raw, out

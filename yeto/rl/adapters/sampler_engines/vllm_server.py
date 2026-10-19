"""``vllm-server``: a vLLM OpenAI-compatible server behind the ``SamplerEngine``
port (vllm-ascend 0.30.0rc1 on 910B, stage 2: TP4 per instance, W4A8).

Vendor facts handled here (A-INF §3, report items 15-18; none verified on a
910B by this code):

* reload: restart only. In-place reload of W4A8_DYNAMIC on vllm-ascend is not
  tested (GOAL §8); ``in_place_reload`` stays False until N4.2 measures it, and
  this adapter has no in-place path yet;
* item 17: 910B reads ``int-quantized`` only, not ``pack-quantized``; the
  default ``expected_format`` for a quantized scheme is ``int-quantized``.
  W4A8 stages int8 weights in host memory (``VLLM_ASCEND_W4A8_CPU_STAGE=1``),
  so set ``min_host_free_gib`` from the measured peak (report: about 610 GiB);
* item 18: eager mode serves about 2.75 tok/s against about 38 tok/s with
  graphs; ``--enforce-eager`` is refused unless ``allow_eager`` (debug only);
* item 15: a missing CANN 9.1 libprofapi stub hangs HCCL without an error; the
  start timeout names it;
* items 16 and the PT_INTERP fix belong to the image or rootfs start script
  (N0.2), not to this adapter.

API calls (vLLM OpenAI-compatible server): ``GET /health``;
``POST /v1/completions`` with ``logprobs: 1`` and
``return_tokens_as_token_ids`` (tokens come back as ``token_id:<id>``);
teacher-forced scores with ``echo: true`` and ``max_tokens: 1``. Field
support in vllm-ascend 0.30.0rc1 is NOT verified.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from yeto.rl.engine.sampler_engine import (
    FINISH_LENGTH,
    FINISH_STOP,
    Completion,
    SamplerEngineError,
    SamplingParams,
)

from .server import ServerConfig, ServerEngine

INT_QUANTIZED = "int-quantized"


@dataclass
class VllmOptions:
    tp: int = 4
    max_num_seqs: int | None = None  # N4.1 measures 8 / 32 / 64
    served_model_name: str = "yeto-policy"
    allow_eager: bool = False


def _token_id(token: str) -> int:
    if not isinstance(token, str) or not token.startswith("token_id:"):
        raise SamplerEngineError(f"vllm token {token!r} is not 'token_id:<id>'; "
                                 "the server ignored return_tokens_as_token_ids")
    return int(token.split(":", 1)[1])


class VllmServerEngine(ServerEngine):
    ENGINE = "vllm"
    HANG_HINT = ("Suspected HCCL hang: check that the CANN 9.1 libprofapi stub is installed "
                 "(report item 15).")
    REFUSED_FORMATS = frozenset({"pack-quantized", "dense"})

    def __init__(self, config: ServerConfig, options: VllmOptions | None = None, **kw: Any) -> None:
        if config.quant_scheme != "bf16" and config.expected_format is None:
            config.expected_format = INT_QUANTIZED  # item 17
        if config.in_place_reload:
            raise SamplerEngineError("vllm-server: in-place reload is not built; restart is the "
                                     "only reload mode until N4.2 tests it")
        super().__init__(config, **kw)
        self.options = options or VllmOptions()
        if "--enforce-eager" in config.extra_args and not self.options.allow_eager:
            raise SamplerEngineError("vllm-server: --enforce-eager serves about 14x slower on 910B "
                                     "(report item 18); set allow_eager only for debugging")

    def argv(self, model_dir: Path) -> list[str]:
        c, o = self.config, self.options
        argv = [c.python, "-m", "vllm.entrypoints.openai.api_server", "--model", str(model_dir),
                "--host", c.host, "--port", str(c.port), "--tensor-parallel-size", str(o.tp),
                "--max-model-len", str(c.max_model_len), "--served-model-name", o.served_model_name,
                "--trust-remote-code"]
        if o.max_num_seqs is not None:
            argv += ["--max-num-seqs", str(int(o.max_num_seqs))]
        return argv + list(c.extra_args)

    def env(self) -> dict[str, str]:
        env = super().env()
        if self.config.precision == "w4a8":
            env["VLLM_ASCEND_W4A8_CPU_STAGE"] = "1"  # int8 staging in host memory
        return env

    def _body(self, prompt: Any, sampling: SamplingParams, **extra: Any) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.options.served_model_name, "prompt": prompt, "n": int(sampling.n),
            "max_tokens": int(sampling.max_tokens), "temperature": float(sampling.temperature),
            "top_p": float(sampling.top_p), "top_k": int(sampling.top_k),
            "repetition_penalty": float(sampling.repetition_penalty),
            "frequency_penalty": float(sampling.frequency_penalty),
            "presence_penalty": float(sampling.presence_penalty),
            "logprobs": 1, "return_tokens_as_token_ids": True, **extra}
        if sampling.stop:
            body["stop"] = list(sampling.stop)
        if sampling.seed is not None:
            body["seed"] = int(sampling.seed)
        return body

    def generate(self, prompts: Sequence[str], sampling: SamplingParams) -> list[list[Completion]]:
        n = int(sampling.n)
        body = self.call("POST", "/v1/completions", self._body(list(prompts), sampling))
        choices = sorted(body.get("choices") or [], key=lambda c: int(c["index"]))
        if len(choices) != len(prompts) * n:
            raise SamplerEngineError(f"vllm returned {len(choices)} choices for {len(prompts)} x {n}")
        out: list[list[Completion]] = []
        for i in range(len(prompts)):
            group = []
            for choice in choices[i * n:(i + 1) * n]:
                logprobs = choice.get("logprobs") or {}
                reason = choice.get("finish_reason")
                if reason not in ("stop", "length"):
                    raise SamplerEngineError(f"vllm finish_reason {reason!r}")
                group.append(Completion(
                    token_ids=tuple(_token_id(t) for t in logprobs.get("tokens") or []),
                    logprobs=tuple(float(x) for x in logprobs.get("token_logprobs") or []),
                    text=str(choice.get("text", "")),
                    finish_reason=FINISH_LENGTH if reason == "length" else FINISH_STOP))
            out.append(group)
        return out

    def _echo_logprobs(self, text: str) -> list[float | None]:
        greedy = SamplingParams(max_tokens=1, temperature=0.0)
        body = self.call("POST", "/v1/completions", self._body(text, greedy, echo=True))
        choice = (body.get("choices") or [{}])[0]
        values = (choice.get("logprobs") or {}).get("token_logprobs") or []
        return list(values[:-1])  # the last value is the one generated token

    def score(self, prompt: str, continuation: str) -> list[float]:
        prompt_len = len(self._echo_logprobs(prompt))
        tail = self._echo_logprobs(prompt + continuation)[prompt_len:]
        if not tail or any(x is None for x in tail):
            raise SamplerEngineError(f"vllm gave no prompt log-probs for continuation {continuation!r}")
        return [float(x) for x in tail]

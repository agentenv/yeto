"""``sglang-server``: an SGLang HTTP server behind the ``SamplerEngine`` port
(PPU, stage 1: TP8 per machine, MXFP4 experts RTN; user decision 2026-10-19).

Vendor facts handled here (A-INF §3; none verified on a PPU by this code):

* item 1: SGLang's ``load_weights`` does not map ``weight_packed`` to
  ``weight``; expert weights stored as ``weight_packed`` are skipped without an
  error. A release whose index names ``*.weight_packed`` is refused unless the
  image carries the rename patch (``renames_weight_packed=True``);
* item 5: an MXFP4 export with ``quantization_config.format == "dense"``
  (llmcompressor 0.14) is read wrongly; ``dense`` is refused for every
  quantized scheme;
* item 7: the MXFP4 W4A16 path needs ``MX_PATCH=1``; set when the precision is
  ``mxfp4`` and recorded in the load report;
* item 8: dp8 + DeepEP cannot run NEXTN; the pair is refused before start;
* item 9: dp8 + DeepGEMM compiles per new batch size M (75-113 s each); with
  ``warmup_batch_sizes`` the adapter sends one batch per size before the
  engine is checked and records the seconds;
* item 13: TP8 exits 134 from ``pcclCommAbort`` after a stop request; normal
  then, a failure while serving.

API calls (SGLang native HTTP API; the exact fields of SGLang 0.5.17 on PPU are
NOT verified): ``GET /health``; ``POST /generate`` with ``return_logprob``
(``meta_info.output_token_logprobs`` = ``[logprob, token_id, ...]`` per
token, ``meta_info.finish_reason.type``); teacher-forced scores from
``meta_info.input_token_logprobs`` with ``max_new_tokens: 0``;
``POST /update_weights_from_disk`` for an in-place reload (off unless
``in_place_reload``: refit of quantized experts is not tested, A-INF 2.1).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from yeto.rl.engine.sampler_checks import INDEX_FILE
from yeto.rl.engine.sampler_engine import (
    FINISH_LENGTH,
    FINISH_STOP,
    Completion,
    SamplerEngineError,
    SamplingParams,
)

from .server import ServerConfig, ServerEngine

TOPOLOGY_TP8 = "tp8"
TOPOLOGY_DP8 = "dp8"
SGLANG_TOPOLOGIES = (TOPOLOGY_TP8, TOPOLOGY_DP8)


@dataclass
class SGLangOptions:
    tp: int = 8
    nextn: bool = False
    renames_weight_packed: bool = False  # the image patches the weight_packed -> weight map
    warmup_batch_sizes: tuple[int, ...] = ()
    served_model_name: str = "yeto-policy"


def _finish(meta: dict[str, Any]) -> str:
    reason = meta.get("finish_reason")
    kind = reason.get("type") if isinstance(reason, dict) else reason
    if kind == "length":
        return FINISH_LENGTH
    if kind == "stop":
        return FINISH_STOP
    raise SamplerEngineError(f"sglang finish_reason {reason!r}: request aborted or unknown")


class SGLangServerEngine(ServerEngine):
    ENGINE = "sglang"
    HANG_HINT = "Check the PCCL/ACCL-P collective setup of the PPU image."
    EXIT_CODES_AFTER_STOP = frozenset({134})  # pcclCommAbort at TP8 shutdown (item 13)
    REFUSED_FORMATS = frozenset({"dense"})  # item 5

    def __init__(self, config: ServerConfig, options: SGLangOptions | None = None, **kw: Any) -> None:
        super().__init__(config, **kw)
        self.options = options or SGLangOptions()
        if config.topology not in SGLANG_TOPOLOGIES:
            raise SamplerEngineError(f"sglang topology {config.topology!r} is not supported "
                                     f"(known: {SGLANG_TOPOLOGIES}; PD disaggregation is not used)")
        if config.topology == TOPOLOGY_DP8 and self.options.nextn:
            raise SamplerEngineError("sglang dp8 + DeepEP cannot run NEXTN (A-INF item 8); "
                                     "turn one of them off")

    def argv(self, model_dir: Path) -> list[str]:
        c, o = self.config, self.options
        argv = [c.python, "-m", "sglang.launch_server", "--model-path", str(model_dir),
                "--host", c.host, "--port", str(c.port), "--tp", str(o.tp),
                "--context-length", str(c.max_model_len),
                "--served-model-name", o.served_model_name, "--trust-remote-code"]
        if c.topology == TOPOLOGY_DP8:
            argv += ["--dp", str(o.tp), "--enable-dp-attention", "--moe-a2a-backend", "deepep"]
        if o.nextn:
            argv += ["--speculative-algorithm", "NEXTN"]
        return argv + list(c.extra_args)

    def env(self) -> dict[str, str]:
        env = super().env()
        if self.config.precision == "mxfp4":
            env["MX_PATCH"] = "1"  # item 7
        return env

    def preflight(self, model_dir: Path) -> None:
        super().preflight(model_dir)
        index = Path(model_dir) / INDEX_FILE
        if index.is_file() and not self.options.renames_weight_packed:
            names = json.loads(index.read_text(encoding="utf-8")).get("weight_map") or {}
            packed = [n for n in names if n.endswith(".weight_packed")]
            if packed:
                raise SamplerEngineError(
                    f"{model_dir}: {len(packed)} tensors are named *.weight_packed (e.g. "
                    f"{packed[0]}); SGLang skips them without an error (A-INF item 1). Use an "
                    "image with the rename patch and set renames_weight_packed")

    def after_ready(self) -> dict[str, Any]:
        detail: dict[str, Any] = {"env": {k: v for k, v in self.env().items() if k == "MX_PATCH"}}
        if self.options.warmup_batch_sizes:
            began = self._clock()
            for size in self.options.warmup_batch_sizes:
                self._generate_raw(["warm-up"] * int(size),
                                   SamplingParams(max_tokens=8, temperature=0.0))
            detail["warmup_batch_sizes"] = list(self.options.warmup_batch_sizes)
            detail["warmup_s"] = round(self._clock() - began, 3)
        return detail

    def reload_in_place(self, model_dir: Path, version: int) -> None:
        body = self.call("POST", "/update_weights_from_disk",
                         {"model_path": str(model_dir), "weight_version": str(int(version))})
        if isinstance(body, dict) and body.get("success") is False:
            raise SamplerEngineError(f"sglang update_weights_from_disk: {body.get('message')}")

    # ------------------------------------------------------------------ generation
    @staticmethod
    def _params(sampling: SamplingParams, max_new_tokens: int | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {
            "max_new_tokens": int(sampling.max_tokens if max_new_tokens is None else max_new_tokens),
            "temperature": float(sampling.temperature), "top_p": float(sampling.top_p),
            "top_k": int(sampling.top_k),
            "repetition_penalty": float(sampling.repetition_penalty),
            "frequency_penalty": float(sampling.frequency_penalty),
            "presence_penalty": float(sampling.presence_penalty),
        }
        if sampling.stop:
            params["stop"] = list(sampling.stop)
        if sampling.seed is not None:
            params["sampling_seed"] = int(sampling.seed)
        return params

    def _generate_raw(self, texts: Sequence[str], sampling: SamplingParams) -> list[dict[str, Any]]:
        body = self.call("POST", "/generate", {"text": list(texts), "sampling_params": self._params(sampling),
                                               "return_logprob": True})
        items = body if isinstance(body, list) else [body]
        if len(items) != len(texts):
            raise SamplerEngineError(f"sglang returned {len(items)} outputs for {len(texts)} inputs")
        return items

    def generate(self, prompts: Sequence[str], sampling: SamplingParams) -> list[list[Completion]]:
        n = int(sampling.n)
        texts = [p for p in prompts for _ in range(n)]  # one request per sample: order is explicit
        items = self._generate_raw(texts, sampling.replace(n=1))
        out: list[list[Completion]] = []
        for i in range(len(prompts)):
            group = []
            for item in items[i * n:(i + 1) * n]:
                meta = item.get("meta_info") or {}
                tokens = meta.get("output_token_logprobs") or []
                group.append(Completion(token_ids=tuple(int(t[1]) for t in tokens),
                                        logprobs=tuple(float(t[0]) for t in tokens),
                                        text=str(item.get("text", "")), finish_reason=_finish(meta)))
            out.append(group)
        return out

    def _input_logprobs(self, text: str) -> list[float | None]:
        body = self.call("POST", "/generate", {
            "text": text, "sampling_params": {"max_new_tokens": 0, "temperature": 0.0},
            "return_logprob": True, "logprob_start_len": 0})
        body = body[0] if isinstance(body, list) else body
        return [None if t[0] is None else float(t[0])
                for t in (body.get("meta_info") or {}).get("input_token_logprobs") or []]

    def score(self, prompt: str, continuation: str) -> list[float]:
        prompt_len = len(self._input_logprobs(prompt))
        full = self._input_logprobs(prompt + continuation)
        tail = full[prompt_len:]
        if not tail or any(x is None for x in tail):
            raise SamplerEngineError(f"sglang gave no input log-probs for continuation {continuation!r}")
        return [float(x) for x in tail]

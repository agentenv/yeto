"""``sglang-server``: an SGLang HTTP server behind the ``SamplerEngine`` port
(PPU, stage 1: TP8 per machine, MXFP4 experts RTN; user decision 2026-10-19).

Vendor facts handled here (A-INF §3; none verified on a PPU by this code):

* item 1: SGLang's ``load_weights`` does not map ``weight_packed`` to
  ``weight``; expert weights stored as ``weight_packed`` are skipped without an
  error. A release whose index names ``*.weight_packed`` is refused unless the
  image carries the rename patch (``renames_weight_packed=True``). The yeto
  patch (:mod:`.sglang_load_report`, applied to the image's model file) renames
  when ``YETO_SGLANG_RENAME_WEIGHT_PACKED=1`` and, with ``load_report_dir``,
  reports the expert matrices each rank loaded (post-load check 1);
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

Updates during a generation (``in_place_reload`` and ``live_update``; the
``updates_during_generation`` capability). Without a pause, SGLang's
``update_weights_from_disk`` takes the model-update writer lock and waits for
every running request, so a release waits behind the whole round. With
``live_update`` the adapter instead:

1. ``POST /pause_generation`` with ``mode: retract``: the scheduler stops and
   moves the running requests back to its waiting queue (their sampled tokens
   are kept, their KV cache is freed);
2. ``POST /update_weights_from_disk`` with ``flush_cache: false`` (a cache
   flush asserts an idle scheduler, and the retracted requests are queued);
   the paused server takes the update at once;
3. ``POST /continue_generation`` (also after a failed update).

The retracted requests then recompute their KV cache with the new weights.
The adapter starts the server with ``--disable-radix-cache`` so no prefix KV
computed with older weights is reused. Each completion's ``weight_version``
is the version of its first output token and ``final_weight_version`` the
version of its last, from ``meta_info.weight_versions`` (per-token version
spans of the pinned SGLang fork 2fa88018). An SGLang without spans gives
only ``meta_info.weight_version`` (the version when it ended); the adapter
then uses the version served when the request was sent as
``weight_version`` (never newer than the real one: a request still queued
at the update counts at the older version).
Checked against SGLang 0.5.20 (miles-next venv; the miles-ports image pins
0.5.21.dev70) source; on card see SESSION27/gpu-review/SAMPLER-LIVE.md.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
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

from . import sglang_load_report
from .server import ServerConfig, ServerEngine

TOPOLOGY_TP8 = "tp8"
TOPOLOGY_DP8 = "dp8"
_TOPOLOGY = re.compile(r"(tp|dp)([1-9][0-9]*)")


def sglang_topology(tp: int, *, dp_attention: bool = False) -> str:
    """The topology label of one SGLang instance over ``tp`` cards: ``tp<N>``, or
    ``dp<N>`` with DP attention (``--dp N --enable-dp-attention``)."""
    return f"{'dp' if dp_attention else 'tp'}{int(tp)}"


def is_dp_attention(topology: str) -> bool:
    return topology.startswith("dp")


@dataclass
class SGLangOptions:
    tp: int = 8
    nextn: bool = False
    renames_weight_packed: bool = False  # the image patches the weight_packed -> weight map
    warmup_batch_sizes: tuple[int, ...] = ()
    served_model_name: str = "yeto-policy"
    # the yeto load-report patch writes each rank's loaded expert count here (None: no report)
    load_report_dir: str | None = None
    # update the weights while a generation runs (module doc); needs ServerConfig.in_place_reload
    live_update: bool = False


def _finish(meta: dict[str, Any]) -> str:
    reason = meta.get("finish_reason")
    kind = reason.get("type") if isinstance(reason, dict) else reason
    if kind == "length":
        return FINISH_LENGTH
    if kind == "stop":
        return FINISH_STOP
    raise SamplerEngineError(f"sglang finish_reason {reason!r}: request aborted or unknown")


def _completion_versions(meta: dict[str, Any], sent: int, started: int) -> dict[str, int]:
    """``weight_version`` and ``final_weight_version`` of one completion (module doc).

    With ``meta_info.weight_versions`` (spans ``{version, start, end}`` of the
    output tokens; the pinned SGLang fork) they are the first and last span's
    version. Without spans: the version the request was sent at, and
    ``meta_info.weight_version``. ``default`` (no in-place update yet) is the
    version the server was started on. The final version may be below the
    first after a new lineage (a lower version with a higher seq)."""

    def number(value: Any, fallback: int) -> int:
        if value == "default":
            return started
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    spans = meta.get("weight_versions")
    if isinstance(spans, list) and spans:
        first = number(spans[0].get("version"), sent)
        return {"weight_version": first,
                "final_weight_version": number(spans[-1].get("version"), first)}
    return {"weight_version": sent, "final_weight_version": number(meta.get("weight_version"), sent)}


class SGLangServerEngine(ServerEngine):
    ENGINE = "sglang"
    HANG_HINT = "Check the PCCL/ACCL-P collective setup of the PPU image."
    EXIT_CODES_AFTER_STOP = frozenset({134})  # pcclCommAbort at TP8 shutdown (item 13)
    REFUSED_FORMATS = frozenset({"dense"})  # item 5

    def __init__(self, config: ServerConfig, options: SGLangOptions | None = None, **kw: Any) -> None:
        super().__init__(config, **kw)
        self.options = options or SGLangOptions()
        self._started_version = 0
        # The topology is the source label of every sample: it must name the real
        # layout, tp<N> or dp<N> with N = the instance's card count (options.tp).
        match = _TOPOLOGY.fullmatch(config.topology)
        if match is None or int(match.group(2)) != int(self.options.tp):
            raise SamplerEngineError(
                f"sglang topology {config.topology!r} does not match TP {self.options.tp}: expected "
                f"{sglang_topology(self.options.tp)!r} or {sglang_topology(self.options.tp, dp_attention=True)!r} "
                "(PD disaggregation is not used)")
        if self.options.live_update and not config.in_place_reload:
            raise SamplerEngineError("sglang live_update loads weights in place; set in_place_reload")
        if is_dp_attention(config.topology) and self.options.nextn:
            raise SamplerEngineError("sglang DP attention + DeepEP cannot run NEXTN (A-INF item 8); "
                                     "turn one of them off")

    @property
    def updates_during_generation(self) -> bool:
        """The port capability (``sampler_engine.updates_during_generation``)."""
        return bool(self.options.live_update)

    def argv(self, model_dir: Path) -> list[str]:
        c, o = self.config, self.options
        argv = [c.python, "-m", "sglang.launch_server", "--model-path", str(model_dir),
                "--host", c.host, "--port", str(c.port), "--tp", str(o.tp),
                "--context-length", str(c.max_model_len),
                "--served-model-name", o.served_model_name, "--trust-remote-code"]
        if is_dp_attention(c.topology):
            argv += ["--dp", str(o.tp), "--enable-dp-attention", "--moe-a2a-backend", "deepep"]
        if o.nextn:
            argv += ["--speculative-algorithm", "NEXTN"]
        if o.live_update and "--disable-radix-cache" not in c.extra_args:
            argv += ["--disable-radix-cache"]  # no prefix KV of older weights after an update
        return argv + list(c.extra_args)

    def env(self) -> dict[str, str]:
        env = super().env()
        if self.config.precision == "mxfp4":
            env["MX_PATCH"] = "1"  # item 7
        if self.options.renames_weight_packed:
            env[sglang_load_report.RENAME_ENV] = "1"
        if self.options.load_report_dir:
            env[sglang_load_report.REPORT_DIR_ENV] = str(self.options.load_report_dir)
        return env

    def start(self, release_dir: Path | None, version: int):
        self._started_version = int(version)  # what ``default`` means in this server's versions
        if self.options.load_report_dir:  # reports of an earlier server must not count for this one
            sglang_load_report.clear_reports(self.options.load_report_dir)
        return super().start(release_dir, version)

    def loaded_expert_matrices(self) -> int | None:
        if not self.options.load_report_dir:
            return None
        report = sglang_load_report.read_reports(self.options.load_report_dir)
        if report is None:
            return None
        if report["ranks"] < int(self.options.tp):
            return 0  # a rank that did not report loaded nothing we can vouch for
        return int(report["expert_matrices"])

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

    def reload_in_place(self, model_dir: Path, version: int) -> dict[str, Any] | None:
        request = {"model_path": str(model_dir), "weight_version": str(int(version))}
        if not self.options.live_update:
            self._update_from_disk(request)
            return None
        began = self._clock()
        self.call("POST", "/pause_generation", {"mode": "retract"})
        paused = self._clock()
        try:
            body = self._update_from_disk({**request, "flush_cache": False})
        finally:
            updated = self._clock()
            self.call("POST", "/continue_generation", {})
        return {"live": True, "pause_call_s": round(paused - began, 3),
                "update_s": round(updated - paused, 3), "paused_s": round(self._clock() - paused, 3),
                "num_paused_requests": body.get("num_paused_requests") if isinstance(body, dict) else None}

    def _update_from_disk(self, request: dict[str, Any]) -> Any:
        body = self.call("POST", "/update_weights_from_disk", request)
        if isinstance(body, dict) and body.get("success") is False:
            raise SamplerEngineError(f"sglang update_weights_from_disk: {body.get('message')}")
        return body

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

    def _sent_version(self) -> int | None:
        """The version a request sent now starts on (only with live updates)."""
        return self._version if self.options.live_update else None

    def generate(self, prompts: Sequence[str], sampling: SamplingParams) -> list[list[Completion]]:
        n = int(sampling.n)
        texts = [p for p in prompts for _ in range(n)]  # one request per sample: order is explicit
        sent = self._sent_version()
        return self._completions(self._generate_raw(texts, sampling.replace(n=1)), len(prompts), n,
                                 sent_version=sent, started_version=self._started_version)

    def generate_tokens(self, prompts: Sequence[Sequence[int]], sampling: SamplingParams, *,
                        stop_token_ids: Sequence[int] = ()) -> list[list[Completion]]:
        """Token-id prompts (``input_ids``), for the records proxy (S25 N10.5)."""
        n = int(sampling.n)
        inputs = [[int(t) for t in p] for p in prompts for _ in range(n)]
        params = self._params(sampling.replace(n=1))
        sent = self._sent_version()
        if stop_token_ids:
            params["stop_token_ids"] = [int(t) for t in stop_token_ids]
        body = self.call("POST", "/generate", {"input_ids": inputs, "sampling_params": params,
                                               "return_logprob": True})
        items = body if isinstance(body, list) else [body]
        if len(items) != len(inputs):
            raise SamplerEngineError(f"sglang returned {len(items)} outputs for {len(inputs)} inputs")
        completions = self._completions(items, len(prompts), n, sent_version=sent,
                                        started_version=self._started_version)
        return [[c if c.prompt_token_ids else replace(c, prompt_token_ids=tuple(inputs[i * n]))
                 for c in group] for i, group in enumerate(completions)]

    @staticmethod
    def _completions(items: Sequence[dict[str, Any]], prompts: int, n: int, *,
                     sent_version: int | None = None, started_version: int = 0) -> list[list[Completion]]:
        """``sent_version`` (live updates only): set both completion versions
        (:func:`_completion_versions`)."""
        out: list[list[Completion]] = []
        for i in range(prompts):
            group = []
            for item in items[i * n:(i + 1) * n]:
                meta = item.get("meta_info") or {}
                tokens = meta.get("output_token_logprobs") or []
                versions = ({} if sent_version is None else
                            _completion_versions(meta, sent_version, started_version))
                group.append(Completion(token_ids=tuple(int(t[1]) for t in tokens),
                                        logprobs=tuple(float(t[0]) for t in tokens),
                                        text=str(item.get("text", "")), finish_reason=_finish(meta),
                                        **versions))
            out.append(group)
        return out

    def _input_token_logprobs(self, prompt: dict[str, Any]) -> list[tuple[float | None, int]]:
        """``(logprob, token id)`` of every input token; ``prompt`` is ``{"text": ...}`` or
        ``{"input_ids": [...]}``."""
        body = self.call("POST", "/generate", {
            **prompt, "sampling_params": {"max_new_tokens": 0, "temperature": 0.0},
            "return_logprob": True, "logprob_start_len": 0})
        body = body[0] if isinstance(body, list) else body
        return [(None if t[0] is None else float(t[0]), int(t[1]))
                for t in (body.get("meta_info") or {}).get("input_token_logprobs") or []]

    def _input_logprobs(self, text: str) -> list[float | None]:
        return [lp for lp, _ in self._input_token_logprobs({"text": text})]

    def score(self, prompt: str, continuation: str) -> list[float]:
        prompt_len = len(self._input_logprobs(prompt))
        full = self._input_logprobs(prompt + continuation)
        tail = full[prompt_len:]
        if not tail or any(x is None for x in tail):
            raise SamplerEngineError(f"sglang gave no input log-probs for continuation {continuation!r}")
        return [float(x) for x in tail]

    def score_tokens(self, prompt: str, token_ids: Sequence[int]) -> list[float]:
        """Teacher-forced log-probs of exactly ``token_ids`` after ``prompt`` (no re-tokenizing
        of the continuation, so a trailing EOS that has no text is scored too)."""
        prompt_ids = [tid for _, tid in self._input_token_logprobs({"text": prompt})]
        full = self._input_token_logprobs({"input_ids": [*prompt_ids, *map(int, token_ids)]})
        tail = [lp for lp, _ in full[len(prompt_ids):]]
        if len(tail) != len(token_ids) or any(x is None for x in tail):
            raise SamplerEngineError(f"sglang scored {len(tail)} of {len(token_ids)} continuation tokens")
        return [float(x) for x in tail]

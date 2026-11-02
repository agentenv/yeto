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

The retracted requests then go on from the KV cache SGLang still holds: the
radix (prefix) cache is kept across the update (``live_prefix_cache``, default
on), so prefix KV computed with older weights may be reused, as PipelineRL,
prime-rl and SkyRL do (PipelineRL measured the extra divergence as small). The
generation log-probabilities are recorded as sampled, so TIS corrects against
the real behaviour. ``live_prefix_cache=False`` starts the server with
``--disable-radix-cache`` instead (no stale prefix, but every agentic turn
re-prefills its whole history). Each completion's ``weight_version``
is the version of its first output token and ``final_weight_version`` the
version of its last, from ``meta_info.weight_versions`` (per-token version
spans of the pinned SGLang fork 2fa88018). An SGLang without spans gives
only ``meta_info.weight_version`` (the version when it ended); the adapter
then uses the version served when the request was sent as
``weight_version`` (never newer than the real one: a request still queued
at the update counts at the older version).
Checked against SGLang 0.5.20 (miles-next venv; the miles-ports image pins
0.5.21.dev70) source.

Sparse updates (``sparse_update`` ``auto``; off by default). A full in-place
reload reads the whole checkpoint on every rank. When the new release was
rebuilt from the served one with change tracking (``release_delta``
``DeltaMirror(track_changes=True)``, ``YETO_RL_MIRROR_TRACK_CHANGES=1``), the
adapter instead sends only the changed elements (:mod:`.sparse_update`):

1. at start, the custom weight loader (:mod:`.sglang_sparse_loader`, passed as
   ``--custom-weight-loader``) probes how ``load_weights`` places every
   checkpoint element on every rank and writes an element map; the served
   release is then reloaded from disk (the probe may leave side effects) and a
   sample of elements is compared with the release (a layout the probe did not
   see, e.g. a post-load reshuffle, disables sparse updates);
2. per update, before any pause: the changed words of the manifest become
   changed tensor elements, and those become one patch per rank (flat
   parameter positions and new bits) plus check samples of changed and
   unchanged elements;
3. ``POST /update_weights_from_tensor`` with ``load_format`` = the loader (one
   command per rank, no weight data over HTTP; under ``live_update`` between
   the same pause and continue as a full reload); each rank writes its patch
   and compares the samples.

It falls back to the full reload, before touching the weights, when the
release has no change manifest, was not rebuilt from the served directory,
changes more than ``sparse_max_ratio`` of the elements, or changes a tensor
the probe could not map; after a failed write or check it reloads in full at
once (``sparse_repair`` in the load record). Supported: bf16 weights without
NEXTN (a draft model would not be updated). Not supported: quantized weights
(FP8/MXFP4 parameters are not the checkpoint's bf16 words) and models whose
``load_weights`` derives other tensors from the loaded ones.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from yeto.rl.engine.sampler_checks import INDEX_FILE
from yeto.rl.engine.sampler_engine import (
    FINISH_LENGTH,
    FINISH_STOP,
    Completion,
    SamplerEngineError,
    SamplingParams,
)

from . import sglang_load_report, sparse_update
from .server import ServerConfig, ServerEngine
from .sglang_sparse_loader import LOADER as SPARSE_LOADER
from .sglang_sparse_loader import command_name

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
    # keep the radix prefix cache under live_update (stale prefix KV is reused; module doc)
    live_prefix_cache: bool = True
    # update only the changed weight elements (module doc): "off" or "auto" (a full
    # reload whenever a sparse update is not possible)
    sparse_update: str = "off"
    sparse_max_ratio: float = 0.05  # changed elements / all elements above which the update is full
    sparse_dir: str | None = None  # element maps and patches (default: /dev/shm or the temp directory)
    sparse_probe_device: str = "cpu"  # where the probe's float64 parameters live
    sparse_check_tensors: int = 8  # check samples: changed tensors and as many other tensors
    sparse_check_elements: int = 256  # elements per sampled tensor
    sparse_prepare_timeout_s: float = 3600.0


SPARSE_MODES = ("off", "auto")


def _sparse_payload(command: dict[str, Any]) -> str:
    """One rank's ``serialized_named_tensors`` entry: the command in the tensor
    name and a one-element placeholder tensor (plain pickle, which SGLang's
    restricted unpickler accepts, base64 for JSON)."""
    import base64
    import pickle

    import torch

    return base64.b64encode(pickle.dumps([(command_name(command), torch.zeros(1))])).decode("ascii")


class _ProbeReloadFailed(SamplerEngineError):
    """The reload of the served release after the sparse probe failed: the
    weights may still hold probe side effects, so the start fails."""


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
        if self.options.sparse_update not in SPARSE_MODES:
            raise SamplerEngineError(f"sglang sparse_update must be one of {SPARSE_MODES}, "
                                     f"got {self.options.sparse_update!r}")
        # why sparse updates are off for this engine (None: they are tried)
        self.sparse_off: str | None = self._sparse_unsupported()
        self._served_dir: Path | None = None
        self._sparse_maps: list[dict[str, Any]] | None = None
        self._sparse_shapes: dict[str, list[int]] | None = None
        self.sparse_prepare: dict[str, Any] | None = None  # the last probe's record
        self._weight_sessions: bool | None = None  # /begin_weight_update exists (None: not asked yet)

    def _sparse_unsupported(self) -> str | None:
        c, o = self.config, self.options
        if o.sparse_update == "off":
            return "off"
        if not c.in_place_reload:
            return "needs in_place_reload"
        if c.precision != "bf16" or c.quant_scheme != "bf16":
            return f"weights are {c.precision}/{c.quant_scheme}, not bf16"
        if o.nextn:
            return "NEXTN draft model"
        return None

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
        if o.live_update and not o.live_prefix_cache and "--disable-radix-cache" not in c.extra_args:
            argv += ["--disable-radix-cache"]  # no prefix KV of older weights after an update
        if self.sparse_off is None:
            argv += ["--custom-weight-loader", SPARSE_LOADER]
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
        self._served_dir = Path(release_dir) if release_dir is not None else Path(self.config.base_dir)
        self._sparse_maps = None  # a new server: probe again
        self._weight_sessions = None  # whether it has weight-update sessions: asked at the first call
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
        if self.sparse_off is None:
            detail["sparse_prepare"] = self._sparse_prepare()
        return detail

    def reload_in_place(self, model_dir: Path, version: int) -> dict[str, Any] | None:
        sparse: dict[str, Any] = {}
        if self.sparse_off is None and self._sparse_maps is not None:
            try:
                result = self._sparse_update(Path(model_dir), int(version))
            except sparse_update.SparseUnavailable as error:
                sparse = {"sparse_fallback": error.reason, "sparse_fallback_detail": str(error)[:300]}
            else:  # written sparsely, or reloaded in full after a failed write or check
                self._served_dir = Path(model_dir)
                return result
        elif self.options.sparse_update != "off":
            sparse = {"sparse_fallback": self.sparse_off or "not_prepared"}
        detail = self._reload_full(model_dir, version)
        self._served_dir = Path(model_dir)
        merged = {**(detail or {}), **sparse}
        return merged or None

    def _reload_full(self, model_dir: Path, version: int) -> dict[str, Any] | None:
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

    # ------------------------------------------------------------------ sparse updates
    def _sparse_root(self) -> Path:
        if self.options.sparse_dir:
            return Path(self.options.sparse_dir)
        import tempfile

        shm = Path("/dev/shm")
        return (shm if shm.is_dir() else Path(tempfile.gettempdir())) / f"yeto-sparse-{self.config.port}"

    def _sparse_call(self, commands: list[dict[str, Any]], *, version: int | None = None,
                     timeout_s: float | None = None) -> list[dict[str, Any]]:
        """Send one command per rank through the custom loader; the ranks' results."""
        for command in commands:
            Path(command["result"]).unlink(missing_ok=True)
        body: dict[str, Any] = {"serialized_named_tensors": [_sparse_payload(c) for c in commands],
                                "load_format": SPARSE_LOADER, "flush_cache": False}
        if version is not None:
            body["weight_version"] = str(int(version))
        session = self._begin_weight_update()
        try:
            reply = self.call("POST", "/update_weights_from_tensor", body, timeout_s=timeout_s)
        finally:
            if session:
                self.call("POST", "/end_weight_update", {})
        if isinstance(reply, dict) and reply.get("success") is False:
            raise SamplerEngineError(f"sglang update_weights_from_tensor: {reply.get('message')}")
        results = []
        for command in commands:
            try:
                results.append(json.loads(Path(command["result"]).read_text(encoding="utf-8")))
            except (OSError, ValueError):
                results.append({"ok": False, "error": f"no result from rank {command['rank']} "
                                f"(is yeto importable by the server's python?)"})
        return results

    def _begin_weight_update(self) -> bool:
        """Open a weight-update session where the server has them (the pinned
        fork: ``/update_weights_from_tensor`` outside a ``/begin_weight_update``
        session fails an assertion that ends the scheduler). False: the server has
        no sessions (upstream SGLang; remembered until the next start)."""
        if self._weight_sessions is False:
            return False
        try:
            self.call("POST", "/begin_weight_update", {})
        except SamplerEngineError as error:
            if self._weight_sessions is None and "HTTP 404" in str(error):
                self._weight_sessions = False
                return False
            raise
        self._weight_sessions = True
        return True

    def _sparse_prepare(self) -> dict[str, Any]:
        """Probe the element maps of the started server (module doc). On any
        failure sparse updates stay off for this server (``sparse_off``)."""
        assert self._served_dir is not None
        root, tp = self._sparse_root(), int(self.options.tp)
        began = self._clock()
        record: dict[str, Any] = {}
        try:
            slots = sparse_update.release_slots(self._served_dir)
            if not slots:
                raise SamplerEngineError(f"{self._served_dir}: no *.safetensors files")
            shapes = {name: list(slot.shape) for name, slot in slots.items()}
            maps_dir = root / "maps"
            maps_dir.mkdir(parents=True, exist_ok=True)
            commands = [{"op": "prepare", "rank": r, "tensors": shapes, "out": str(maps_dir / f"rank{r}.pkl"),
                         "result": str(maps_dir / f"prepare-rank{r}.json"),
                         "probe_device": self.options.sparse_probe_device,
                         "threads": max(1, (os.cpu_count() or 1) // tp)} for r in range(tp)]
            try:
                results = self._sparse_call(commands, timeout_s=self.options.sparse_prepare_timeout_s)
            finally:  # the probe may have touched real tensors: serve the release as loaded from disk
                try:
                    self._update_from_disk({"model_path": str(self._served_dir),
                                            "weight_version": str(self._started_version)})
                except BaseException as error:  # not a sparse failure: the served weights are unknown
                    raise _ProbeReloadFailed(f"reload after the sparse probe: {error}") from error
            failed = [r for r in results if not r.get("ok")]
            if failed:
                raise SamplerEngineError(f"sparse probe failed: {failed[0].get('error')}")
            maps = [sparse_update.load_map(Path(c["out"])) for c in commands]
            record = {"mapped": min(r["mapped"] for r in results),
                      "unmapped": max(r["unmapped"] for r in results), "probe_s": max(r["seconds"] for r in results)}
            mapped = set.intersection(*(set(m["tensors"]) for m in maps))
            rng = np.random.default_rng(int(self._started_version))
            checks = sparse_update.pick_checks(rng, slots, mapped, {}, tensors=4 * self.options.sparse_check_tensors,
                                               per_tensor=self.options.sparse_check_elements)
            summary = sparse_update.rank_patches(root / "check", maps, slots, {}, checks,
                                                 {"version": self._started_version})
            results = self._sparse_call([{"op": "check", "rank": r, "patch": summary[r]["dir"],
                                          "result": str(root / f"check-rank{r}.json")} for r in range(tp)])
            bad = [r for r in results if not r.get("ok") or r.get("mismatches")]
            record["checked"] = sum(int(r.get("checked") or 0) for r in results)
            if bad:
                raise SamplerEngineError(f"sparse layout check failed: {bad[0].get('error') or bad[0]}")
            self._sparse_maps = maps
            self._sparse_shapes = shapes
        except (SamplerEngineError, sparse_update.SparseUnavailable, OSError, ValueError, KeyError) as error:
            self._sparse_maps = None
            if isinstance(error, _ProbeReloadFailed):
                raise
            record["error"] = f"{type(error).__name__}: {str(error)[:300]}"
        record["prepare_s"] = round(self._clock() - began, 3)
        record["enabled"] = self._sparse_maps is not None
        self.sparse_prepare = record
        return record

    def _sparse_plan(self, model_dir: Path, version: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """The rank patches of the update to ``model_dir`` (written before any pause),
        or :class:`~.sparse_update.SparseUnavailable`."""
        from yeto.rl.engine.release_delta import MIRROR_STAMP, read_changes

        manifest = read_changes(model_dir)
        if manifest is None:
            raise sparse_update.SparseUnavailable("no_change_manifest", str(model_dir))
        served = self._served_dir
        if served is None or Path(manifest["base_dir"]).resolve() != served.resolve():
            raise sparse_update.SparseUnavailable("base_mismatch", f"rebuilt from {manifest['base_dir']}, "
                                                  f"serving {served}")
        if manifest.get("base_stamp") is not None:
            try:
                stamp = json.loads((served / MIRROR_STAMP).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                stamp = None
            if stamp != manifest["base_stamp"]:
                raise sparse_update.SparseUnavailable("base_mismatch", "the served copy is another release")
        slots = sparse_update.release_slots(model_dir)
        if {n: list(s.shape) for n, s in slots.items()} != self._sparse_shapes:
            raise sparse_update.SparseUnavailable("layout_changed", "tensor names or shapes differ")
        changes = sparse_update.changed_elements(model_dir, manifest["files"])
        changed = sum(int(i.size) for i in changes.values())
        total = sum(s.numel for s in slots.values())
        ratio = changed / total if total else 1.0
        if ratio > float(self.options.sparse_max_ratio):
            raise sparse_update.SparseUnavailable("too_many_changes", f"{ratio:.4f} > {self.options.sparse_max_ratio}")
        maps = self._sparse_maps or []
        rng = np.random.default_rng(int(version))
        mapped = set.intersection(*(set(m["tensors"]) for m in maps))
        checks = sparse_update.pick_checks(rng, slots, mapped, changes, tensors=self.options.sparse_check_tensors,
                                           per_tensor=self.options.sparse_check_elements)
        summary = sparse_update.rank_patches(self._sparse_root() / "patch", maps, slots, changes, checks,
                                             {"version": int(version)})
        return summary, {"changed_elements": changed, "changed_tensors": len(changes),
                         "changed_ratio": round(ratio, 6)}

    def _sparse_update(self, model_dir: Path, version: int) -> dict[str, Any]:
        began = self._clock()
        try:  # may raise SparseUnavailable: nothing touched
            summary, detail = self._sparse_plan(model_dir, version)
        except (OSError, ValueError, KeyError) as error:  # e.g. /dev/shm full: still nothing touched
            raise sparse_update.SparseUnavailable("plan_error", f"{type(error).__name__}: {error}") from error
        planned = self._clock()
        root = self._sparse_root()
        commands = [{"op": "apply", "rank": r, "patch": s["dir"], "result": str(root / f"apply-rank{r}.json")}
                    for r, s in enumerate(summary)]
        live = bool(self.options.live_update)
        if live:
            self.call("POST", "/pause_generation", {"mode": "retract"})
        paused = self._clock()
        try:
            try:
                results = self._sparse_call(commands, version=version)
            except SamplerEngineError as error:
                results = [{"ok": False, "error": str(error)}]
        finally:
            updated = self._clock()
            resumed_error = None
            if live:
                try:
                    self.call("POST", "/continue_generation", {})
                except SamplerEngineError as error:  # the patch may be partly written: repair below
                    resumed_error = error
        if resumed_error is not None:
            results = [{"ok": False, "error": f"continue_generation after the sparse write: {resumed_error}"}]
        detail.update({"plan_s": round(planned - began, 3), "update_s": round(updated - paused, 3),
                       "patch_elements": sum(s["elements"] for s in summary),
                       "checked": sum(int(r.get("checked") or 0) for r in results),
                       "write_s": max(float(r.get("write_s") or 0.0) for r in results)})
        if live:
            detail.update({"live": True, "paused_s": round(self._clock() - paused, 3)})
        bad = [r for r in results if not r.get("ok") or r.get("mismatches")]
        if not bad:
            return {"sparse": True, **detail}
        # the weights may be partly written: load the whole release now
        reason = bad[0].get("error") or f"{bad[0].get('mismatches')} check mismatches {bad[0].get('examples')}"
        repaired = self._reload_full(model_dir, version) or {}
        return {"sparse": False, "sparse_repair": str(reason)[:300], **{f"sparse_{k}": v for k, v in detail.items()},
                **repaired}

    def _remove_sparse_dir(self) -> None:
        if self.sparse_off is None:  # element maps and patches (in /dev/shm by default)
            import shutil

            shutil.rmtree(self._sparse_root(), ignore_errors=True)

    def stop(self) -> None:
        super().stop()
        self._remove_sparse_dir()

    def _kill(self) -> None:
        super()._kill()
        self._remove_sparse_dir()

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
        """Token-id prompts (``input_ids``), for the records proxy."""
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

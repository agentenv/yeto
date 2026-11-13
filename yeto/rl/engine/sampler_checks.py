"""Load a release through the ``SamplerEngine`` port and check what the engine
serves before it may generate.

A load that returns without error does not prove that the engine serves the
right weights: the reports list loads that skip expert weights or misread them
silently (A-INF items 1, 2, 3, 5). :class:`PortReleaseLoader` is the
``PolicyFollower`` load hook of an engine sampler island. For each release it:

1. refuses a release whose scheme is not the one the engine serves
   (:class:`~.policy_release.ReleaseNotLoaded`, reason ``scheme_mismatch``;
   the engine keeps its version);
2. counts the routed-expert matrices in the release index and compares them
   with the expected count (release record, else the check config);
   an FP8 release (``fp8-w8a8-channel``, scoped tier) must quantize only the
   in-scope linears (:func:`yeto.rl.quant.tiers.check_release_scope`);
3. loads the release (``engine.load``; a retry after a failed check restarts
   the engine with ``engine.start``);
4. runs the three post-load checks of :class:`PostLoadChecks`:

   * ``expert_count``: the engine's loaded expert-matrix count equals the
     expected count (GLM5.3-Flash: 36,288); when the engine cannot tell, its
     weight memory per card is compared with the expected value; when neither
     is known the check fails unless ``allow_unverified_count`` is set (and
     the tape says so);
   * ``fixed_prompts``: greedy answers to fixed prompts (4 GSM8K questions by
     default) are non-empty, contain the expected answer, carry no forbidden
     marker and almost no unreadable characters;
   * ``logprob_probe``: teacher-forced log-probs of fixed continuations; the
     mean absolute change against the previous version (or against a stored
     round-0 reference) must not exceed ``max_probe_jump`` (None: recorded only;
     the threshold is set after the first run on cards, A-INF §3);

5. checks that the engine reports the release version (``served_version``).

Only then is the release served. A failed step writes
``sampler_load_check_failed`` and raises ``ReleaseNotLoaded`` so the follower
tries again; after ``max_consecutive_failures`` failures in a row it raises
:class:`SamplerLoadFatal` and the island stops with the reason. A step that
failed after the engine was touched (load, post-load checks) marks the engine
not ready (the loop generates nothing) and the next load restarts it. A
release refused before the engine was touched (scheme, release expert count)
leaves the engine serving its checked version (``ready`` unchanged; the event
carries ``engine_touched: false`` and ``kept_version``).
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from yeto.rl.quant.schemes import FP8_FFN_CHANNEL

from .policy_release import LORA, PolicyRelease, ReleaseNotLoaded
from .release_overlay import is_partial_release, overlay_release
from .sampler_engine import SamplerEngine, SamplerEngineError, SamplingParams

# Routed-expert matrices of GLM5.3-Flash as the GLM inference report counts them.
GLM53_FLASH_EXPERT_MATRICES = 36_288
# Names of routed-expert weight tensors in an HF / compressed-tensors index:
# "...mlp.experts.<e>.<gate|up|down>_proj.weight" or ".weight_packed".
EXPERT_WEIGHT_PATTERN = r"\.experts\.\d+\.\w+_proj\.weight(?:_packed)?$"
INDEX_FILE = "model.safetensors.index.json"
BF16 = "bf16"

LOAD_CHECK_FAILED_EVENT = "sampler_load_check_failed"
EXPERT_SELECTION_SHA256 = "expert_selection_sha256"  # release manifest key (INTERFACES 2.3)
# manifest["partial_experts"]["trainable_sha256"]: the trainable expert sets alone, independent of
# the trainer's EP slot order (release tensors are named by original expert id)
TRAINABLE_SHA256 = "trainable_sha256"
LOAD_CHECKED_EVENT = "rl_sampler_load_checked"


class SamplerLoadFatal(RuntimeError):
    """Post-load checks failed too often; the island must stop."""


@dataclass(frozen=True)
class FixedPrompt:
    prompt: str
    answer: str  # must appear in the greedy output


GSM8K_FIXED_PROMPTS = (
    FixedPrompt("Question: Natalia sold clips to 48 of her friends in April, and then she sold "
                "half as many clips in May. How many clips did Natalia sell altogether in April "
                "and May?\nAnswer:", "72"),
    FixedPrompt("Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 "
                "minutes of babysitting. How much did she earn?\nAnswer:", "10"),
    FixedPrompt("Question: Betty is saving money for a new wallet which costs $100. Betty has only "
                "half of the money she needs. Her parents decided to give her $15 for that "
                "purpose, and her grandparents twice as much as her parents. How much more money "
                "does Betty need to buy the wallet?\nAnswer:", "5"),
    FixedPrompt("Question: Julie is reading a 120-page book. Yesterday, she was able to read 12 "
                "pages and today, she read twice as many pages as yesterday. If she wants to read "
                "half of the remaining pages tomorrow, how many pages should she read?\nAnswer:",
                "42"),
)


def release_scheme(release: PolicyRelease | None) -> str:
    """The scheme of a release: ``derived_from.scheme`` of a derived release, else bf16."""
    derived = None if release is None else release.derived_from
    return str((derived or {}).get("scheme") or BF16)


def overlaid_release_scheme(release: PolicyRelease, directory: Path, base_dir: Any, served: str) -> str:
    """The scheme a release is served in. A partial-experts release (the Kimi-K3.1
    recipe: ``manifest.partial_experts``, no ``derived_from``) carries the trained
    experts in the base checkpoint's own tensor format and is laid over that base
    (:func:`.release_overlay.overlay_release`), so it is served in the base's
    scheme when its ``quantization_config`` is the base's; every other release
    keeps :func:`release_scheme` (trained weights: bf16)."""
    if release.weights == LORA and release.derived_from is None and base_dir is not None:
        # an adapter merged into the engine's base (lora_release_merge): only BF16 modules
        # change, every quantized tensor stays the base's, so the base's scheme is served
        return served
    if release.derived_from is not None or not is_partial_release(release.manifest) or base_dir is None:
        return release_scheme(release)
    released, based = checkpoint_quant_format(Path(directory)), checkpoint_quant_format(Path(base_dir))
    if released[0] is not None and released == based:
        return served
    return release_scheme(release)


def count_release_experts(directory: Path, pattern: str = EXPERT_WEIGHT_PATTERN) -> int | None:
    """Routed-expert weight tensors named in the release's safetensors index
    (None: the release has no index file)."""
    index = Path(directory) / INDEX_FILE
    if not index.is_file():
        return None
    weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map") or {}
    matcher = re.compile(pattern)
    return sum(1 for name in weight_map if matcher.search(name))


def release_quant_format(directory: Path) -> tuple[str | None, str | None]:
    """``(quant_method, format)`` of ``config.json``'s ``quantization_config``
    (``(None, None)``: no quantization config, a BF16 checkpoint)."""
    config = Path(directory) / "config.json"
    if not config.is_file():
        return None, None
    quant = json.loads(config.read_text(encoding="utf-8")).get("quantization_config") or {}
    method = quant.get("quant_method")
    fmt = quant.get("format")
    return (None if method is None else str(method)), (None if fmt is None else str(fmt))


def checkpoint_quant_format(directory: Path) -> tuple[str | None, str | None]:
    """:func:`release_quant_format`, also reading ``text_config.quantization_config``
    (a multimodal checkpoint such as Kimi-K3.1 nests the language model's config)."""
    top = release_quant_format(directory)
    if top[0] is not None:
        return top
    config = Path(directory) / "config.json"
    if not config.is_file():
        return top
    text = json.loads(config.read_text(encoding="utf-8")).get("text_config")
    quant = (text.get("quantization_config") if isinstance(text, dict) else None) or {}
    method, fmt = quant.get("quant_method"), quant.get("format")
    return (None if method is None else str(method)), (None if fmt is None else str(fmt))


def _unreadable_share(text: str) -> float:
    if not text:
        return 1.0
    bad = sum(1 for c in text if c == "�" or (ord(c) < 32 and c not in "\n\r\t"))
    return bad / len(text)


def _contains_answer(text: str, answer: str) -> bool:
    """A numeric answer must stand alone ("5" is not found in "15")."""
    if answer.isdigit():
        return re.search(rf"(?<![\d.]){answer}(?![\d])", text) is not None
    return answer in text


@dataclass
class PostLoadChecks:
    """The three post-load checks (module doc). Thresholds that are not known
    yet are None and only recorded."""

    expected_expert_matrices: int | None = GLM53_FLASH_EXPERT_MATRICES
    expected_weight_gib_per_card: float | None = None
    weight_gib_tolerance: float = 0.10
    allow_unverified_count: bool = False
    fixed_prompts: Sequence[FixedPrompt] = GSM8K_FIXED_PROMPTS
    fixed_max_tokens: int = 2048
    # False only for a model that cannot answer (a cut-down rehearsal model): a missing
    # answer is recorded, the other fixed-prompt problems still fail the check
    require_fixed_answers: bool = True
    forbidden_markers: Sequence[str] = ()
    max_unreadable_share: float = 0.05
    probe_pairs: Sequence[tuple[str, str]] | None = None  # None: fixed prompts + their answers
    max_probe_jump: float | None = None
    reference_probe: Sequence[float] | None = None  # round-0 per-pair mean log-probs
    _last_probe: list[float] | None = field(default=None, init=False, repr=False)
    _pending_probe: list[float] | None = field(default=None, init=False, repr=False)

    def pairs(self) -> list[tuple[str, str]]:
        if self.probe_pairs is not None:
            return list(self.probe_pairs)
        return [(p.prompt, f" {p.answer}") for p in self.fixed_prompts]

    # ------------------------------------------------------------------ checks
    def expert_count(self, engine: SamplerEngine, expected: int | None) -> dict[str, Any]:
        if expected is None:
            return {"name": "expert_count", "passed": True, "detail": "no expected count (bf16 engine)"}
        loaded = engine.loaded_expert_matrices()
        if loaded is not None:
            return {"name": "expert_count", "passed": int(loaded) == int(expected),
                    "loaded": int(loaded), "expected": int(expected)}
        if int(expected) == 0:
            # a dense model has no routed experts; the release index count (0) was
            # compared before the load, so there is nothing an engine report could add.
            return {"name": "expert_count", "passed": True, "expected": 0,
                    "detail": "no routed experts (dense model); release index count checked"}
        gib = engine.weight_gib_per_card()
        want = self.expected_weight_gib_per_card
        if gib is not None and want is not None:
            ok = abs(float(gib) - want) <= self.weight_gib_tolerance * want
            return {"name": "expert_count", "passed": ok, "weight_gib_per_card": float(gib),
                    "expected_weight_gib_per_card": want, "tolerance": self.weight_gib_tolerance}
        return {"name": "expert_count", "passed": bool(self.allow_unverified_count),
                "unverified": True, "expected": int(expected),
                "detail": ("the engine reports neither its expert count nor its weight memory; "
                           + ("allowed by allow_unverified_count" if self.allow_unverified_count
                              else "set expected_weight_gib_per_card or allow_unverified_count"))}

    def fixed_prompt_answers(self, engine: SamplerEngine) -> dict[str, Any]:
        if not self.fixed_prompts:
            return {"name": "fixed_prompts", "passed": True, "detail": "no fixed prompts"}
        greedy = SamplingParams(max_tokens=int(self.fixed_max_tokens), temperature=0.0, seed=0).check()
        groups = engine.generate([p.prompt for p in self.fixed_prompts], greedy)
        answers = []
        for item, group in zip(self.fixed_prompts, groups):
            text = group[0].text if group else ""
            problems = []
            if not text.strip():
                problems.append("empty")
            answered = _contains_answer(text, item.answer)
            if not answered and self.require_fixed_answers:
                problems.append("answer missing")
            if _unreadable_share(text) > self.max_unreadable_share:
                problems.append("unreadable characters")
            problems += [f"marker {m!r}" for m in self.forbidden_markers if m in text]
            answers.append({"answer": item.answer, "answered": answered, "ok": not problems,
                            "problems": problems, "text": text[:160]})
        return {"name": "fixed_prompts", "passed": all(a["ok"] for a in answers)
                and len(groups) == len(self.fixed_prompts), "answers": answers,
                "answers_required": bool(self.require_fixed_answers)}

    def logprob_probe(self, engine: SamplerEngine) -> dict[str, Any]:
        if not self.pairs():  # no fixed prompts and no probe pairs: nothing to score or compare
            return {"name": "logprob_probe", "passed": True, "detail": "no probe pairs"}
        means = []
        for prompt, continuation in self.pairs():
            logprobs = engine.score(prompt, continuation)
            if not logprobs or not all(math.isfinite(x) for x in logprobs):
                return {"name": "logprob_probe", "passed": False,
                        "detail": f"empty or non-finite log-probs for {continuation!r}"}
            means.append(sum(logprobs) / len(logprobs))
        against, label = ((self.reference_probe, "reference") if self.reference_probe is not None
                          else (self._last_probe, "previous"))
        out: dict[str, Any] = {"name": "logprob_probe", "means": [round(m, 6) for m in means],
                               "against": label if against is not None else None}
        jump = None
        if against is not None:
            if len(against) != len(means):
                return {**out, "passed": False, "detail": "probe pair count differs from the reference"}
            jump = sum(abs(a - b) for a, b in zip(means, against)) / len(means)
            out["mean_abs_jump"] = round(jump, 6)
        out["max_probe_jump"] = self.max_probe_jump
        out["passed"] = jump is None or self.max_probe_jump is None or jump <= self.max_probe_jump
        self._pending_probe = means if out["passed"] else None
        return out

    def accept(self) -> None:
        """Make the last probe the next comparison base; only after every check of the load passed
        (a probe of weights that failed another check must not become the base)."""
        if self._pending_probe is not None:
            self._last_probe, self._pending_probe = self._pending_probe, None

    def run(self, engine: SamplerEngine, *, expected_experts: int | None) -> list[dict[str, Any]]:
        results = [self.expert_count(engine, expected_experts)]
        results.append(self.fixed_prompt_answers(engine))
        results.append(self.logprob_probe(engine))
        return results


class PortReleaseLoader:
    """``PolicyFollower`` load hook over a :class:`SamplerEngine` (module doc).

    ``emit(event, **fields)`` writes the tape. ``expected_experts`` for a release
    is ``derived_from.expert_matrices`` when the release records it, else the
    check config's ``expected_expert_matrices`` (bf16 engines: None)."""

    def __init__(self, engine: SamplerEngine, checks: PostLoadChecks, emit: Callable[..., None], *,
                 max_consecutive_failures: int = 3,
                 clock: Callable[[], float] = time.monotonic,
                 expected_selection_sha256: str | None = None,
                 expected_trainable_sha256: str | None = None) -> None:
        """``expected_selection_sha256``: the partial-experts selection this sampler
        is bound to (a Kimi-K3.1 run); a release whose record's ``manifest`` names
        no selection or another one is refused before the engine is touched.
        ``expected_trainable_sha256``: bind to the trainable expert sets instead
        (``manifest.partial_experts.trainable_sha256``), so a trainer that runs the same
        experts under another EP layout (another slot order, another ``sha256``) is
        accepted; when set it replaces the whole-selection comparison."""
        if int(max_consecutive_failures) < 1:
            raise ValueError("max_consecutive_failures must be >= 1")
        self.engine = engine
        self.expected_selection_sha256 = (None if expected_selection_sha256 is None
                                          else str(expected_selection_sha256).lower())
        self.expected_trainable_sha256 = (None if expected_trainable_sha256 is None
                                          else str(expected_trainable_sha256).lower())
        self.checks = checks
        self.emit = emit
        self.max_consecutive_failures = int(max_consecutive_failures)
        self.clock = clock
        self.ready = False  # True after a load whose checks all passed
        self.failures = 0
        self._restart = False  # a failed load or check left the engine in an unknown state
        self.unchecked_loads = 0  # loads or checks that failed after touching the engine (lifetime)
        self.last_load: dict[str, Any] | None = None  # reload_mode, pause_s, checks

    @property
    def scheme(self) -> str:
        return self.engine.identity.quant_scheme

    def _expected(self, release: PolicyRelease | None) -> int | None:
        recorded = None if release is None else (release.derived_from or {}).get("expert_matrices")
        if recorded is not None:
            return int(recorded)
        return None if self.scheme == BF16 else self.checks.expected_expert_matrices

    def start(self, directory: Path | None, version: int, release: PolicyRelease | None = None) -> None:
        """Start the engine (base weights when ``directory`` is None) and check it."""
        self._serve(release, directory, int(version), first=True)

    def __call__(self, release: PolicyRelease, directory: Path) -> None:
        directory = Path(directory)
        base = getattr(getattr(self.engine, "config", None), "base_dir", None)
        scheme = overlaid_release_scheme(release, directory, base, self.scheme)
        if scheme != self.scheme:
            raise ReleaseNotLoaded(
                f"release r{release.seq}-v{release.version} has scheme {scheme}; this engine "
                f"serves {self.scheme}; keeping v{self.engine.served_version()}",
                reason="scheme_mismatch")
        if self.expected_trainable_sha256 is not None:
            declared = ((release.manifest or {}).get("partial_experts") or {}).get(TRAINABLE_SHA256)
            if declared is None or str(declared).lower() != self.expected_trainable_sha256:
                self._refuse(int(release.version), "expert_selection",
                             {"release_trainable": None if declared is None else str(declared),
                              "expected_trainable": self.expected_trainable_sha256})
        elif self.expected_selection_sha256 is not None:
            declared = (release.manifest or {}).get(EXPERT_SELECTION_SHA256)
            if declared is None or str(declared).lower() != self.expected_selection_sha256:
                self._refuse(int(release.version), "expert_selection",
                             {"release": None if declared is None else str(declared),
                              "expected": self.expected_selection_sha256})
        if release.weights == LORA:
            directory = self._merged_lora(release, directory, base)
        elif is_partial_release(release.manifest) and base is not None:
            # a release of the trained experts only: lay it over the engine's base checkpoint,
            # as the role switch does for the release a sampler starts on (GPU-A it11b: the
            # follower started SGLang on the bare partial release, which lacks every other weight)
            directory = overlay_release(base, directory, directory.with_name(directory.name + ".overlay"))
        self._serve(release, directory, int(release.version), first=False)

    def _merged_lora(self, release: PolicyRelease, directory: Path, base: Any) -> Path:
        """A LoRA release on an engine that serves full checkpoints only: merge the
        adapter into the engine's base (:mod:`.lora_release_merge`). Refused before
        the engine is touched when there is no base or the adapter cannot be merged
        without re-quantizing (routed-expert LoRA over MXFP4 experts)."""
        from .lora_release_merge import LoraMergeError, merge_lora_release

        version = int(release.version)
        if base is None:
            self._refuse(version, "lora_merge", "a LoRA release needs the engine's base_dir to merge into")
        declared = (release.manifest or {}).get("lora") or {}
        if declared.get("routed_experts"):
            self._refuse(version, "lora_merge", "the adapter targets the routed experts, which only a LoRA "
                         "bypass engine serves (merging re-quantizes the MXFP4 experts); train with "
                         "--lora-targets attention for this sampler")
        try:
            return merge_lora_release(base, directory, directory.with_name(directory.name + ".merged"))
        except (LoraMergeError, OSError) as error:
            self._refuse(version, "lora_merge", str(error))
        raise AssertionError("unreachable")

    def _fail(self, version: int, step: str, detail: Any) -> None:
        """A load or check failed: the engine holds unchecked weights, so it stops serving."""
        self.ready = False
        self._restart = True
        self.unchecked_loads += 1
        self._count_failure(version, step, detail, engine_touched=True)
        raise ReleaseNotLoaded(f"v{version}: {step} failed: {str(detail)[:300]}",
                               reason=f"check_failed:{step}")

    def _refuse(self, version: int, step: str, detail: Any) -> None:
        """The release failed a check before the engine was touched: the engine
        keeps serving its checked version (``ready`` is unchanged)."""
        kept = self.engine.served_version() if self.ready else None
        self._count_failure(version, step, detail, engine_touched=False, kept_version=kept)
        raise ReleaseNotLoaded(f"v{version}: {step} failed: {str(detail)[:300]}; "
                               f"keeping v{kept}", reason=f"release_refused:{step}")

    def _count_failure(self, version: int, step: str, detail: Any, **fields: Any) -> None:
        self.failures += 1
        self.emit(LOAD_CHECK_FAILED_EVENT, policy_version=version, step=step, detail=detail,
                  failures=self.failures, max_consecutive_failures=self.max_consecutive_failures,
                  **fields)
        if self.failures >= self.max_consecutive_failures:
            raise SamplerLoadFatal(f"post-load checks failed {self.failures} times in a row "
                                   f"(last: {step} for v{version}): {str(detail)[:300]}")

    def _serve(self, release: PolicyRelease | None, directory: Path | None, version: int,
               *, first: bool) -> None:
        expected = self._expected(release)
        if directory is not None and expected is not None:
            in_release = count_release_experts(directory)
            if in_release is not None and in_release != expected:
                self._refuse(version, "release_expert_count",
                             {"in_release": in_release, "expected": expected})
        if directory is not None and self.scheme == FP8_FFN_CHANNEL:
            # only in-scope linears may be FP8 (yeto.rl.quant.scope)
            from yeto.rl.quant.scope import QuantScopeError
            from yeto.rl.quant.tiers import check_release_scope

            try:
                check_release_scope(directory)
            except QuantScopeError as error:
                self._refuse(version, "quant_scope", str(error))
        restart = first or self._restart  # after a failed load or check: restart, never patch in place
        self.ready = False  # the engine holds unchecked weights from here until the checks pass
        started = self.clock()
        try:
            report = (self.engine.start(directory, version) if restart
                      else self.engine.load(directory, version))
        except SamplerEngineError as error:
            self._fail(version, "load", f"{type(error).__name__}: {error}")
        try:
            results = self.checks.run(self.engine, expected_experts=expected)
        except SamplerEngineError as error:  # an engine call of a check failed: a failed check
            self._fail(version, "checks", f"{type(error).__name__}: {error}")
        served = self.engine.served_version()
        results.append({"name": "served_version", "passed": served == version,
                        "served": served, "expected": version})
        failed = [r for r in results if not r["passed"]]
        if failed:
            self._fail(version, failed[0]["name"], failed)
        self.checks.accept()
        self.ready = True
        self.failures = 0
        self._restart = False
        self.last_load = {"reload_mode": report.reload_mode, "pause_s": round(report.pause_s, 3),
                          "load_and_check_s": round(self.clock() - started, 3),
                          "scheme": self.scheme, "checks": results, **dict(report.detail)}
        self.emit(LOAD_CHECKED_EVENT, policy_version=version, **self.last_load)

    def weights_state(self) -> tuple[bool, int]:
        """(clean, ``unchecked_loads``): a streaming round drops a group whose trials
        ran across a load that failed after touching the engine (``AgenticRound.weights_state``).
        clean is False only while a failed load awaits its restart: a load in progress is not
        a taint (a live engine pauses its trials and continues on the new checked weights;
        should it fail, ``unchecked_loads`` moves and the groups that ran across it are dropped)."""
        # the count first: _fail sets _restart before it counts, so a reader on another
        # thread that races a failure sees either the old count (compared at settle) or the taint
        failed = self.unchecked_loads
        return not self._restart, failed

    def loaded_fields(self) -> dict[str, Any]:
        """Extra fields of ``rl_sampler_policy_loaded`` (``SamplerLoop.loaded_fields``)."""
        if self.last_load is None:
            return {}
        fields = {k: self.last_load[k] for k in ("reload_mode", "pause_s", "scheme")}
        # how the weights moved (a fast path, or why it fell back to a whole reload)
        fields.update({k: v for k, v in self.last_load.items() if k in LOAD_PATH_FIELDS})
        return fields


LOAD_PATH_FIELDS = ("expert_update", "expert_update_fallback", "expert_update_fallback_detail",
                    "expert_update_repair", "update_s", "files", "sparse", "sparse_fallback",
                    "sparse_fallback_detail", "sparse_repair", "memory_guard")


def check_sampling_fits(sampling: SamplingParams, engine: SamplerEngine, *,
                        max_prompt_tokens: int) -> None:
    """Refuse a run whose ``max_tokens`` this engine cannot serve: every source
    must truncate at the same length (A-INF 5.1), so an engine with a lower
    limit may not silently cut earlier."""
    need = int(sampling.max_tokens) + int(max_prompt_tokens)
    if need > int(engine.max_model_len):
        raise SamplerEngineError(
            f"{engine.identity.engine} serves max_model_len {engine.max_model_len}, but the run "
            f"needs max_tokens {sampling.max_tokens} + prompt {max_prompt_tokens} = {need}; raise "
            "the engine limit or lower max_tokens for every source of the run")


"""The ``SamplerEngine`` port: one interface for every inference engine a
sampler island drives (rl-hetero-sampler-engines D1, S25 N2.3).

The sampler island's loop, buffer writer, reward call and release follower are
engine-neutral. Everything that depends on the engine (process start, HTTP API,
weight reload, vendor patches and environment) lives in an adapter that
implements :class:`SamplerEngine`. Adapters: ``sglang-server`` (PPU, stage 1)
and ``vllm-server`` (vllm-ascend on 910B, stage 2), in
:mod:`yeto.rl.adapters.sampler_engines`. Tests use
:class:`yeto.rl.adapters.sampler_engines.fake.FakeEngine`.

The port has four kinds of operation:

* ``load(release_dir, version)``: serve the weights of a release. The adapter
  reloads in place or restarts the engine; :class:`LoadReport` says which and
  how long the engine did not serve (``reload_mode``, ``pause_s``);
* ``generate(prompts, sampling)``: ``sampling.n`` completions per prompt, each
  with the per-token log-probabilities the engine sampled from;
* ``score(prompt, continuation)``: teacher-forced log-probabilities of a fixed
  continuation (the post-load log-prob probe, :mod:`.sampler_checks`);
* introspection for the post-load checks: ``served_version()``,
  ``loaded_expert_matrices()``, ``weight_gib_per_card()``.

Who uses it: :class:`yeto.rl.engine.sampler_checks.PortReleaseLoader` (the
``PolicyFollower`` load hook) and :mod:`yeto.rl.engine.engine_sampler`
(the island's round). A records proxy (S25 N10.5) can wrap a ``SamplerEngine``
and forward every call; the port has no hidden state the proxy must copy.

Sampling parameters are the same for every source of a run (A-INF 5, guard
D11): :class:`SamplingParams` is built once from the run config, and
:meth:`SamplingParams.check` refuses repetition penalties, which change the
behavior distribution away from the model's own distribution.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# finish_reason of a completion. ``loop_abort``: the island cut a repeating
# completion (:mod:`.loop_guard`); the kept tokens are unchanged.
FINISH_STOP = "stop"
FINISH_LENGTH = "length"
FINISH_LOOP_ABORT = "loop_abort"
FINISH_REASONS = (FINISH_STOP, FINISH_LENGTH, FINISH_LOOP_ABORT)

# reload_mode of a LoadReport.
RELOAD_IN_PLACE = "in_place"
RELOAD_RESTART = "restart"
RELOAD_MODES = (RELOAD_IN_PLACE, RELOAD_RESTART)

# Neutral values of the repetition penalties (no change to the distribution).
NEUTRAL_PENALTIES = {"repetition_penalty": 1.0, "frequency_penalty": 0.0, "presence_penalty": 0.0}


class SamplerEngineError(RuntimeError):
    """An engine call failed (start, load, generate, score)."""


class SamplingConfigError(ValueError):
    """The sampling parameters are not allowed for a rollout (guard D11)."""


@dataclass(frozen=True)
class SamplingParams:
    """Sampling parameters of one run, identical for every sampler source.

    ``max_tokens`` comes from the task config, never from an adapter default:
    a source with a lower limit would truncate more and bias the per-source
    reward (A-INF 5.1). ``allow_penalties`` must be set explicitly (and is
    written to the tape) to use a non-neutral penalty."""

    max_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1  # -1: no top-k limit
    n: int = 1
    seed: int | None = None
    stop: tuple[str, ...] = ()
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    allow_penalties: bool = False

    def check(self) -> "SamplingParams":
        if int(self.max_tokens) < 1:
            raise SamplingConfigError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if int(self.n) < 1:
            raise SamplingConfigError(f"n must be >= 1, got {self.n}")
        if not self.allow_penalties:
            changed = {name: getattr(self, name) for name, neutral in NEUTRAL_PENALTIES.items()
                       if float(getattr(self, name)) != neutral}
            if changed:
                raise SamplingConfigError(
                    f"rollout sampling sets {changed}; repetition penalties change the behavior "
                    "distribution and engines differ in whether the returned log-probs include "
                    "them. Use the loop guard (finish_reason loop_abort) instead, or set "
                    "allow_penalties explicitly (recorded in the tape)")
        return self

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["stop"] = list(self.stop)
        return out

    def digest(self) -> str:
        """Short digest of the parameters; equal on every source of a run."""
        text = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def replace(self, **changes: Any) -> "SamplingParams":
        return SamplingParams(**{**{f.name: getattr(self, f.name) for f in fields(self)}, **changes})


@dataclass(frozen=True)
class Completion:
    """One completion: response tokens, the log-probability of each sampled
    token (same length), the response text and why it ended.
    ``prompt_token_ids``: the prompt as the engine tokenized it (empty when the
    engine does not return it; the trainer then tokenizes the prompt text)."""

    token_ids: tuple[int, ...]
    logprobs: tuple[float, ...]
    text: str
    finish_reason: str
    prompt_token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if len(self.token_ids) != len(self.logprobs):
            raise SamplerEngineError(f"completion has {len(self.token_ids)} tokens but "
                                     f"{len(self.logprobs)} log-probs")
        if self.finish_reason not in FINISH_REASONS:
            raise SamplerEngineError(f"unknown finish_reason {self.finish_reason!r} "
                                     f"(known: {FINISH_REASONS})")


@dataclass(frozen=True)
class LoadReport:
    """What one ``load`` did. ``pause_s``: seconds the engine served no request."""

    reload_mode: str
    pause_s: float
    detail: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.reload_mode not in RELOAD_MODES:
            raise SamplerEngineError(f"unknown reload_mode {self.reload_mode!r}")


@dataclass(frozen=True)
class EngineIdentity:
    """The source facts of an engine (the provenance of its samples).

    ``precision``: the weight precision class (``bf16``, ``w4a8``, ``mxfp4``...);
    ``quant_scheme``: the release scheme the engine serves (``bf16`` for trained
    weights; else the ``derived_from.scheme`` of a derived release);
    ``topology``: the parallel layout of one engine instance (``tp8``, ``dp8``,
    ``tp4``...)."""

    engine: str
    engine_version: str
    device_family: str
    card_model: str
    precision: str
    quant_scheme: str
    topology: str


@runtime_checkable
class SamplerEngine(Protocol):
    """The port (module doc). Every method may raise :class:`SamplerEngineError`."""

    @property
    def identity(self) -> EngineIdentity: ...

    @property
    def max_model_len(self) -> int:
        """Prompt plus response token limit of the served model."""
        ...

    def start(self, release_dir: Path | None, version: int) -> LoadReport:
        """Start the engine on a release (None: the base weights it was configured with)."""
        ...

    def load(self, release_dir: Path, version: int) -> LoadReport:
        """Serve the weights of ``release_dir`` as policy ``version``."""
        ...

    def served_version(self) -> int | None:
        """The policy version the engine serves (None: not started)."""
        ...

    def generate(self, prompts: Sequence[str], sampling: SamplingParams) -> list[list[Completion]]:
        """``sampling.n`` completions for each prompt, in prompt order."""
        ...

    def score(self, prompt: str, continuation: str) -> list[float]:
        """Teacher-forced log-probabilities of the continuation tokens."""
        ...

    def loaded_expert_matrices(self) -> int | None:
        """Quantized routed-expert matrices the engine loaded (None: it cannot tell)."""
        ...

    def weight_gib_per_card(self) -> float | None:
        """Weight memory per card after the load (None: it cannot tell)."""
        ...

    def stop(self) -> None:
        """Stop the engine (idempotent)."""
        ...


# ---------------------------------------------------------------------- provenance
UNKNOWN = "unknown"


@dataclass(frozen=True)
class Provenance:
    """The source of a buffer entry (rl-hetero-sampler-engines D3, S25 N2.4).

    Data only: it never enters ``algorithm_sha256`` or the session contract
    hash. The behavior policy version stays in the entry's ``behavior_version``.
    An entry written before this field exists reads as :meth:`unknown`."""

    device_family: str
    card_model: str
    engine: str
    engine_version: str
    precision: str
    quant_scheme: str
    topology: str = UNKNOWN
    sampler_id: str = UNKNOWN
    sampling_digest: str = UNKNOWN

    # The per-source metrics group by these fields (D4, A-INF 4.1).
    KEY_FIELDS = ("device_family", "card_model", "engine", "engine_version", "precision",
                  "topology")

    @classmethod
    def unknown(cls) -> "Provenance":
        return cls(*([UNKNOWN] * 6))

    @classmethod
    def of(cls, identity: EngineIdentity, *, sampler_id: str, sampling: SamplingParams) -> "Provenance":
        return cls(device_family=identity.device_family, card_model=identity.card_model,
                   engine=identity.engine, engine_version=identity.engine_version,
                   precision=identity.precision, quant_scheme=identity.quant_scheme,
                   topology=identity.topology, sampler_id=str(sampler_id),
                   sampling_digest=sampling.digest())

    @classmethod
    def from_record(cls, record: Mapping[str, Any] | None) -> "Provenance":
        """The provenance of an index record field; missing or partial -> ``unknown``."""
        if not isinstance(record, Mapping):
            return cls.unknown()
        names = [f.name for f in fields(cls)]
        return cls(**{name: str(record.get(name, UNKNOWN)) for name in names})

    def to_record(self) -> dict[str, str]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def key(self) -> str:
        """The grouping key as one string, ``family/card/engine@version/precision/topology``."""
        return (f"{self.device_family}/{self.card_model}/{self.engine}@{self.engine_version}/"
                f"{self.precision}/{self.topology}")

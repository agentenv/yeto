"""A fake :class:`~yeto.rl.engine.sampler_engine.SamplerEngine` for CPU tests.

Deterministic: the token ids and log-probs of a completion depend only on the
prompt, the sample index and the served version. Faults are injected by
attribute (``expert_matrices``, ``answers``, ``probe_shift``, ``fail_load``).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from yeto.rl.engine.sampler_engine import (
    FINISH_LENGTH,
    FINISH_STOP,
    RELOAD_IN_PLACE,
    RELOAD_RESTART,
    Completion,
    EngineIdentity,
    LoadReport,
    SamplerEngineError,
    SamplingParams,
)

FAKE_IDENTITY = EngineIdentity(engine="fake", engine_version="0", device_family="cpu",
                               card_model="none", precision="bf16", quant_scheme="bf16",
                               topology="tp1")


def _unit(*parts: object) -> float:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


class FakeEngine:
    """In-memory engine. ``answers``: prompt -> greedy answer text (fixed-prompt
    check); other prompts get ``"fake answer"``. ``tokens_for``: response length
    per (prompt, sample index); a completion that reaches ``max_tokens`` ends
    with ``length``."""

    def __init__(self, *, identity: EngineIdentity = FAKE_IDENTITY, max_model_len: int = 4096,
                 in_place: bool = True, expert_matrices: int | None = 36_288,
                 weight_gib: float | None = None, answers: Mapping[str, str] | None = None,
                 tokens_for: Callable[[str, int], int] = lambda prompt, i: 4 + i,
                 token_ids_for: Callable[[str, int, int], list[int]] | None = None) -> None:
        self._identity = identity
        self._max_model_len = int(max_model_len)
        self.in_place = bool(in_place)
        self.expert_matrices = expert_matrices
        self.weight_gib = weight_gib
        self.answers = dict(answers or {})
        self.tokens_for = tokens_for
        self.token_ids_for = token_ids_for
        self.probe_shift = 0.0  # added to every score log-prob (a "silently wrong" load)
        self.fail_load: str | None = None  # raise this message on the next load
        self._version: int | None = None
        self.release_dir: Path | None = None
        self.loads: list[tuple[str | None, int]] = []
        self.requests: list[tuple[tuple[str, ...], SamplingParams]] = []
        self.stopped = False

    @property
    def identity(self) -> EngineIdentity:
        return self._identity

    @property
    def max_model_len(self) -> int:
        return self._max_model_len

    def start(self, release_dir: Path | None, version: int) -> LoadReport:
        self.stopped = False
        return self._serve(release_dir, version, RELOAD_RESTART)

    def load(self, release_dir: Path, version: int) -> LoadReport:
        return self._serve(release_dir, version, RELOAD_IN_PLACE if self.in_place else RELOAD_RESTART)

    def _serve(self, release_dir: Path | None, version: int, mode: str) -> LoadReport:
        if self.fail_load is not None:
            message, self.fail_load = self.fail_load, None
            raise SamplerEngineError(message)
        self.release_dir = None if release_dir is None else Path(release_dir)
        self._version = int(version)
        self.loads.append((None if release_dir is None else str(release_dir), int(version)))
        return LoadReport(reload_mode=mode, pause_s=0.0)

    def served_version(self) -> int | None:
        return self._version

    def generate(self, prompts: Sequence[str], sampling: SamplingParams) -> list[list[Completion]]:
        if self._version is None or self.stopped:
            raise SamplerEngineError("fake engine is not serving")
        self.requests.append((tuple(prompts), sampling))
        out = []
        for prompt in prompts:
            group = []
            for i in range(int(sampling.n)):
                if prompt in self.answers:
                    text = self.answers[prompt]
                    ids = [ord(c) for c in text][: sampling.max_tokens]
                else:
                    length = int(self.tokens_for(prompt, i))
                    ids = (self.token_ids_for(prompt, i, length) if self.token_ids_for
                           else [int(_unit(prompt, i, k) * 1000) for k in range(length)])
                    ids = ids[: sampling.max_tokens]
                    text = "fake answer"
                finish = FINISH_LENGTH if len(ids) >= sampling.max_tokens else FINISH_STOP
                logprobs = tuple(-_unit(prompt, i, k, self._version) for k in range(len(ids)))
                group.append(Completion(token_ids=tuple(ids), logprobs=logprobs, text=text,
                                        finish_reason=finish))
            out.append(group)
        return out

    def score(self, prompt: str, continuation: str) -> list[float]:
        if self._version is None:
            raise SamplerEngineError("fake engine is not serving")
        return [-_unit(prompt, continuation, k) - self.probe_shift for k in range(len(continuation))]

    def loaded_expert_matrices(self) -> int | None:
        return self.expert_matrices

    def weight_gib_per_card(self) -> float | None:
        return self.weight_gib

    def stop(self) -> None:
        self.stopped = True

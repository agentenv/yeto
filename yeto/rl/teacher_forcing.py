"""Teacher-forcing replay for the legacy-vs-ports equivalence check (rl-engine-ports 6.2/6.3 tier 2).

``replay_generate`` is a Miles ``--custom-generate-function-path`` callable
(legacy signature ``(args, sample, sampling_params)``, accepted by the legacy
fork directly and by upstream Miles through ``LegacyGenerateFnAdapter``).
Instead of sampling from SGLang it fills each prompt's sample with the
response recorded by a previous run's ``--save-debug-rollout-data`` dump
(``rollouts/island-<i>/<rollout_id>.pt``, written by ``scripts/benchmark_rl.py``).
Both engines then train on byte-identical tokens, so their loss / grad_norm /
LoRA update differ only by trainer numerics.

Configuration (environment, inherited by the Ray rollout process):

* ``YETO_RL_REPLAY_ROLLOUTS`` -- one or more globs (``os.pathsep``-separated)
  of recorded dumps, e.g. ``/work/ref/rollouts/island-*/0.pt``.

A prompt that is not in the recording raises: generation is never silently
substituted. The policy token is re-stamped for the running engine (legacy:
``weight_versions=[<token>]``; ports: the token the driver published to the
metadata sink), so the engines' own policy filters accept the replayed groups.

Import-light: torch / miles are imported lazily inside the rollout process.
"""

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from typing import Any

REPLAY_ENV = "YETO_RL_REPLAY_ROLLOUTS"
REPLAY_FIELDS = (
    "tokens",
    "response",
    "response_length",
    "loss_mask",
    "rollout_log_probs",
    "reward",
)


class ReplayMiss(RuntimeError):
    """The running rollout asked for a prompt that the recording does not contain."""


def prompt_key(prompt: Any) -> str:
    return prompt if isinstance(prompt, str) else json.dumps(prompt, sort_keys=True)


class ReplayIndex:
    """Recorded samples keyed by (prompt, sample index), with an occurrence fallback."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.exact: dict[tuple[str, Any], dict[str, Any]] = {}
        self.by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for rec in records:
            key = prompt_key(rec.get("prompt", ""))
            exact = (key, rec.get("index"))
            if rec.get("index") is not None and exact in self.exact:
                raise ValueError(f"duplicate recorded sample index {rec.get('index')!r} for one prompt")
            if rec.get("index") is not None:
                self.exact[exact] = rec
            self.by_prompt[key].append(rec)
        self.used: set[int] = set()
        self.hits = {"exact": 0, "occurrence": 0}

    def __len__(self) -> int:
        return sum(len(v) for v in self.by_prompt.values())

    def take(self, prompt: Any, index: Any) -> dict[str, Any]:
        key = prompt_key(prompt)
        rec = self.exact.get((key, index))
        if rec is not None and id(rec) not in self.used:
            self.used.add(id(rec))
            self.hits["exact"] += 1
            return rec
        for rec in self.by_prompt.get(key, ()):
            if id(rec) not in self.used:
                self.used.add(id(rec))
                self.hits["occurrence"] += 1
                return rec
        raise ReplayMiss(
            f"teacher forcing: prompt (sample index {index!r}) is not in the recording or all "
            f"{len(self.by_prompt.get(key, ()))} recorded responses were used; refusing to generate"
        )


def load_records(patterns: str) -> list[dict[str, Any]]:
    import torch

    paths = sorted({p for pat in patterns.split(os.pathsep) if pat for p in glob.glob(pat)})
    if not paths:
        raise FileNotFoundError(f"{REPLAY_ENV}={patterns!r} matched no recorded rollout dumps")
    records: list[dict[str, Any]] = []
    for path in paths:
        dump = torch.load(path, map_location="cpu", weights_only=False)
        records.extend(dump["samples"])
    return records


_INDEX: ReplayIndex | None = None


def _index() -> ReplayIndex:
    global _INDEX
    if _INDEX is None:
        patterns = os.environ.get(REPLAY_ENV)
        if not patterns:
            raise RuntimeError(f"teacher forcing requires {REPLAY_ENV}")
        _INDEX = ReplayIndex(load_records(patterns))
    return _INDEX


def _ports_policy_token() -> str | None:
    try:
        from yeto.rl.engine.miles_adapter.rollout_meta_hook import current_policy_token

        return current_policy_token()
    except Exception:  # noqa: BLE001 - legacy env has no sink
        return None


def _legacy_policy_token(args: Any) -> str | None:
    version = getattr(args, "yeto_rl_policy_version", None)
    if version is None:
        return None
    from yeto.rl.miles import _policy_token_for_rollout

    return _policy_token_for_rollout(args, version)


def apply_record(sample: Any, rec: dict[str, Any], *, ports_token: str | None,
                 legacy_token: str | None) -> Any:
    if prompt_key(rec.get("prompt", "")) != prompt_key(getattr(sample, "prompt", "")):
        raise ReplayMiss("teacher forcing: recorded prompt does not match the running sample")
    for name in REPLAY_FIELDS:
        if name in rec:
            setattr(sample, name, rec[name])
    status = rec.get("status")
    if status is not None:
        cls = type(sample.status)
        sample.status = cls(getattr(status, "value", status))
    sample.metadata = dict(getattr(sample, "metadata", None) or {})
    sample.metadata["yeto_teacher_forced"] = True
    if ports_token is not None:
        sample.weight_versions = []
        sample.metadata["weight_version"] = ports_token
    elif legacy_token is not None:
        sample.weight_versions = [legacy_token]
    elif "weight_versions" in rec:
        sample.weight_versions = rec["weight_versions"]
    return sample


async def replay_generate(args: Any, sample: Any, sampling_params: Any, evaluation: bool = False):
    if evaluation:
        raise RuntimeError("teacher forcing replays training rollouts only; disable eval")
    rec = _index().take(getattr(sample, "prompt", ""), getattr(sample, "index", None))
    ports_token = _ports_policy_token()
    legacy_token = None if ports_token is not None else _legacy_policy_token(args)
    return apply_record(sample, rec, ports_token=ports_token, legacy_token=legacy_token)

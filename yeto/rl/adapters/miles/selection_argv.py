"""Miles flag spellings read by the ``--rl-engine`` support matrix (decoupling 4.3, audit E15).

``yeto.rl.engine.selection`` decides which combinations the ports engine
supports from neutral facts; the facts hidden in pass-through Miles argv
(``--extra``) are read here, so the core never spells a Miles flag.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


def flag_value(argv: Sequence[str], flag: str) -> str | None:
    tokens = list(argv)
    for index, token in enumerate(tokens):
        if token == flag and index + 1 < len(tokens):
            return tokens[index + 1]
        if token.startswith(flag + "="):
            return token.split("=", 1)[1]
    return None


@dataclass(frozen=True)
class ExtraArgvFacts:
    streaming_compaction: bool  # any --sao* flag
    advantage_estimator: str | None  # --advantage-estimator VALUE
    critic_flag: bool  # --use-critic (legacy fork only, not an upstream Miles flag)
    rollout_gpus_flag: bool  # --rollout-num-gpus VALUE


def extra_argv_facts(extra_argv: Sequence[str]) -> ExtraArgvFacts:
    tokens = [str(t) for t in extra_argv]
    return ExtraArgvFacts(
        streaming_compaction=any(t.startswith("--sao") for t in tokens),
        advantage_estimator=flag_value(tokens, "--advantage-estimator"),
        critic_flag="--use-critic" in tokens,
        rollout_gpus_flag=bool(flag_value(tokens, "--rollout-num-gpus")),
    )

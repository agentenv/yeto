"""Skip a round whose gradient norm is not finite instead of failing the run (opt-in).

Miles already leaves such a step out: with ``--no-check-for-nan-in-loss-and-grad`` Megatron's DDP
bucket check no longer raises "found NaN in local grad norm", and Miles' ``train_one_step`` drops the
``optimizer.step`` (and the weight-version bump) when ``get_grad_norm`` is NaN or inf. The strict driver
then refuses the round (``nonfinite_grad_norm``). With ``YETO_RL_NAN_SKIP`` set, the driver records the
round as skipped (event ``rl_nan_skip``) and goes on; the weights are unchanged, so a release of that
round republishes the previous policy. The round's groups are consumed, not retried.

``YETO_RL_NAN_SKIP=max_consecutive=2,max_fraction=0.05,min_rounds=20``: the run still stops
(``nan_skip_breaker``) after more than ``max_consecutive`` skipped rounds in a row, or when skipped rounds
exceed ``max_fraction`` of all rounds once ``min_rounds`` rounds ran. Unset: the strict behavior.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

ENV = "YETO_RL_NAN_SKIP"


class NanSkipConfigError(ValueError):
    pass


SPEC_KEYS = {"max_consecutive": int, "max_fraction": float, "min_rounds": int}


def parse_spec(raw: str | None) -> dict[str, float | int] | None:
    """A ``YETO_RL_NAN_SKIP`` value as :class:`NanSkipPolicy` keyword arguments (``{}``:
    the defaults); None when it is unset or empty (the strict behavior). The breaker
    registry reads the same value through it (``yeto.breakers.layers.trainer``)."""
    raw = (raw or "").strip()
    if not raw:
        return None
    if raw in ("1", "on", "true"):
        return {}
    kwargs = {}
    for part in raw.split(","):
        key, eq, value = part.strip().partition("=")
        if not eq or key not in SPEC_KEYS:
            raise NanSkipConfigError(f"{ENV}={raw!r}: expected key=value with keys {sorted(SPEC_KEYS)}")
        kwargs[key] = SPEC_KEYS[key](value)
    return kwargs


@dataclass
class NanSkipPolicy:
    max_consecutive: int = 2
    max_fraction: float = 0.05
    min_rounds: int = 20
    rounds: int = field(default=0, init=False)
    skipped: int = field(default=0, init=False)
    consecutive: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.max_consecutive < 1 or not 0 < self.max_fraction <= 1 or self.min_rounds < 1:
            raise NanSkipConfigError(f"{ENV}: needs max_consecutive >= 1, 0 < max_fraction <= 1, min_rounds >= 1")

    @classmethod
    def from_env(cls, environ=os.environ) -> NanSkipPolicy | None:
        kwargs = parse_spec(environ.get(ENV))
        return None if kwargs is None else cls(**kwargs)

    def finite_round(self) -> None:
        self.rounds += 1
        self.consecutive = 0

    def skipped_round(self) -> str | None:
        """Count a skipped round; return why the run must stop, or None."""
        self.rounds += 1
        self.skipped += 1
        self.consecutive += 1
        if self.consecutive > self.max_consecutive:
            return f"{self.consecutive} consecutive rounds with a non-finite grad_norm (> {self.max_consecutive})"
        if self.rounds >= self.min_rounds and self.skipped / self.rounds > self.max_fraction:
            return (f"{self.skipped}/{self.rounds} rounds with a non-finite grad_norm "
                    f"(> {self.max_fraction:.0%})")
        return None

    def fields(self) -> dict[str, float | int]:
        return {"nan_skipped": self.skipped, "nan_consecutive": self.consecutive, "rounds_seen": self.rounds,
                "nan_skip_fraction": self.skipped / self.rounds if self.rounds else 0.0}

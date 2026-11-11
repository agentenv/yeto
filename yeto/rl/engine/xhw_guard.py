"""Per-source guard of the train/rollout log-prob mismatch (opt-in, ``YETO_RL_XHW_GUARD``).

A buffer trainer trains on groups from several sampler shards, possibly on different
hardware and engines (NC MI355X vLLM next to B300 SGLang). With
``rollout_logprobs_as_old`` the rollout log-probs are the IS denominator, so a source
whose numerics drift biases the update. Miles reports the mismatch only per step
(``train_rollout_logprob_abs_diff``, ``train_rollout_kl``, ``sc_is_clipped``); this module

* names each group's **source** (``sampler_id`` + engine + engine version + card, from
  the buffer entry's provenance) -- :func:`source_key`;
* attributes the step values to the sources by their token share: a step from one
  source is exact; a mixed window is solved by least squares over the trailing steps
  (:class:`SourceAttribution`);
* judges each source against warn / act limits on a trailing window and returns an
  action per source (:class:`XhwGuard`): ``warn`` only, or ``quarantine`` (the reader
  passes its groups over for ``quarantine_rounds`` rounds, then lets one probe round
  through), ``downweight`` (its groups' advantages scaled by ``weight``; exact only
  without GRPO std normalization), or ``stop`` (the driver raises).

Rollout-side statistics that need no trainer pass (sampled-token log-prob, top-K head
mass, non-finite values) are computed per source at read time (:func:`rollout_stats`):
an engine whose kernels break shows there one round before the trainer sees it.

Switching ``rollout_logprobs_as_old`` off is not a fallback for score centering: its
loss reads ``rollout_log_probs`` and the rollout top-K directly (Miles
``score_centering_loss.py``), the recomputed old log-probs only add a diagnostic.

``YETO_RL_XHW_GUARD=warn_abs_diff=0.10,act_abs_diff=0.15,warn_kl=0.03,act_kl=0.06,
warn_clip=0.01,act_clip=0.03,window=3,min_rounds=2,action=quarantine,quarantine_rounds=8,
weight=0.5`` (every key optional; ``1``/``on`` = defaults). Unset: nothing changes.

Import-light: no torch, no Miles (numpy only inside :func:`rollout_stats`).
"""

from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

ENV = "YETO_RL_XHW_GUARD"
SCHEMA = "yeto-xhw-guard-v1"
UNKNOWN = "unknown"
# Step metrics this guard reads (Miles names; with or without the "train/" prefix)
METRICS = {"abs_diff": "train_rollout_logprob_abs_diff", "kl": "train_rollout_kl", "clip": "sc_is_clipped"}
ACTIONS = ("warn", "quarantine", "downweight", "stop")
SEP = "|"


class XhwGuardConfigError(ValueError):
    pass


class XhwGuardStop(RuntimeError):
    """A source crossed the act limits with ``action=stop``."""


def source_key(provenance: Mapping[str, Any] | None, shard: str | None = None) -> str:
    """``sampler_id|engine|engine_version|card_model`` of a buffer entry (``unknown`` parts
    for entries without provenance; ``shard`` stands in for a missing sampler id)."""
    p = dict(provenance or {})
    sampler = str(p.get("sampler_id") or UNKNOWN)
    if sampler == UNKNOWN:
        sampler = shard or UNKNOWN
    parts = (sampler, p.get("engine"), p.get("engine_version"), p.get("card_model"))
    return SEP.join(str(x) if x not in (None, "") else UNKNOWN for x in parts)


def source_fields(key: str) -> dict[str, str]:
    parts = (key.split(SEP) + [UNKNOWN] * 4)[:4]
    return dict(zip(("sampler_id", "engine", "engine_version", "card_model"), parts))


def step_value(train_metrics: Mapping[str, Any], name: str) -> float | None:
    for key in (name, f"train/{name}"):
        value = train_metrics.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            return float(value)
    return None


# ------------------------------------------------------------------ rollout-side stats
def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def rollout_stats(samples: Iterable[Mapping[str, Any]], *, low_head_mass: float = 0.9) -> dict[str, float]:
    """Per-source rollout-side numbers of a set of buffer samples (no trainer needed).

    ``tokens``; ``sampled_lp_mean`` (mean sampled-token log-prob = minus a cross-entropy
    estimate); ``head_mass_mean`` (mean of sum(exp(top-K)) over tokens with top-K);
    ``low_head_mass_frac`` (tokens whose top-K mass is below ``low_head_mass``);
    ``nonfinite`` (non-finite sampled log-probs; the -inf pads of the top-K are not counted)."""
    import numpy as np

    tokens = nonfinite = head_tokens = low = 0
    lp_sum = mass_sum = 0.0
    for sample in samples:
        lps = sample.get("rollout_log_probs")
        if lps is not None and len(lps):
            lp = np.asarray(lps, dtype=np.float64).reshape(-1)
            ok = np.isfinite(lp)
            tokens += int(ok.sum())
            nonfinite += int((~ok).sum())
            lp_sum += float(lp[ok].sum())
        top = sample.get("rollout_top_logprobs")
        if top is not None and len(top):
            t = np.asarray(top, dtype=np.float64)
            if t.ndim == 2:
                mass = np.where(np.isfinite(t), np.exp(np.minimum(t, 0.0)), 0.0).sum(-1)
                head_tokens += int(mass.shape[0])
                mass_sum += float(mass.sum())
                low += int((mass < low_head_mass).sum())
    return {"tokens": float(tokens), "nonfinite": float(nonfinite),
            "sampled_lp_mean": lp_sum / tokens if tokens else math.nan,
            "head_mass_mean": mass_sum / head_tokens if head_tokens else math.nan,
            "low_head_mass_frac": low / head_tokens if head_tokens else math.nan}


# ------------------------------------------------------------------ attribution
def _solve(rows: Sequence[tuple[Mapping[str, float], float]], sources: Sequence[str]) -> dict[str, float]:
    """Non-negative least squares of ``value = sum_s share_s * x_s`` (small: a few
    sources). Sources the rows cannot separate get the value of the rows they appear in."""
    n = len(sources)
    index = {s: i for i, s in enumerate(sources)}
    a = [[0.0] * n for _ in range(n)]
    b = [0.0] * n
    for shares, value in rows:
        vec = [0.0] * n
        for s, w in shares.items():
            vec[index[s]] = w
        for i in range(n):
            b[i] += vec[i] * value
            for j in range(n):
                a[i][j] += vec[i] * vec[j]
    active = list(range(n))
    x = [0.0] * n
    for _ in range(n):  # active-set: drop variables that come out negative
        m = len(active)
        mat = [[a[i][j] for j in active] + [b[i]] for i in active]
        ok = True
        for c in range(m):  # Gauss-Jordan with partial pivoting
            p = max(range(c, m), key=lambda r: abs(mat[r][c]))
            if abs(mat[p][c]) < 1e-12:
                ok = False
                break
            mat[c], mat[p] = mat[p], mat[c]
            for r in range(m):
                if r != c:
                    f = mat[r][c] / mat[c][c]
                    for k in range(c, m + 1):
                        mat[r][k] -= f * mat[c][k]
        if not ok:
            break
        sol = {active[c]: mat[c][m] / mat[c][c] for c in range(m)}
        neg = [i for i, v in sol.items() if v < 0]
        if not neg:
            x = [sol.get(i, 0.0) for i in range(n)]
            return {s: x[index[s]] for s in sources}
        active = [i for i in active if i not in neg]
        if not active:
            break
    # rank deficient: each source gets the share-weighted mean of the rows it is in
    out = {}
    for s in sources:
        num = den = 0.0
        for shares, value in rows:
            w = shares.get(s, 0.0)
            num += w * value
            den += w
        out[s] = num / den if den else math.nan
    return out


@dataclass
class SourceAttribution:
    """Trailing window of (token shares, step value) rows for one metric."""

    window: int = 3
    rows: deque = field(default_factory=deque)

    def add(self, shares: Mapping[str, float], value: float) -> None:
        self.rows.append((dict(shares), float(value)))
        while len(self.rows) > max(1, self.window):
            self.rows.popleft()

    def estimate(self) -> dict[str, dict[str, float]]:
        """{source: {"value", "exact" (1 when a single-source step of the window names it), "rounds"}}."""
        sources = sorted({s for shares, _ in self.rows for s in shares})
        if not sources:
            return {}
        solved = _solve(list(self.rows), sources)
        out = {}
        for s in sources:
            alone = [v for shares, v in self.rows if set(shares) == {s}]
            seen = sum(1 for shares, _ in self.rows if shares.get(s, 0.0) > 0)
            value = sum(alone) / len(alone) if alone else solved[s]
            out[s] = {"value": value, "exact": 1.0 if alone else 0.0, "rounds": float(seen)}
        return out


# ------------------------------------------------------------------ policy
@dataclass(frozen=True)
class Limits:
    abs_diff: float
    kl: float
    clip: float


@dataclass
class XhwGuard:
    """Judges every source each round; see the module doc for the env syntax.

    Defaults (measured, Kimi-K3.1-Pro, rollout top-128, R3 on, step 1 = pure numerics):
    B300 SGLang -> B300 Megatron abs_diff 0.023-0.028 / KL 0.0013-0.0016; MI355X vLLM
    ``amd-article`` -> B300 abs_diff 0.076-0.099 / KL 0.015-0.020 / clipfrac 0.40-0.55%.
    warn sits above the MI355X band (0.10 / 0.03 / 1%), act at about 2x it."""

    warn: Limits = Limits(0.10, 0.03, 0.01)
    act: Limits = Limits(0.15, 0.06, 0.03)
    window: int = 3
    min_rounds: int = 2
    action: str = "quarantine"
    quarantine_rounds: int = 8
    weight: float = 0.5
    _attr: dict[str, SourceAttribution] = field(default_factory=dict, init=False)
    _state: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.action not in ACTIONS:
            raise XhwGuardConfigError(f"{ENV}: action must be one of {ACTIONS}")
        if self.window < 1 or self.min_rounds < 1 or self.quarantine_rounds < 1 or not 0 < self.weight <= 1:
            raise XhwGuardConfigError(f"{ENV}: needs window, min_rounds, quarantine_rounds >= 1 and 0 < weight <= 1")
        for name in ("abs_diff", "kl", "clip"):
            w, a = getattr(self.warn, name), getattr(self.act, name)
            if not (0 < w <= a):
                raise XhwGuardConfigError(f"{ENV}: needs 0 < warn_{name} <= act_{name}")
        self._attr = {m: SourceAttribution(self.window) for m in METRICS}

    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ) -> "XhwGuard | None":
        raw = (environ.get(ENV) or "").strip()
        if not raw or raw in ("0", "off", "false"):
            return None
        if raw in ("1", "on", "true"):
            return cls()
        floats = {f"{lvl}_{m}" for lvl in ("warn", "act") for m in METRICS} | {"weight"}
        ints = {"window", "min_rounds", "quarantine_rounds"}
        values: dict[str, Any] = {}
        for part in raw.split(","):
            key, eq, value = part.strip().partition("=")
            if not eq or key not in floats | ints | {"action"}:
                raise XhwGuardConfigError(f"{ENV}={raw!r}: unknown or malformed {part!r}")
            try:
                values[key] = float(value) if key in floats else int(value) if key in ints else value
            except ValueError:
                raise XhwGuardConfigError(f"{ENV}={raw!r}: bad value in {part!r}") from None
        base = cls.__dataclass_fields__
        limits = {lvl: Limits(*(values.pop(f"{lvl}_{m}", getattr(base[lvl].default, m)) for m in METRICS))
                  for lvl in ("warn", "act")}
        return cls(**limits, **values)

    # ---------------------------------------------------------------- per round
    def observe(self, rollout_id: int, source_tokens: Mapping[str, float],
                train_metrics: Mapping[str, Any]) -> dict[str, Any]:
        """Record one trained step; returns the ``rl_xhw_mismatch`` event fields:
        per-source estimates, level (ok/warn/act), the action now in force, and
        ``actions`` {source: {"action", "until_round"/"weight"}} for the reader."""
        total = float(sum(v for v in source_tokens.values() if v > 0))
        shares = {s: float(v) / total for s, v in source_tokens.items() if v > 0} if total else {}
        step = {name: step_value(train_metrics, metric) for name, metric in METRICS.items()}
        for name, value in step.items():
            if value is not None and shares:
                self._attr[name].add(shares, value)
        estimates = {name: self._attr[name].estimate() for name in METRICS}
        sources: dict[str, Any] = {}
        stop: list[str] = []
        for s in sorted(set(shares) | set(self._state)):
            st = self._state.setdefault(s, {"level": "ok", "action": None, "until_round": None, "trips": 0})
            if st["action"] == "quarantine" and st["until_round"] is not None and rollout_id >= st["until_round"]:
                st.update(action=None, until_round=None)  # probe: let the next round through
            row = {**source_fields(s), "share": shares.get(s, 0.0)}
            level, over = "ok", []
            rounds = 0.0
            for name in METRICS:
                e = estimates[name].get(s)
                if e is None:
                    continue
                row[name], row[f"{name}_exact"] = e["value"], bool(e["exact"])
                rounds = max(rounds, e["rounds"])
                v = e["value"]
                if not math.isfinite(v):
                    continue
                if v > getattr(self.act, name):
                    over.append(name)
                    level = "act"
                elif v > getattr(self.warn, name) and level == "ok":
                    level = "warn"
            row["rounds"] = int(rounds)
            judged = rounds >= self.min_rounds
            if not judged and level == "act":
                level = "warn"  # too few rounds to act on
            row["level"], row["over"] = level, over
            if level == "act" and st["action"] is None and s in shares:
                st["trips"] += 1
                if self.action == "quarantine":
                    st.update(action="quarantine", until_round=int(rollout_id) + 1 + self.quarantine_rounds)
                    for attr in self._attr.values():  # the probe round re-measures from scratch
                        attr.rows = deque(r for r in attr.rows if s not in r[0])
                elif self.action == "downweight":
                    st.update(action="downweight", until_round=None)
                elif self.action == "stop":
                    stop.append(s)
            if level == "ok" and st["action"] == "downweight" and judged:
                st.update(action=None)
            st["level"] = level
            row.update(action=st["action"], until_round=st["until_round"], trips=st["trips"])
            sources[s] = row
        event = {"schema": SCHEMA, "rollout_id": int(rollout_id),
                 "step": {k: v for k, v in step.items()}, "sources": sources,
                 "actions": self.actions(), "policy": self.action,
                 "warn": vars(self.warn), "act": vars(self.act)}
        if stop:
            raise XhwGuardStop(f"rollout {rollout_id}: sources over the act limits: {stop} "
                               f"({ {s: sources[s]['over'] for s in stop} })")
        return event

    def actions(self) -> dict[str, dict[str, Any]]:
        out = {}
        for s, st in sorted(self._state.items()):
            if st["action"] == "quarantine":
                out[s] = {"action": "quarantine", "until_round": st["until_round"]}
            elif st["action"] == "downweight":
                out[s] = {"action": "downweight", "weight": self.weight}
        return out


# ------------------------------------------------------------------ reader side
def downweight_group(samples: Sequence[dict[str, Any]], weight: float) -> list[dict[str, Any]]:
    """Scale a group's centered rewards by ``weight`` around the group mean: with
    group-centered advantages and no std normalization (the K3.1 score-centering
    recipe) every advantage is scaled by exactly ``weight``; the group mean (and the
    reward metrics' mean) is unchanged."""
    rewards = [s.get("reward") for s in samples]
    if not samples or any(not _finite(r) for r in rewards):
        return [dict(s) for s in samples]
    mean = sum(rewards) / len(rewards)
    out = []
    for sample, r in zip(samples, rewards):
        sample = dict(sample)
        metadata = dict(sample.get("metadata") or {})
        metadata["yeto_xhw_downweight"] = {"weight": float(weight), "reward": float(r)}
        sample.update(reward=mean + float(weight) * (float(r) - mean), metadata=metadata)
        out.append(sample)
    return out

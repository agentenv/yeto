"""Training/inference mismatch criteria, computed by yeto in the Miles convention
(rl-verl-backend 1.6, design D7).

Every backend hands yeto the same three per-token tensors -- training-side
log-probs (recomputed old log-probs), inference-side log-probs (what the
rollout engine reported) and the response mask -- and yeto computes the
criteria itself, so the numbers mean the same thing whichever engine produced
them.  Backend-native metrics (verl ``rollout_corr/*`` ...) are archived next to
these, never used as a criterion.

Convention (Miles ``losses.py`` / ``math_utils.py``, cross-checked in
VERL-MODAL-CHECK-S16.md section 2 against ``compute_approx_kl(low_var_kl)``):

* ``abs_diff``: |train - infer| per token, mean within each sample (masked),
  then mean over samples;
* ``k3``: Schulman k3 of KL(infer || train): x = train - infer (nan -> 0,
  clamped to +-20), exp(x) - 1 - x, clamped per token to [-10, 10], then the
  same per-sample mean;
* ``tis_clipfrac``: share of tokens whose ratio exp(train - infer) exceeds the
  TIS upper bound (default 2.0), per-sample mean;
* ``signed_mean``: the signed mean of (train - infer) over all masked tokens.
  k3 and tis_clipfrac cannot see a one-sided shift such as top_p < 1
  (S16: top_p 0.9 gives signed mean about -0.032 while k3 stays 0.0017), so it
  is a criterion of its own (user decision 2026-10-08).

Thresholds are keyed by (backend, train engine, inference engine+version,
hardware); a missing key is reported as "uncalibrated" instead of guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

LOG_RATIO_CLAMP = 20.0
K3_TOKEN_CLAMP = 10.0
DEFAULT_TIS_UPPER = 2.0
SCHEMA = "yeto-mismatch-criteria-v1"


def _sample_mean(values, mask):
    m = mask.float()
    per = (values * m).sum(-1) / m.sum(-1).clamp(min=1)
    valid = m.sum(-1) > 0
    if not bool(valid.any()):
        return float("nan")
    return float(per[valid].mean().item())


def k3_per_token(train_logp, infer_logp):
    import torch

    x = torch.nan_to_num((train_logp - infer_logp).float(), nan=0.0, posinf=LOG_RATIO_CLAMP,
                         neginf=-LOG_RATIO_CLAMP).clamp(-LOG_RATIO_CLAMP, LOG_RATIO_CLAMP)
    return (torch.exp(x) - 1 - x).clamp(-K3_TOKEN_CLAMP, K3_TOKEN_CLAMP)


def compute(train_logp, infer_logp, mask, *, tis_upper: float = DEFAULT_TIS_UPPER) -> dict:
    """Criteria for one batch of (n_samples, n_tokens) tensors."""
    import torch

    if train_logp.shape != infer_logp.shape or train_logp.shape != mask.shape:
        raise ValueError(f"shape mismatch: train {tuple(train_logp.shape)} infer "
                         f"{tuple(infer_logp.shape)} mask {tuple(mask.shape)}")
    t, r, m = train_logp.float(), infer_logp.float(), mask.bool()
    d = t - r
    d = torch.where(torch.isfinite(d), d, torch.zeros_like(d))
    mf = m.float()
    n_tok = float(mf.sum().item())
    clipped = (torch.exp(d) > tis_upper).float()
    nonfinite = int((~torch.isfinite(t[m])).sum().item() + (~torch.isfinite(r[m])).sum().item())
    return {
        "schema": SCHEMA,
        "n_samples": int(m.shape[0]),
        "n_tokens": int(n_tok),
        "abs_diff": _sample_mean(d.abs(), m),
        "k3": _sample_mean(k3_per_token(t, r), m),
        "tis_clipfrac": _sample_mean(clipped, m),
        "signed_mean": float((d * mf).sum().item() / n_tok) if n_tok else float("nan"),
        "tis_upper": float(tis_upper),
        "nonfinite_tokens": nonfinite,
    }


@dataclass(frozen=True)
class Thresholds:
    abs_diff: float
    k3: float
    tis_clipfrac: float
    signed_mean_abs: float


# (backend, train engine, inference engine + version, hardware) -> thresholds.
# Only entries backed by a measurement; the values are provisional (design
# "待定" 2) and come from VERL-MODAL-CHECK-S16.md 7.4 (measured 0.017 / 0.0008 /
# <=5e-5 / signed -0.0009) with headroom, the design's suggested 0.03 / 0.01 / 1%.
THRESHOLDS: dict[tuple[str, str, str, str], Thresholds] = {
    ("verl", "fsdp2", "vllm-0.29.0", "H100"): Thresholds(0.03, 0.01, 0.01, 0.01),
}


class Uncalibrated(LookupError):
    """No threshold row for this backend/engine/hardware combination."""


def thresholds_for(key: tuple[str, str, str, str],
                   table: Mapping[tuple[str, str, str, str], Thresholds] = THRESHOLDS) -> Thresholds:
    try:
        return table[key]
    except KeyError:
        raise Uncalibrated(f"训推不一致阈值未标定: {key}") from None


def judge(metrics: Mapping[str, float], limits: Thresholds) -> list[str]:
    """Names of the criteria that alarm (empty = pass); non-finite values alarm."""
    import math

    alarms = []
    for name, limit in (("abs_diff", limits.abs_diff), ("k3", limits.k3),
                        ("tis_clipfrac", limits.tis_clipfrac)):
        value = metrics.get(name)
        if value is None or not math.isfinite(value) or value > limit:
            alarms.append(name)
    signed = metrics.get("signed_mean")
    if signed is None or not math.isfinite(signed) or abs(signed) > limits.signed_mean_abs:
        alarms.append("signed_mean")
    if metrics.get("nonfinite_tokens"):
        alarms.append("nonfinite_tokens")
    return alarms

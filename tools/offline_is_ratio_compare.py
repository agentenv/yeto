"""Offline comparison of IS corrections (TIS / IcePop / M2PO) on saved logprobs.

openspec change rl-inter-island-scheduling, tasks 0.10.

Input: dump files (torch .pt dicts) each holding, for one batch of samples,
``rollout_log_probs`` (behaviour policy) and ``old_log_probs`` (training-side
policy) of shape [B, T] plus ``response_mask``.  The lag between the two
policies is supplied by the caller (``--lag``); the verl dumps found so far
(s16-verl-mismatch) are lag 0 (train/inference mismatch only).

Corrections (ratio r = exp(train - behaviour), token level):
  tis    : w = clamp(r, lo, hi)                 (project default lo=0, hi=2.0)
  icepop : w = r if lo <= r <= hi else 0        (miles icepop_function; lo=0.5, hi=2.0 here)
  m2po   : mask tokens with largest (log r)^2 until mean (log r)^2 of the kept
           tokens <= tau (M2PO second-moment budget, default tau=0.04); w = r
           on kept tokens, 0 otherwise.

Metrics per correction: altered_frac (tokens whose weight != r), var(w),
ESS/N = (sum w)^2 / (N * sum w^2).
"""
from __future__ import annotations

import argparse
import json
import math
from typing import Iterable, Sequence


def tis(r: Sequence[float], lo: float = 0.0, hi: float = 2.0) -> list[float]:
    return [min(max(x, lo), hi) for x in r]


def icepop(r: Sequence[float], lo: float = 0.5, hi: float = 2.0) -> list[float]:
    return [x if lo <= x <= hi else 0.0 for x in r]


def m2po(r: Sequence[float], tau: float = 0.04) -> list[float]:
    sq = [math.log(x) ** 2 if x > 0 else float("inf") for x in r]
    order = sorted(range(len(r)), key=lambda i: sq[i], reverse=True)
    total, n = sum(s for s in sq if math.isfinite(s)), len(r)
    keep = [True] * n
    inf_cnt = sum(1 for s in sq if not math.isfinite(s))
    for i in order:
        if n == 0 or (inf_cnt == 0 and total / n <= tau):
            break
        keep[i] = False
        n -= 1
        if math.isfinite(sq[i]):
            total -= sq[i]
        else:
            inf_cnt -= 1
    return [x if k else 0.0 for x, k in zip(r, keep)]


def stats(r: Sequence[float], w: Sequence[float]) -> dict:
    n = len(w)
    if n == 0:
        return {"n": 0}
    mean = sum(w) / n
    var = sum((x - mean) ** 2 for x in w) / n
    s2 = sum(x * x for x in w)
    return {
        "n": n,
        "altered_frac": sum(1 for a, b in zip(r, w) if a != b) / n,
        "mean_w": mean,
        "var_w": var,
        "ess_over_n": (sum(w) ** 2 / (n * s2)) if s2 > 0 else 0.0,
    }


def compare(r: Sequence[float], tis_hi=2.0, tis_lo=0.0, ice_lo=0.5, ice_hi=2.0, tau=0.04) -> dict:
    r = list(r)
    return {
        "raw": stats(r, r),
        "tis": stats(r, tis(r, tis_lo, tis_hi)),
        "icepop": stats(r, icepop(r, ice_lo, ice_hi)),
        "m2po": stats(r, m2po(r, tau)),
    }


def ratios_from_dump(path: str) -> list[float]:
    import torch  # only needed for real dumps

    d = torch.load(path, map_location="cpu", weights_only=False)
    m = d["response_mask"].bool()
    lr = (d["old_log_probs"].double() - d["rollout_log_probs"].double())[m]
    return lr.exp().tolist()


def main(argv: Iterable[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--lag", type=int, required=True)
    ap.add_argument("--tau", type=float, default=0.04)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    per, allr = {}, []
    for p in a.dumps:
        r = ratios_from_dump(p)
        allr += r
        per[p] = compare(r, tau=a.tau)
    res = {"lag": a.lag, "tau": a.tau, "tis_bounds": [0.0, 2.0], "icepop_bounds": [0.5, 2.0],
           "pooled": compare(allr, tau=a.tau), "per_dump": per}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res["pooled"], indent=1))


if __name__ == "__main__":
    main()

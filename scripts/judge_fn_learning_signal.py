#!/usr/bin/env python3
"""Learning-signal judge for the Flash-Next 4-layer stage-A rerun (fnrun.sh fn8r).

try27 (fn8s, gsm8k exact match) trained 6 rounds with reward/loss/grad_norm all 0:
zero-variance GRPO groups -> zero advantage -> LoRA B never left zero, so the
loss/logprob judge was vacuous (evidence/fn/FN-A-RESULT.md section 4). This judge
only asks whether a learning signal existed, from the island event tape
(rl-island-0.jsonl):

  rounds              >= --min-rounds rl_local_round events
  nonzero_grad        at least one round with finite grad_norm > 0
  group_variance      at least one round with zero_variance_group_ratio < 1
                      (fallback when the field is absent: reward_std > 0)
  export_hash_change  rl_publication sync/publication_payload_hash takes >= 2 distinct
                      values (the trained LoRA export differs from the initial one).
                      Fewer than 2 hashed publications -> "unavailable" (not a FAIL
                      unless --require-export-hash).

usage: judge_fn_learning_signal.py <rl-island-0.jsonl> [--min-rounds N] [--require-export-hash] [--json OUT]
exit: 0 PASS, 1 FAIL, 2 unreadable input.
"""
from __future__ import annotations

import argparse
import json
import math
import sys


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def judge(events: list[dict], *, min_rounds: int = 1, require_export_hash: bool = False) -> dict:
    rounds = [e for e in events if e.get("event") == "rl_local_round"]
    grads = [_num(e.get("grad_norm")) for e in rounds]
    zv = [_num(e.get("zero_variance_group_ratio")) for e in rounds]
    rstd = [_num(e.get("reward_std")) for e in rounds]
    var_rounds = [i for i, (z, s) in enumerate(zip(zv, rstd))
                  if (z is not None and z < 1.0) or (z is None and s is not None and s > 0.0)]
    hashes = [e.get("sync/publication_payload_hash") for e in events
              if e.get("event") == "rl_publication" and e.get("sync/publication_payload_hash")]
    distinct = list(dict.fromkeys(hashes))
    if len(hashes) < 2:
        export = {"status": "unavailable", "ok": not require_export_hash, "publications": len(hashes)}
    else:
        export = {"status": "changed" if len(distinct) > 1 else "unchanged", "ok": len(distinct) > 1,
                  "publications": len(hashes), "distinct": len(distinct),
                  "first": distinct[0][:16], "last": hashes[-1][:16]}
    checks = {
        "rounds": {"ok": len(rounds) >= min_rounds, "count": len(rounds), "min": min_rounds},
        "nonzero_grad": {"ok": any(g is not None and g > 0 for g in grads),
                         "rounds_with_grad": sum(1 for g in grads if g is not None and g > 0),
                         "grad_norm": grads},
        "group_variance": {"ok": bool(var_rounds), "rounds_with_variance": len(var_rounds),
                           "zero_variance_group_ratio": zv, "reward_std": rstd},
        "export_hash_change": export,
    }
    return {"verdict": "PASS" if all(c["ok"] for c in checks.values()) else "FAIL", "checks": checks}


def load(path: str) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if isinstance(d, dict):
                out.append(d)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("jsonl")
    ap.add_argument("--min-rounds", type=int, default=1)
    ap.add_argument("--require-export-hash", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    try:
        events = load(a.jsonl)
    except OSError as exc:
        print(f"judge_fn_learning_signal: cannot read {a.jsonl}: {exc}", file=sys.stderr)
        return 2
    res = judge(events, min_rounds=a.min_rounds, require_export_hash=a.require_export_hash)
    for name, c in res["checks"].items():
        brief = {k: v for k, v in c.items() if k not in ("grad_norm", "zero_variance_group_ratio", "reward_std")}
        print(f"{name}: {'PASS' if c['ok'] else 'FAIL'} {json.dumps(brief, sort_keys=True)}")
    print(f"verdict: {res['verdict']}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    return 0 if res["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

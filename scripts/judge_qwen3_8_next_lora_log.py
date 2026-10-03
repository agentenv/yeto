#!/usr/bin/env python3
"""Judge a Qwen3.8-Flash-Next-4layer native LoRA GRPO trainer log (M4 G3/G4, T2-S7 §4).

    judge_qwen3_8_next_lora_log.py LOG --num-gpus 4 [--rollouts 5] [--rank 32]
        [--expert-rank 8] [--eval-min 0] [--adapter-restart] [--json OUT]

Checks (each reported PASS/FAIL, exit 0 only if all pass):
  trainable      `native LoRA applied: ... trainable=N` shows exactly the two
                 per-rank values of expected_rank_trainable_4layer, each
                 num_gpus/2 times (M3 acceptance #1, F2 wording)
  lora_check     no `[LORA-CHECK]` / `end_weight_update failed` line, i.e. the
                 --check-lora-weight-equal name-set + sha256 read-back passed
                 (M3 #3/#4: 54 serving modules incl. zero-padded experts)
  rollouts       `step <i>: {'train/...}` rounds == --rollouts
  rewards        (info) rollout/rewards and truncated ratio per round, grad_norm
                 == 0 everywhere is flagged: LoRA B stays 0 and the logprob
                 check below is vacuous for the QSA de-interleave
  finite         train/loss, ppo_kl, grad_norm, train_rollout_logprob_abs_diff
                 present and finite in every round
  logprob_diff   rounds >= 2 of train_rollout_logprob_abs_diff within 10x of
                 round 1 (QSA de-interleave end-to-end, M3 #2)
  load_keys      no `missing key` / `unexpected key` from checkpoint loading
  eval           (--eval-min N) at least N `eval <i>:` lines carrying eval/aime
  adapter        (--adapter-restart) `Successfully loaded LoRA adapter from`
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from yeto.rl.profiles.qwen3_8_next import expected_rank_trainable_4layer  # noqa: E402

# Miles logs the per-round train metrics as "step <i>: {...}" (log_utils.py:554 / model.py:856)
# and older builds as "train <i>: {...}"; accept both, de-duplicating per round.
TRAIN_RE = re.compile(r"\b(?:train|step) (\d+): (\{'train/.*\})")
ROLLOUT_RE = re.compile(r"\brollout (\d+): (\{'rollout/.*\})")
EVAL_RE = re.compile(r"\beval (\d+): (\{.*\})")
REPEAT_RE = re.compile(r"\[repeated (\d+)x across cluster\]")
TRAINABLE_RE = re.compile(r"native LoRA applied: rank=(\d+) expert_rank=(\d+) .*?trainable=(\d+)")
METRICS = ("train/loss", "train/ppo_kl", "train/grad_norm", "train/train_rollout_logprob_abs_diff")


def _parse_metrics(blob: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, val in re.findall(r"'([^']+)': ([-+]?[0-9.eE+-]+|nan|inf|-inf)", blob):
        try:
            out[key] = float(val)
        except ValueError:
            pass
    return out


def judge(text: str, *, num_gpus: int, rollouts: int, rank: int, expert_rank: int,
          eval_min: int = 0, adapter_restart: bool = False) -> dict[str, object]:
    res: dict[str, object] = {}
    exp0, exp1 = expected_rank_trainable_4layer(rank, expert_rank, num_gpus)
    seen: Counter[int] = Counter()
    repeats = 0  # ray's log dedup folds near-identical actor lines into "[repeated Nx across cluster]"
    total = 0  # a folded line stands for N occurrences (itself included), a plain line for 1
    for line in text.splitlines():
        m = TRAINABLE_RE.search(line)
        if m and int(m.group(1)) == rank and int(m.group(2)) == expert_rank:
            seen[int(m.group(3))] += 1
            r = REPEAT_RE.search(line)
            n = int(r.group(1)) if r else 1
            repeats += n - 1
            total += n
    expected = Counter({exp0: num_gpus // 2, exp1: num_gpus // 2})
    exact = seen == expected and repeats == 0
    # deduplicated log: the folded lines cannot be attributed, accept when the value
    # set is exactly the expected pair and the folded total equals the rank count
    folded_ok = repeats > 0 and set(seen) == set(expected) and total == num_gpus
    res["trainable"] = {
        "pass": exact or folded_ok,
        "expected": {str(exp0): num_gpus // 2, str(exp1): num_gpus // 2},
        "seen": {str(k): v for k, v in sorted(seen.items())},
        "dedup_repeats": repeats,
        "exact_count": exact,
    }
    bad = [l for l in text.splitlines() if "LORA-CHECK" in l or "end_weight_update failed" in l]
    res["lora_check"] = {"pass": not bad, "lines": bad[:5]}
    rounds = {int(m.group(1)): _parse_metrics(m.group(2)) for m in TRAIN_RE.finditer(text)}
    res["rollouts"] = {"pass": len(rounds) == rollouts, "seen": sorted(rounds), "expected": rollouts}
    finite = all(k in r and math.isfinite(r[k]) for r in rounds.values() for k in METRICS)
    res["finite"] = {"pass": bool(rounds) and finite,
                     "metrics": {str(i): {k: r.get(k) for k in METRICS} for i, r in sorted(rounds.items())}}
    rewards = {int(m.group(1)): _parse_metrics(m.group(2)) for m in ROLLOUT_RE.finditer(text)}
    grads = [rounds[i].get("train/grad_norm") for i in sorted(rounds)]
    res["rewards"] = {
        "pass": True,
        "nonzero_grad_rounds": [i for i in sorted(rounds) if (rounds[i].get("train/grad_norm") or 0) > 0],
        "all_grad_zero": bool(grads) and all((g or 0) == 0 for g in grads),
        "rollout": {str(i): {k: r.get(k) for k in ("rollout/rewards", "rollout/truncated", "rollout/response_lengths")}
                    for i, r in sorted(rewards.items())},
    }
    diffs = [rounds[i].get("train/train_rollout_logprob_abs_diff") for i in sorted(rounds)]
    ok = bool(diffs) and all(d is not None and math.isfinite(d) for d in diffs)
    if ok and len(diffs) > 1:
        base = max(diffs[0], 1e-6)
        ok = all(d <= 10 * base for d in diffs[1:])
    res["logprob_diff"] = {"pass": ok, "values": diffs}
    keys = [l for l in text.splitlines() if re.search(r"(missing|unexpected) key", l, re.I)]
    res["load_keys"] = {"pass": not keys, "lines": keys[:5]}
    if eval_min:
        evals = [(int(m.group(1)), m.group(2)) for m in EVAL_RE.finditer(text) if "eval/aime" in m.group(2)]
        res["eval"] = {"pass": len(evals) >= eval_min, "seen": [i for i, _ in evals],
                       "metrics": {str(i): _parse_metrics(b) for i, b in evals}}
    if adapter_restart:
        hit = [l for l in text.splitlines() if "Successfully loaded LoRA adapter from" in l]
        res["adapter"] = {"pass": bool(hit), "lines": hit[:2]}
    res["verdict"] = "PASS" if all(v["pass"] for k, v in res.items() if isinstance(v, dict)) else "FAIL"
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--num-gpus", type=int, required=True)
    ap.add_argument("--rollouts", type=int, default=5)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--expert-rank", type=int, default=8)
    ap.add_argument("--eval-min", type=int, default=0)
    ap.add_argument("--adapter-restart", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    res = judge(open(a.log, errors="replace").read(), num_gpus=a.num_gpus, rollouts=a.rollouts, rank=a.rank,
                expert_rank=a.expert_rank, eval_min=a.eval_min, adapter_restart=a.adapter_restart)
    out = json.dumps(res, indent=2)
    if a.json:
        open(a.json, "w").write(out + "\n")
    for k, v in res.items():
        if isinstance(v, dict):
            print(f"{'PASS' if v['pass'] else 'FAIL'} {k}")
    print(f"verdict {res['verdict']}")
    return 0 if res["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

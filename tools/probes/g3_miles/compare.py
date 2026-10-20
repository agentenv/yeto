"""Judge the T-G3 harness outputs (CPU only). usage: python compare.py <out-dir>

Writes ``<out>/verdict.json`` and prints it. Thresholds are pre-registered in
SESSION25/gpu-review/T-G3.md; the noise floor (TE vs TE repeat, cont5 vs cont5b)
is reported next to every TE-vs-local and resume-vs-continuous difference.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import map_local_to_te  # noqa: E402

LP_MEAN_MAX, LP_ABS_MAX = 5e-3, 0.1  # label log-prob |delta|, nats
GRAD_REL_MAX, GRAD_GLOBAL_REL_MAX = 0.05, 0.02  # per-tensor / all-tensor relative L2
LOSS_ABS_MAX = 1e-3


def lp_stats(a: dict, b: dict) -> dict:
    d = (a["token_loss"][:-1] - b["token_loss"][:-1]).abs().double()
    return {"mean": float(d.mean()), "p99": float(d.quantile(0.99)), "max": float(d.max()),
            "loss_a": a["loss"], "loss_b": b["loss"], "loss_abs": abs(a["loss"] - b["loss"]),
            "logits_max_abs": float((a["logits"] - b["logits"]).abs().max())}


def grad_stats(out: Path, tag_a: str, tag_b: str, local_b: bool) -> dict:
    per, worst, num, den, ranks = [], None, 0.0, 0.0, 0
    for r in range(8):
        fa, fb = out / f"grads-{tag_a}.rank{r}.pt", out / f"grads-{tag_b}.rank{r}.pt"
        if not fa.exists():
            break
        ranks += 1
        ga = torch.load(fa, weights_only=True)
        gb = torch.load(fb, weights_only=True)
        if local_b:
            mapping, missing, unused = map_local_to_te(gb, set(ga))
            if missing or unused:
                return {"error": f"rank{r}: names do not map: missing={missing[:5]} unused={unused[:5]}"}
            gb = {mapping[n]: t for n, t in gb.items()}
        if set(ga) != set(gb):
            return {"error": f"rank{r}: grad name sets differ: {sorted(set(ga) ^ set(gb))[:6]}"}
        for n in sorted(ga):
            a, b = ga[n].double(), gb[n].double()
            na = float(a.norm())
            diff = float((a - b).norm())
            rel = diff / na if na > 0 else (0.0 if diff == 0 else math.inf)
            cos = float((a * b).sum() / (a.norm() * b.norm())) if na > 0 and float(b.norm()) > 0 else None
            num += diff ** 2
            den += na ** 2
            item = {"rank": r, "name": n, "rel": rel, "cos": cos, "norm": na}
            per.append(item)
            if worst is None or rel > worst["rel"]:
                worst = item
    per.sort(key=lambda x: -x["rel"])
    return {"ranks": ranks, "tensors": len(per), "global_rel": math.sqrt(num / den) if den else None,
            "worst": per[:8], "over_threshold": [p for p in per if p["rel"] > GRAD_REL_MAX and p["norm"] > 0]}


def n75(out: Path) -> dict:
    v: dict = {}
    load = lambda t: torch.load(out / f"logits-{t}.pt", weights_only=True)  # noqa: E731
    have = {t for t in ("te", "te2", "local") if (out / f"logits-{t}.pt").exists()}
    if {"te", "te2"} <= have:
        v["noise_te_vs_te2"] = {"logprob": lp_stats(load("te"), load("te2")), "grads": grad_stats(out, "te", "te2", False)}
    if {"te", "local"} <= have:
        lp = lp_stats(load("te"), load("local"))
        g = grad_stats(out, "te", "local", True)
        ok = (lp["mean"] <= LP_MEAN_MAX and lp["max"] <= LP_ABS_MAX and lp["loss_abs"] <= LOSS_ABS_MAX
              and "error" not in g and not g["over_threshold"] and (g["global_rel"] or 0) <= GRAD_GLOBAL_REL_MAX)
        v["te_vs_local"] = {"logprob": lp, "grads": g, "pass": ok}
    for t in sorted(have):
        info = json.loads((out / f"info-{t}.json").read_text())
        v[f"info_{t}"] = [{k: i.get(k) for k in ("impl", "transformer_impl", "dsa_core_attention", "loss", "timing",
                                                    "name_map", "num_params_local", "gpu")} for i in info]
        v[f"n82_{t}"] = [i.get("n82") for i in info[:1]]
    return v


def n71d(out: Path) -> dict:
    load = lambda m: json.loads((out / f"n71d-{m}.json").read_text()) if (out / f"n71d-{m}.json").exists() else None  # noqa: E731
    c5, c5b, r3, rs = load("cont5"), load("cont5b"), load("run3"), load("resume2")
    v: dict = {"present": [m for m, x in (("cont5", c5), ("cont5b", c5b), ("run3", r3), ("resume2", rs)) if x]}
    if c5:
        v["flags"] = c5[0]["flags"]
        v["hdo_cont5_final"] = c5[0]["hdo_final"]
    if c5 and c5b:
        v["noise_cont5_vs_cont5b"] = {
            "params_bitwise_equal": all(a["final_params"]["sha256"] == b["final_params"]["sha256"] for a, b in zip(c5, c5b)),
            "loss_abs": [abs(a["loss_mean_dp"] - b["loss_mean_dp"]) for a, b in zip(c5[0]["steps"], c5b[0]["steps"])],
            "grad_norm_abs": [abs(a["grad_norm"] - b["grad_norm"]) for a, b in zip(c5[0]["steps"], c5b[0]["steps"])]}
    if c5 and r3:
        v["run3_vs_cont5_steps1to3_loss_abs"] = [abs(a["loss_mean_dp"] - b["loss_mean_dp"])
                                                 for a, b in zip(r3[0]["steps"], c5[0]["steps"][:3])]
    if rs:
        r0 = rs
        v["restore"] = {
            "digest_equal_per_rank": [x["restore_digest_equal"] for x in r0],
            "restore_problems": r0[0]["restore_problems"], "first_step_problems": r0[0]["first_step_problems"],
            "hdo_steps_after_restore": [[l.get("steps") for l in x["hdo_after_restore"]["leaves"]] for x in r0],
            "hdo_steps_after_first_step": [[l.get("steps") for l in x["hdo_after_first_step"]["leaves"]] for x in r0],
            "diff": [x["restore_summary"].get("diff") for x in r0]}
        if c5:
            steps = {s["step"]: s for s in c5[0]["steps"]}
            v["resume_vs_cont5"] = {
                "params_bitwise_equal": all(a["final_params"]["sha256"] == b["final_params"]["sha256"]
                                            for a, b in zip(rs, c5)),
                "loss_abs": {s["step"]: abs(s["loss_mean_dp"] - steps[s["step"]]["loss_mean_dp"]) for s in rs[0]["steps"]},
                "grad_norm_abs": {s["step"]: abs(s["grad_norm"] - steps[s["step"]]["grad_norm"]) for s in rs[0]["steps"]},
                "lr": {s["step"]: (s["lr"], steps[s["step"]]["lr"]) for s in rs[0]["steps"]},
                "param_norm_max_rel": max(
                    abs(a["final_params"]["norms"][n] - b["final_params"]["norms"][n]) / max(b["final_params"]["norms"][n], 1e-30)
                    for a, b in zip(rs, c5) for n in b["final_params"]["norms"])}
    return v


def main() -> None:
    out = Path(sys.argv[1])
    verdict = {}
    for name, fn, sub in (("n75", n75, "n75"), ("n71d", n71d, "n71d")):
        try:
            verdict[name] = fn(out / sub)
        except Exception as exc:  # noqa: BLE001
            verdict[name] = {"error": f"{type(exc).__name__}: {exc}"}
    (out / "verdict.json").write_text(json.dumps(verdict, indent=1, default=str))
    print(json.dumps(verdict, indent=1, default=str)[:20000])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Judge S13 critic-family fork G1 (tasks 6.4 / 7.3-G1 / 8.4-G1) from a Modal run dir.

Common (tasks.md: "指标有限"): the main tape has N trained rounds (gae-* 2, vapo/sao 3) and
rl_learner_finalized; every round's critic/value_loss, critic/explained_variance, pg_loss and
grad_norm are present and finite; the container reported an H100; the overlay was applied
(launch.log '[yeto-setup] miles overlay applied' and no 'REFUSING'). Missing data -> INCOMPLETE,
never filled in. Case extras are RECORDED (no threshold without a source):
  vapo: stage-W product line (warmup_steps 50), main receipts carry its critic hash; any
        positive-example metric keys; sao: EV per round next to the legacy-path value (+0.1365,
        docs/TBENCH21_SAO_QWEN35_08B_VALIDATION_20260826.md, Terminal-Bench, different data).
usage: s13-forkg1-judge.py <run dir> <case> [expected GPU, default H100]
"""
import json, math, re, sys
from pathlib import Path

R = Path(sys.argv[1]); CASE = sys.argv[2]; EXPECT_GPU = sys.argv[3] if len(sys.argv) > 3 else "H100"
N = {"gae-la": 2, "gae-cs": 2, "vapo": 3, "sao": 3}[CASE]
LOG = (R / "launch.log").read_text(errors="replace") if (R / "launch.log").is_file() else ""


def tape(name):
    for path in sorted((R / "tape-direct").glob(f"**/{name}")) + sorted((R / "runs").glob(f"**/modal-tape/**/{name}")):
        if path.is_file() and path.stat().st_size:
            out = []
            for line in path.read_text(errors="replace").splitlines():
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
            return out, str(path)
    # fallback: echoed events in launch.log
    out = []
    for line in LOG.splitlines():
        at = line.find('{"event": ')
        if at >= 0 and "YETO_RL_EVENT" in line:
            try:
                out.append(json.loads(line[at:]))
            except ValueError:
                pass
    return out, ("launch.log echo" if out else None)


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


records, src = tape("rl-island-0.jsonl")
trained = {e.get("rollout_id"): e for e in records if e.get("event") == "rl_round_trained"}
local = {e.get("local_round_id"): e for e in records if e.get("event") == "rl_local_round"}
critic = {e.get("rollout_id"): e for e in records if e.get("event") == "rl_critic_round"}
rows, problems, missing, notes = [], [], [], []
# EV: neither Miles c35702e nor the fork emits critic/explained_variance (trainer.py:429 reads it,
# found S13): required only where tasks.md asks for the value (8.4), recorded otherwise.
REQUIRED = ("value_loss", "pg_loss", "grad_norm") + (("explained_variance",) if CASE == "sao" else ())
for rid in sorted(k for k in trained if k is not None):
    m = trained[rid].get("train_metrics") or {}
    lr = local.get(rid + 1) or local.get(rid) or {}
    pg = lr.get("pg_loss")
    if pg is None:
        pg = next((v for k, v in m.items() if re.search(r"(^|/)(pg_loss|policy_loss)$", k)), None)
    row = {"rollout_id": rid, "value_loss": m.get("critic/value_loss"),
           "explained_variance": m.get("critic/explained_variance"), "pg_loss": pg,
           "grad_norm": lr.get("grad_norm"),
           "critic_init_sha256": (critic.get(rid) or {}).get("rl/critic/critic_init_sha256"),
           "extra_metrics": {k: v for k, v in m.items() if re.search(r"positive|sao|advantage|return|value|reward|entropy|clipfrac", k)}}
    rows.append(row)
    for key in REQUIRED:
        if row[key] is None:
            missing.append(f"round {rid}: {key} missing")
        elif not finite(row[key]):
            problems.append(f"round {rid}: {key}={row[key]} not finite")
    for k, v in row["extra_metrics"].items():
        if isinstance(v, float) and not math.isfinite(v):
            problems.append(f"round {rid}: {k}={v} not finite")
if len(rows) < N:
    missing.append(f"{len(rows)} trained rounds < {N}")
if not any(e.get("event") == "rl_learner_finalized" for e in records):
    missing.append("no rl_learner_finalized")
if "REFUSING miles overlay" in LOG:
    problems.append("overlay REFUSING in launch.log")
overlay = [l.split("] ", 1)[-1][:400] for l in LOG.splitlines() if "miles overlay applied" in l]
if not overlay:
    missing.append("no 'miles overlay applied' line")
if re.search(r"error: patch failed|git apply.*failed|sha256sum: WARNING", LOG):
    problems.append("overlay patch check/apply failure in launch.log")
gpu = [l for l in LOG.splitlines() if re.search(r"\[modal-island \d+\] requested", l)]
if not any(EXPECT_GPU in l for l in gpu):
    missing.append(f"GPU identity ({EXPECT_GPU}) not recorded")
product = None
if CASE == "vapo":
    for line in LOG.splitlines():
        at = line.find('{"event": "rl_critic_warmup_product"')
        if at >= 0:
            try:
                product = json.loads(line[at:])
            except ValueError:
                pass
    if product is None:
        missing.append("no rl_critic_warmup_product line")
    else:
        if product.get("warmup_steps") != 50:
            problems.append(f"warm-up steps {product.get('warmup_steps')} != 50")
        shas = {r["critic_init_sha256"] for r in rows}
        if shas != {product.get("critic_sha256")}:
            problems.append(f"main receipts critic_init_sha256 {sorted(map(str, shas))} != product {product.get('critic_sha256')}")
if CASE == "sao":
    notes.append("legacy-path EV +0.1365 (Terminal-Bench, mixed-label forward-only gate); here gsm8k in-training EV; recorded only")
if CASE == "gae-cs":
    notes.append("segments are synthetic (yeto.rl.synthetic_segments: tokens_after 64 for even sample.index, 0 for odd)")
verdict = "FAIL" if problems else ("INCOMPLETE" if missing else "PASS")
print(json.dumps({"verdict": verdict, "case": CASE, "problems": problems, "missing": missing, "notes": notes,
                  "gpu": gpu[:2], "overlay_applied": overlay[:2], "tape": src, "rounds": rows,
                  "warmup_product": product,
                  "ev_by_round": [r["explained_variance"] for r in rows]}, indent=1, sort_keys=True))

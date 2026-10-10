#!/usr/bin/env python3
"""Judge S19 rl-algo-supplement phase-1 G1 runs (S19-ALGOSUP-PRELAUNCH-REVIEW.md §4, phase1-plan §1-2).
Reads tape + launch.log only; missing data -> INCOMPLETE ("失败（证据不全）"), never filled in.
usage: s19-p1-judge.py <run dir> <case> [base run dir] [k3 run dir]"""
import ast, glob, json, math, re, sys
from pathlib import Path
R = Path(sys.argv[1]); CASE = sys.argv[2]
BASE = Path(sys.argv[3]) if len(sys.argv) > 3 else None; K3 = Path(sys.argv[4]) if len(sys.argv) > 4 else None
PIN = __import__("os").environ.get("PIN","64b591a4bec1ffa37fb089d3e3b99c84773b1ffc"); SPECDIR = Path("/home/michael/work/s1-runs/s19-p1-specs")
N = 3
def fin(x): return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
def tape(root):
    f = sorted(glob.glob(str(root / "tape-direct/**/rl-island-0.jsonl"), recursive=True))
    if not f: return [], None
    return [json.loads(l) for l in open(f[0]) if l.strip().startswith("{")], f[0]
def steps(root):
    """per-optimizer-step Miles train dicts and per-rollout rollout dicts from launch.log"""
    out, roll = [], []
    log = (root / "launch.log").read_text(errors="replace") if (root / "launch.log").is_file() else ""
    for line in log.splitlines():
        m = re.search(r"(\{'train/[^{}]*\})", line)
        if m:
            try: out.append(ast.literal_eval(m.group(1)))
            except Exception: pass
        m = re.search(r"(\{'rollout/[^{}]*\})", line)
        if m:
            try: roll.append(ast.literal_eval(m.group(1)))
            except Exception: pass
    # Miles prints each dict twice (logger + print); dedupe consecutive equal dicts
    ded = lambda xs: [x for i, x in enumerate(xs) if i == 0 or x != xs[i - 1]]
    return ded(out), ded(roll), log
ev, src = tape(R); st, ro, LOG = steps(R)
problems, missing, notes, extra = [], [], [], {}
trained = sorted((e for e in ev if e.get("event") == "rl_round_trained"), key=lambda e: e["rollout_id"])
local = {e.get("local_round_id"): e for e in ev if e.get("event") == "rl_local_round"}
rounds = []
for e in trained:
    m = e.get("train_metrics") or {}; lr = local.get(e["rollout_id"] + 1) or {}
    rounds.append({"rollout_id": e["rollout_id"], **{k: m.get(k) for k in sorted(m)}, "grad_norm": lr.get("grad_norm"),
                   "reward_mean": lr.get("reward_mean"), "reward_std": lr.get("reward_std")})
# G
if len(rounds) < N: missing.append(f"{len(rounds)} trained rounds < {N}")
for r in rounds:
    for k in ("loss", "pg_loss", "grad_norm"):
        if r.get(k) is None: missing.append(f"round {r['rollout_id']} {k} missing")
        elif not fin(r[k]): problems.append(f"round {r['rollout_id']} {k}={r[k]} not finite")
for i, s in enumerate(st):
    for k in ("train/loss", "train/grad_norm"):
        if k in s and not fin(s[k]): problems.append(f"step {i} {k}={s[k]} not finite")
if not any(e.get("event") == "rl_learner_finalized" for e in ev): missing.append("no rl_learner_finalized")
z = sorted(set(re.findall(r"zero_grad_norm_with_\w+|rl_zero_gradient_\w+", LOG)))
if z or any(str(e.get("event", "")).startswith("rl_zero_gradient") for e in ev): problems.append(f"zero-gradient verdicts {z}")
sel = next((e for e in ev if e.get("event") == "rl_engine_selected"), None); spec = None
if sel is None: missing.append("no rl_engine_selected")
else:
    if sel.get("miles_commit") != PIN: problems.append(f"miles_commit {sel.get('miles_commit')} != pin")
    spec = json.loads(sel.get("rl/algorithm_spec") or "null")
    want = json.loads((SPECDIR / f"{CASE}.json").read_text()) if CASE != "base" else None
    if want is not None:
        canon = lambda d: json.dumps(d, sort_keys=True)
        if canon(spec) != canon(want): problems.append("event algorithm_spec != registered spec json")
    extra["spec_sha256"] = sel.get("rl/algorithm_spec_sha256")
gpu = [l for l in LOG.splitlines() if "requested A10G" in l]
if not gpu: missing.append("GPU identity line missing")
def step_val(s, key):
    return s.get(key) if s else None
def first_step(root, idx):
    s = steps(root)[0]; return s[idx] if len(s) > idx else None
# case-specific
if CASE in ("cispo", "sapo", "gmpo"):
    b = first_step(BASE, 1) if BASE else None; v = st[1] if len(st) > 1 else None
    if b is None or v is None: missing.append("(d) round-1 step-2 metrics missing (variant or base)")
    else:
        d = {k: [v.get(k), b.get(k)] for k in ("train/pg_loss", "train/grad_norm")}
        extra["d_round1_step2_[variant,base]"] = d
        if all(x[0] == x[1] for x in d.values()): problems.append("(d) round-1 step-2 pg_loss and grad_norm equal base")
    if CASE == "gmpo":
        for r in rounds:
            num, den = r.get("gmpo_clip_num"), r.get("gmpo_clip_den")
            if num is None or den is None: missing.append(f"(e) round {r['rollout_id']} gmpo num/den missing"); continue
            if not (fin(num) and fin(den) and den > 0 and 0 <= num <= den): problems.append(f"(e) num/den {num}/{den}")
            cf = local.get(r["rollout_id"] + 1, {}).get("clip_fraction")
            if cf is None: missing.append(f"(e) round {r['rollout_id']} clip_fraction missing")
            elif abs(cf - num / den) > 1e-6: problems.append(f"(e) clip_fraction {cf} != {num/den}")
    if CASE == "cispo":
        for r in rounds:
            if not fin(r.get("pg_clipfrac")): (missing if r.get("pg_clipfrac") is None else problems).append(f"(f) round {r['rollout_id']} pg_clipfrac {r.get('pg_clipfrac')}")
if CASE.endswith("-tis"):
    if CASE.startswith("gmpo"):
        for r in rounds:
            num, den = r.get("gmpo_clip_num"), r.get("gmpo_clip_den")
            if num is None or den is None or not (fin(num) and fin(den) and den > 0 and 0 <= num <= den):
                (missing if num is None else problems).append(f"(e) round {r['rollout_id']} num/den {num}/{den}")
if CASE == "dual":
    for r in rounds:
        v = r.get("dual_clipfrac")
        if v is None: missing.append(f"round {r['rollout_id']} dual_clipfrac missing")
        elif not (fin(v) and v > 0): problems.append(f"round {r['rollout_id']} dual_clipfrac={v} not >0 finite")
if CASE.startswith("kl-"):
    kl = [r.get("kl_loss") for r in rounds]; extra["kl_loss_per_round"] = kl
    for r in rounds[1:]:
        v = r.get("kl_loss")
        if v is None: missing.append(f"round {r['rollout_id']} kl_loss missing")
        elif not (fin(v) and v > 0): problems.append(f"round {r['rollout_id']} kl_loss={v} not >0 finite")
    if CASE != "kl-k3":
        if K3 is None: missing.append("k3 run dir not given")
        else:
            k3e, _ = tape(K3)
            k3 = [(e.get("train_metrics") or {}).get("kl_loss") for e in sorted((e for e in k3e if e.get("event") == "rl_round_trained"), key=lambda e: e["rollout_id"])]
            extra["k3_kl_loss_per_round"] = k3
            if len(k3) < 2 or len(kl) < 2 or k3[1] is None or kl[1] is None: missing.append("kl vs k3 comparison: values missing")
            elif kl[1] == k3[1]: problems.append("round-2 kl_loss equals k3 run")
if CASE == "opsm-rollout":
    for r in rounds:
        v = r.get("opsm_clipfrac")
        if v is None: missing.append(f"round {r['rollout_id']} opsm_clipfrac missing")
        elif not (fin(v) and v > 0): problems.append(f"round {r['rollout_id']} opsm_clipfrac={v} not >0")
    corr = (spec or {}).get("correction") or {}
    extra["correction_spec"] = corr
    if not corr.get("use_rollout_logprobs"): problems.append("spec correction.use_rollout_logprobs not true (pi_old source)")
if CASE in ("nonorm", "whiten"):
    extra["rollout_advantages_mean"] = [x.get("rollout/advantages") for x in ro]
    b = steps(BASE)[1] if BASE else []
    extra["base_rollout_advantages_mean"] = [x.get("rollout/advantages") for x in b]
    missing.append("advantage variance not emitted by Miles/tape (adv_std null); offline recompute (1e-5) impossible -> 失败（证据不全）")
if CASE == "os":
    o = [{k.split("/")[-1]: x.get(k) for k in x if "over_sampling" in k} for x in ro]; extra["over_sampling_per_round"] = o
    if not o: missing.append("rollout/over_sampling/* missing")
    elif not any((x.get("submitted_groups") or 0) > 4 and (x.get("filtered_groups") or 0) > 0 for x in o):
        problems.append("no round with submitted_groups>4 and filtered_groups>0")
if CASE in ("clip-sym", "clip-hi"):
    extra["step_grad_norms"] = [s.get("train/grad_norm") for s in st]; extra["step_pg_clipfrac"] = [s.get("train/pg_clipfrac") for s in st]
verdict = "FAIL" if problems else ("INCOMPLETE" if missing else "PASS")
print(json.dumps({"verdict": verdict, "case": CASE, "run": str(R), "rounds": rounds, "problems": problems, "missing": missing,
                  "notes": notes, **extra, "miles_commit": (sel or {}).get("miles_commit"), "gpu": gpu[:1], "tape": src,
                  "n_step_dicts": len(st), "watchdog_fired": (R / "WATCHDOG_FIRED").exists()}, indent=1, default=str))

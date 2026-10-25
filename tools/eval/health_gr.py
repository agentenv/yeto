#!/usr/bin/env python3
"""Read-only training-health check of a yeto 2-island async full-parameter GRPO run directory.

Usage:
  python -I tools/eval/health_gr.py RUN_DIR [--json OUT.json] [--no-table] [--ref REF_RUN_DIR]
  (standard library only; from s1-runs/s27-gr-analysis/health_gr.py)

Reads (never writes) RUN_DIR/launch.stream.log, RUN_DIR/head/yeto-syncer.log (falls back to the
"[syncer]" lines inside launch.stream.log), rc.txt.  Details and threshold sources: GAD-REPORT.md.

Log names used (all taken from real logs of s27-gr-* and s27-x*/s27-xk* runs):
  island        launch.stream.log line prefix "[<run>-l<N>-modal]"
  Miles rollout "log_utils.py:NN - rollout N: {...}"   rollout/raw_reward, rollout/response_lengths,
                                                      rollout/log_probs, rollout/rollout_log_probs
  Miles rollout "metrics.py:NN - perf N: {...}"        rollout/zero_std/all_zero_percentage,
                rollout/zero_std/all_one_percentage, rollout/truncated_ratio, rollout/repetition_frac,
                rollout/weight_version/min|max, rollout/num_training_samples
  Miles train   "log_utils.py:NN - step N: {...}"      train/loss, train/pg_loss, train/ppo_kl,
                (same dict also as "model.py:NN - step N")  train/pg_clipfrac, train/entropy_loss,
                train/train_rollout_logprob_abs_diff, train/train_rollout_kl, train/tis_clipfrac,
                train/grad_norm, train/kl_loss (only with --use-kl-loss)
  yeto          YETO_RL_EVENT rl_trajectory_reward (reward, success, group_index, aborted)
                YETO_RL_EVENT rl_full_parameter_round (policy_version, sync/fragment_versions)
  syncer        "outer step step=S fragment=F responders=[..learner_id: N..] gnorm=\"G\" ms=.."

Merge -> island mapping: sync/fragment_versions[f] is the outer step that last merged fragment f
as applied on that island.  New non-zero values in the rl_full_parameter_round event with
policy_version=b are merges applied at that boundary, so rollout b is the first rollout made
with them (rollout/weight_version == rollout id in these runs: lag 0) and rollout b-1 is "before".

Reward at a merge boundary (check merge_jump): Miles rollout/raw_reward, the mean reward of the
whole batch (s27-bb-a1: 512 samples).  The island tape's rl_trajectory_reward records stop at 256
samples per round (TRAJECTORY_REWARD_MAX_PER_ROUND: 16 of 32 prompts), so their accuracy ("acc")
is a half-batch value: on s27-bb-a1 island 1 round 145 -> 146 it fell 0.72 -> 0.31 (FAIL) while
raw_reward fell 0.17 (GBB-CLOSING.md 5.4).  A boundary whose two rollouts lack raw_reward uses
"acc"; the check value counts the pairs of each source.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import statistics
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m|\[(?:3[0-9]|0)m")
ISL = re.compile(r"\[[\w.-]+-l(\d+)-modal\]")
MILES = re.compile(r"(metrics|log_utils|model|train_metric_utils)\.py:\d+ - (perf|rollout|step) (\d+): (\{.*\})")
EVENT = re.compile(r"YETO_RL_EVENT (\{.*\})")
OUTER = re.compile(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?)Z\s+\w+\s+yeto_syncer::\S+ outer step (.*)")

CFG_LINE = re.compile(r"-modal\]\s+(\w+) \.{3,} (.*)$")
CFG_KEYS = {"global_batch_size", "rollout_batch_size", "n_samples_per_prompt", "num_steps_per_rollout",
            "entropy_coef", "use_kl_loss", "kl_coef", "kl_loss_coef", "clip_grad", "eps_clip", "eps_clip_high",
            "tis_clip", "use_tis", "lr", "rollout_max_response_len", "rollout_temperature",
            "use_rollout_logprobs", "advantage_estimator", "observe_training_entropy", "hf_checkpoint"}

# ---------------------------------------------------------------- thresholds
# Basis for every number: GAD-REPORT.md section 4.  "healthy" = Qwen3 references s27-x/s27-x3-k2base-a,
# s27-xk/s27-xk-x3-k2base-b, s27-xk/s27-xk-x4-bf16-a (0.6B, held-out eval stable in GK/X1-X5 reports) and
# s27-xk/s27-xk-x4b-k1g-c (4B, rc=0).  Values in comments are the max seen in those 4 runs.
T = {
    "min_rounds": 4,                         # trend checks need >= 4 trained rounds per island
    # trainer-vs-rollout mismatch, normalised by the entropy proxy (-rollout/log_probs), max / step-0 value
    "lpn_ratio_warn": 1.5, "lpn_ratio_fail": 2.5,          # healthy max 1.37
    "lp_abs_warn": 0.10, "lp_abs_fail": 0.20,              # healthy max 0.0165; Nemotron step 0: 0.050-0.056
    "trkl_ratio_warn": 2.5, "trkl_ratio_fail": 5.0,        # healthy max 1.24; Nemotron 1.06-1.95
    "tis_cf_warn": 0.01, "tis_cf_fail": 0.05,              # healthy max 1e-4; Nemotron step 0 <= 0.0028
    # 1 optimizer step per rollout -> old == current policy -> exactly 0 expected
    "ppo_kl_warn": 1e-3, "pg_cf_warn": 1e-3,               # healthy and Nemotron |ppo_kl| <= 2.1e-10, clipfrac 0
    "gn_spike_warn": 15.0, "gn_spike_fail": 50.0,          # max / island median; healthy max 11.4
    "loss_abs_warn": 0.05,                                 # healthy max 3e-4; Nemotron max 7.5e-3
    "ogn_spike_warn": 6.0, "ogn_spike_fail": 20.0,         # outer gnorm max / run median; healthy max 4.24
    "acc_drop_se_warn": 3.0,     # last third vs first third, empirical SE.  Prompt order is fixed per island,
                                 # so this is confounded by prompt difficulty: healthy k2base-b drops 3.9 SE.
                                 # FAIL only when the last third is all 0.  --ref removes the confound.
    "ref_drop_se_warn": 3.0, "ref_min_pairs": 3,           # paired (same prompts) accuracy vs --ref run
    "zstd_warn": 0.85, "zstd_fail": 0.97,                  # island mean; healthy means 0.62-0.75
    "len_ratio_warn": 0.6, "len_ratio_fail": 0.3,          # last-third / first-third response length
    "trunc_rise_warn": 0.30, "trunc_rise_fail": 0.60,      # absolute trunc is model dependent (4B: 0.97)
    "rep_warn": 0.05, "rep_fail": 0.20,                    # healthy max 0.000
    "ent_low_warn": 0.6, "ent_low_fail": 0.3, "ent_high_warn": 1.6, "ent_high_fail": 3.0,  # healthy 0.73-1.25
    # jumps at merge boundaries (rollout b-1 -> b, merges applied at boundary b).  The reward drop is
    # the whole batch's rollout/raw_reward (half-batch "acc" only without it, see the module doc).
    "jump_acc_warn": 0.20, "jump_acc_fail": 0.35,          # healthy max |d acc| any transition 0.188
    "jump_lpn_warn": 0.25, "jump_lpn_fail": 0.50,          # relative rise; healthy max any transition +0.172
    "jump_trkl_warn": 1.0, "jump_trkl_fail": 2.0,          # relative rise; healthy max +0.33, Nemotron any +0.59
}


def read_lines(path):
    if not os.path.isfile(path):
        return []
    with open(path, "r", errors="replace") as f:
        return [ANSI.sub("", l.rstrip("\n")) for l in f]


def parse_dict(text):
    """Miles prints a python dict; nan/inf become None so they show up as non-finite."""
    t = re.sub(r"(:\s*)-?(nan|inf|NaN|Infinity)\b", r"\1None", text)
    try:
        return ast.literal_eval(t)
    except Exception:
        out = {}
        for k, v in re.findall(r"'([^']+)':\s*(-?[\d.eE+-]+|None)", t):
            try:
                out[k] = None if v == "None" else float(v)
            except ValueError:
                pass
        return out


def num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def mean(xs):
    xs = [x for x in xs if num(x)]
    return statistics.fmean(xs) if xs else None


def median(xs):
    xs = [x for x in xs if num(x)]
    return statistics.median(xs) if xs else None


class Attempts:
    """Island restarts reset ids to 0; give each (island, source) an attempt counter."""

    def __init__(self):
        self.last, self.att = {}, {}

    def key(self, isl, src, rid, strict=False):
        k = (isl, src)
        if k in self.last and (rid < self.last[k] if strict else rid <= self.last[k]):
            self.att[k] = self.att.get(k, 0) + 1
        self.last[k] = rid
        return (isl, self.att.get(k, 0), rid)


TRAIN_KEYS = ["train/loss", "train/pg_loss", "train/ppo_kl", "train/pg_clipfrac", "train/entropy_loss",
              "train/train_rollout_logprob_abs_diff", "train/train_rollout_kl", "train/tis_clipfrac",
              "train/tis", "train/grad_norm", "train/kl_loss"]


def parse(run):
    stream = read_lines(os.path.join(run, "launch.stream.log"))
    sync_path = os.path.join(run, "head", "yeto-syncer.log")
    synclog = read_lines(sync_path)
    rows, traj, fpr = {}, {}, {}
    at = Attempts()
    seen_kinds = set()
    cfg = {}
    for line in stream:
        m = ISL.search(line)
        isl = int(m.group(1)) if m else None
        if isl is not None and len(cfg) < len(CFG_KEYS):
            mc = CFG_LINE.search(line)
            if mc and mc.group(1) in CFG_KEYS:
                cfg.setdefault(mc.group(1), mc.group(2))
        mm = MILES.search(line)
        if mm and isl is not None:
            if re.search(r"rank0*[1-9]\d*\]", line):
                continue  # rank 0 only
            src, kind, sid, body = mm.groups()
            sk = src + ":" + kind
            seen_kinds.add(sk)
            if sk == "model:step":
                sk = "train:step2"  # duplicate of log_utils step; used only if log_utils is missing
            elif sk == "log_utils:step":
                sk = "train:step"
            if sk not in ("log_utils:rollout", "metrics:perf", "train:step", "train:step2"):
                continue
            k = at.key(isl, sk, int(sid))
            rows.setdefault(k, {}).setdefault(sk, parse_dict(body))
            continue
        me = EVENT.search(line)
        if not me:
            continue
        body = me.group(1)
        if '"rl_trajectory_reward"' not in body and '"rl_full_parameter_round"' not in body:
            continue
        try:
            ev = json.loads(body)
        except Exception:
            continue
        i = ev.get("island_id", isl)
        if ev.get("event") == "rl_trajectory_reward":
            k = at.key(i, "traj", int(ev.get("rollout_id", -1)), strict=True)
            traj.setdefault(k, []).append(ev)
        elif ev.get("event") == "rl_full_parameter_round":
            fpr.setdefault(i, []).append(ev)

    # outer steps
    src = synclog if any("outer step" in l for l in synclog) else [l for l in stream if "[syncer]" in l]
    outer = []
    for l in src:
        mo = OUTER.search(l)
        if not mo:
            continue
        kv = dict(re.findall(r"(\w+)=(\"[^\"]*\"|\[[^\]]*\]|\S+)", mo.group(2)))
        g = kv.get("gnorm", "").strip('"')
        try:
            g = float(g)
        except ValueError:
            g = None
        outer.append({"ts": mo.group(1), "step": int(kv.get("step", -1)), "fragment": int(kv.get("fragment", -1)),
                      "responders": sorted(int(x) for x in re.findall(r"learner_id: (\d+)", mo.group(2))),
                      "gnorm": g, "gnorm_raw": kv.get("gnorm")})
    # syncer refuses a non-finite merge (syncer/src/state.rs) and logs the error instead of a gnorm
    nonfinite_merge = [l[:200] for l in src if "outer gradient is non-finite" in l or "produced non-finite state" in l]
    dedup = {}
    for o in outer:
        dedup.setdefault(o["step"], o)
    outer = [dedup[s] for s in sorted(dedup)]

    # merges applied per island per rollout id (attempt 0 events only by order; restarts reset policy_version)
    merges_at = {}
    for i, evs in fpr.items():
        prev, att, lastpv = set(), 0, None
        for ev in evs:
            pv = ev.get("policy_version")
            if lastpv is not None and pv is not None and pv <= lastpv:
                att += 1
            lastpv = pv
            cur = {v for v in ev.get("sync/fragment_versions") or [] if v}
            new = sorted(cur - prev)
            prev = cur
            if new and pv is not None:
                merges_at[(i, att, pv)] = new

    table = []
    keys = set(rows) | set(traj)
    for k in sorted(k for k in keys if k[0] is not None):
        i, att, rid = k
        d = rows.get(k, {})
        ro, pf = d.get("log_utils:rollout", {}), d.get("metrics:perf", {})
        tr = d.get("train:step") or d.get("train:step2") or {}
        ts = [e for e in traj.get(k, []) if not e.get("aborted")]
        rw = [e.get("reward") for e in ts if num(e.get("reward"))]
        sc = [1.0 if e.get("success") else 0.0 for e in ts if "success" in e]
        r = {"island": i, "att": att, "rollout": rid, "n_traj": len(ts),
             "reward_mean": mean(rw), "reward_var": statistics.pvariance(rw) if len(rw) > 1 else None,
             "acc": mean(sc),
             "raw_reward": ro.get("rollout/raw_reward"),
             "resp_len": ro.get("rollout/response_lengths", pf.get("rollout/response_len/mean")),
             "trunc": pf.get("rollout/truncated_ratio", ro.get("rollout/truncated")),
             "rep": pf.get("rollout/repetition_frac"),
             "zero_std": (pf["rollout/zero_std/all_zero_percentage"] + pf["rollout/zero_std/all_one_percentage"])
             if num(pf.get("rollout/zero_std/all_zero_percentage")) and num(pf.get("rollout/zero_std/all_one_percentage")) else None,
             "wv_min": pf.get("rollout/weight_version/min"), "wv_max": pf.get("rollout/weight_version/max"),
             "train_lp": ro.get("rollout/log_probs"), "rollout_lp": ro.get("rollout/rollout_log_probs"),
             "merges_before": merges_at.get(k, []),
             "tasks": sorted({str(e.get("task_id")) for e in ts})}
        for tk in TRAIN_KEYS:
            r[tk.split("/", 1)[1]] = tr.get(tk)
        r["has_train"] = bool(tr)
        r["has_rollout"] = bool(ro or pf or ts)
        # report NaN/None for keys present but non-finite
        r["nonfinite"] = [tk for tk in TRAIN_KEYS if tk in tr and not num(tr[tk])]
        table.append(r)
    return {"run": run, "cfg": cfg, "rc": (read_lines(os.path.join(run, "rc.txt")) or [None])[0],
            "syncer_source": sync_path if src is synclog else "launch.stream.log [syncer]",
            "miles_kinds": sorted(seen_kinds), "table": table, "outer": outer, "nonfinite_merge": nonfinite_merge,
            "fpr_events": {i: len(v) for i, v in fpr.items()}}


def fmt(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, float):
        if x != 0 and (abs(x) < 10 ** -nd or abs(x) >= 1e5):
            return f"{x:.1e}"
        return f"{x:.{nd}f}"
    return str(x)


def print_table(p):
    hdr = ("isl att  r  n  rew_mean rew_var  acc  raw_rew  resp  trunc zstd   -lp(ent) lp_absdiff tr_kl  "
           "tis_cf  ppo_kl  pg_cf  ent_loss gnorm  loss      merged_before")
    print(hdr)
    for r in p["table"]:
        print(f"{r['island']:>3} {r['att']:>3} {r['rollout']:>2} {r['n_traj']:>3} "
              f"{fmt(r['reward_mean']):>7} {fmt(r['reward_var']):>7} {fmt(r['acc'],2):>5} {fmt(r['raw_reward']):>7} "
              f"{fmt(r['resp_len'],0):>5} {fmt(r['trunc'],2):>5} {fmt(r['zero_std'],2):>5} "
              f"{fmt(-r['train_lp'] if num(r['train_lp']) else None):>7} {fmt(r['train_rollout_logprob_abs_diff'],4):>9} "
              f"{fmt(r['train_rollout_kl'],4):>7} {fmt(r['tis_clipfrac'],4):>7} {fmt(r['ppo_kl']):>7} "
              f"{fmt(r['pg_clipfrac']):>6} {fmt(r['entropy_loss']):>7} {fmt(r['grad_norm']):>6} {fmt(r['loss'],4):>8}  "
              f"{','.join(map(str, r['merges_before'])) or ''}")
    print(f"\nouter steps ({len(p['outer'])}, source {p['syncer_source']}):")
    for o in p["outer"]:
        print(f"  step {o['step']:>3} frag {o['fragment']:>3} responders {o['responders']} gnorm {fmt(o['gnorm'],4)} {o['ts']}")




# ---------------------------------------------------------------- health checks
ORDER = {"PASS": 0, "N/A": 0, "WARN": 1, "FAIL": 2}


def worst(*ss):
    return max(ss, key=lambda s: ORDER[s]) if ss else "PASS"


def thirds(xs):
    k = max(1, len(xs) // 3)
    return xs[:k], xs[-k:]


def ent(r):
    return -r["train_lp"] if num(r.get("train_lp")) and r["train_lp"] < 0 else None


def lpn(r):
    e, d = ent(r), r.get("train_rollout_logprob_abs_diff")
    return d / e if num(d) and e else None


def batch_reward_pair(a, r):
    """(before, after, source) of the reward at a boundary: the whole batch's raw_reward when
    both rollouts have it, else the half-batch trajectory accuracy, else (None, None, None)."""
    for key in ("raw_reward", "acc"):
        if num(a.get(key)) and num(r.get(key)):
            return a[key], r[key], key
    return None, None, None


def check(p, ref=None):
    t = [r for r in p["table"] if r["has_train"]]
    cfg = p["cfg"]
    isl = sorted({r["island"] for r in t})
    by = {i: [r for r in t if r["island"] == i] for i in isl}
    out = []

    def add(cid, name, status, value, thr):
        out.append({"id": cid, "name": name, "status": status, "value": value, "threshold": thr})

    # 0 parsing / data presence
    if not t:
        add("data", "训练步数据", "FAIL", "没有解析到任何训练步", "每岛 >= 1 步")
        return out
    nr = {i: len(by[i]) for i in isl}
    mism = [abs(r["reward_mean"] - r["raw_reward"]) for r in p["table"]
            if num(r["reward_mean"]) and num(r["raw_reward"])]
    st = "PASS" if all(n >= T["min_rounds"] for n in nr.values()) else "WARN"
    add("data", "训练步数据", st,
        f"每岛训练轮 {nr}；轨迹奖励均值与 rollout/raw_reward 最大差 {max(mism) if mism else 'n/a'}",
        f"每岛 >= {T['min_rounds']} 轮才做趋势判定（少于则 WARN：数据不足）")
    short = {i for i in isl if nr[i] < T["min_rounds"]}

    # 1 non-finite
    nf = [(r["island"], r["rollout"], r["nonfinite"]) for r in t if r["nonfinite"]]
    onf = [o["step"] for o in p["outer"] if o["gnorm"] is None]
    snf = p.get("nonfinite_merge", [])
    add("finite", "loss / grad_norm / logprob 差 / outer gnorm 有限", "FAIL" if nf or onf or snf else "PASS",
        f"非有限训练字段 {nf[:5]}；非有限 outer gnorm 步 {onf[:5]}；syncer 拒绝非有限合并 {len(snf)} 行",
        "出现任何 nan/inf 即 FAIL")

    # 2 reward / accuracy trend
    sts, vals = [], []
    for i in isl:
        acc = [r["acc"] for r in by[i] if num(r["acc"])]
        if i in short or len(acc) < 2:
            sts.append("WARN"); vals.append(f"l{i}: 数据不足 acc={['%.2f' % x for x in acc]}"); continue
        f, l = thirds(acc)
        mf, ml = statistics.fmean(f), statistics.fmean(l)
        if len(f) >= 2:
            se = math.sqrt(statistics.variance(f) / len(f) + statistics.variance(l) / len(l))
        else:
            n = max(1, statistics.fmean([r["n_traj"] for r in by[i]]))
            se = math.sqrt(max(mf * (1 - mf), 1e-4) / n + max(ml * (1 - ml), 1e-4) / n)
        drop = (mf - ml) / se if se > 0 else 0.0
        s = "WARN" if drop > T["acc_drop_se_warn"] else "PASS"
        if all(x == 0 for x in l) and len(l) >= 2 and mf > 0:
            s = "FAIL"
        sts.append(s)
        vals.append(f"l{i}: 前1/3 {mf:.3f} 后1/3 {ml:.3f} 降 {drop:+.1f} SE；方差均值 "
                    f"{statistics.fmean([r['reward_var'] for r in by[i] if num(r['reward_var'])]):.3f}")
    add("reward", "奖励/正确率趋势", worst(*sts), "；".join(vals),
        f"降幅 > {T['acc_drop_se_warn']} SE 为 WARN（受题目顺序影响，见报告）；后 1/3 全为 0 为 FAIL")

    # 2b paired accuracy vs a reference run on the same prompts
    if ref is not None:
        rmap = {(r["island"], r["att"], r["rollout"]): r for r in ref["table"] if r.get("tasks")}
        diffs = []
        for r in p["table"]:
            q = rmap.get((r["island"], r["att"], r["rollout"]))
            if q and r.get("tasks") and q["tasks"] == r["tasks"] and num(r["acc"]) and num(q["acc"]):
                diffs.append(r["acc"] - q["acc"])
        if len(diffs) < T["ref_min_pairs"]:
            add("reward_ref", "正确率对照 --ref（同题配对）", "N/A", f"同题配对轮 {len(diffs)} 个", f"需 >= {T['ref_min_pairs']} 个")
        else:
            m = statistics.fmean(diffs)
            se = statistics.stdev(diffs) / math.sqrt(len(diffs))
            add("reward_ref", "正确率对照 --ref（同题配对）", "WARN" if se > 0 and -m / se > T["ref_drop_se_warn"] else "PASS",
                f"{len(diffs)} 个同题配对轮，本 run - ref 均值 {m:+.3f}（SE {se:.3f}）；ref={ref['run']}",
                f"均值低于 0 超过 {T['ref_drop_se_warn']} SE 为 WARN")

    # 3 zero-std groups (no GRPO signal)
    sts, vals = [], []
    for i in isl:
        z = [r["zero_std"] for r in by[i] if num(r["zero_std"])]
        if not z:
            sts.append("N/A"); vals.append(f"l{i}: 无字段"); continue
        mz, lz = statistics.fmean(z), statistics.fmean(thirds(z)[1])
        sts.append("FAIL" if lz >= T["zstd_fail"] else "WARN" if mz > T["zstd_warn"] else "PASS")
        vals.append(f"l{i}: 均值 {mz:.2f} 后1/3 {lz:.2f}")
    add("zero_std", "零方差组比例（无 GRPO 信号）", worst(*sts), "；".join(vals),
        f"岛均值 > {T['zstd_warn']} WARN；后 1/3 >= {T['zstd_fail']} FAIL")

    # 4 response length / truncation / repetition
    sts, vals = [], []
    for i in isl:
        L = [r["resp_len"] for r in by[i] if num(r["resp_len"])]
        tr = [r["trunc"] for r in by[i] if num(r["trunc"])]
        rp = [r["rep"] for r in by[i] if num(r["rep"])]
        s = "PASS"
        txt = f"l{i}: len {L[0]:.0f}->{L[-1]:.0f}" if L else f"l{i}: len -"
        if rp:
            mr = max(rp)
            s = worst(s, "FAIL" if mr > T["rep_fail"] else "WARN" if mr > T["rep_warn"] else "PASS")
            txt += f" rep max {mr:.3f}"
        if i in short:
            s = worst(s, "WARN"); txt += " (数据不足)"
        elif L and tr:
            (lf, ll), (tf, tl) = thirds(L), thirds(tr)
            ratio = statistics.fmean(ll) / statistics.fmean(lf)
            rise = statistics.fmean(tl) - statistics.fmean(tf)
            s = worst(s, "FAIL" if ratio < T["len_ratio_fail"] else "WARN" if ratio < T["len_ratio_warn"] else "PASS",
                      "FAIL" if rise > T["trunc_rise_fail"] else "WARN" if rise > T["trunc_rise_warn"] else "PASS")
            txt += f" 后/前1/3 {ratio:.2f} trunc {statistics.fmean(tf):.2f}->{statistics.fmean(tl):.2f}"
        sts.append(s); vals.append(txt)
    add("length", "response 长度 / 截断 / 重复", worst(*sts), "；".join(vals),
        f"长度后/前 < {T['len_ratio_warn']} WARN, < {T['len_ratio_fail']} FAIL；截断比例升 > {T['trunc_rise_warn']} WARN, "
        f"> {T['trunc_rise_fail']} FAIL；repetition_frac > {T['rep_warn']} WARN, > {T['rep_fail']} FAIL")

    # 5 entropy
    el = [r["entropy_loss"] for r in t if num(r["entropy_loss"])]
    ent_logged = any(abs(x) > 0 for x in el)
    sts, vals = [], []
    for i in isl:
        e = [r["entropy_loss"] for r in by[i]] if ent_logged else [ent(r) for r in by[i]]
        e = [x for x in e if num(x)]
        if not e:
            sts.append("N/A"); vals.append(f"l{i}: 无"); continue
        if i in short:
            sts.append("WARN"); vals.append(f"l{i}: 数据不足 {e[0]:.3f}->{e[-1]:.3f}"); continue
        f, l = thirds(e)
        ratio = statistics.fmean(l) / statistics.fmean(f)
        sts.append("FAIL" if ratio < T["ent_low_fail"] or ratio > T["ent_high_fail"] else
                   "WARN" if ratio < T["ent_low_warn"] or ratio > T["ent_high_warn"] else "PASS")
        vals.append(f"l{i}: {statistics.fmean(f):.3f}->{statistics.fmean(l):.3f} (x{ratio:.2f})")
    src = "train/entropy_loss" if ent_logged else "代理量 -rollout/log_probs（entropy_coef=%s，Miles 未算熵）" % cfg.get("entropy_coef")
    add("entropy", "熵（" + src + "）", worst(*sts), "；".join(vals),
        f"后/前 1/3 < {T['ent_low_warn']} 或 > {T['ent_high_warn']} WARN；< {T['ent_low_fail']} 或 > {T['ent_high_fail']} FAIL")

    # 6 KL to reference
    kls = [r["kl_loss"] for r in t if num(r["kl_loss"])]
    if kls:
        add("ref_kl", "KL(对 ref)", "PASS", f"train/kl_loss max {max(kls):.4g}", "仅报告，无健康参考")
    else:
        add("ref_kl", "KL(对 ref)", "N/A",
            f"未记录（use_kl_loss={cfg.get('use_kl_loss')}，kl_coef={cfg.get('kl_coef')}）", "-")

    # 7 KL to old / clip fraction
    single = (cfg.get("num_steps_per_rollout") == "1" and cfg.get("global_batch_size") and
              cfg.get("rollout_batch_size") and cfg.get("n_samples_per_prompt") and
              int(cfg["global_batch_size"]) == int(cfg["rollout_batch_size"]) * int(cfg["n_samples_per_prompt"]))
    pk = max((abs(r["ppo_kl"]) for r in t if num(r["ppo_kl"])), default=None)
    pc = max((r["pg_clipfrac"] for r in t if num(r["pg_clipfrac"])), default=None)
    if pk is None:
        add("old_kl", "KL(对 old) / clip 比例", "N/A", "未记录", "-")
    else:
        bad = pk > T["ppo_kl_warn"] or (pc or 0) > T["pg_cf_warn"]
        add("old_kl", "KL(对 old)=train/ppo_kl，clip 比例=train/pg_clipfrac",
            ("WARN" if bad else "PASS") if single else "PASS",
            f"|ppo_kl| max {pk:.2g}，pg_clipfrac max {pc:.2g}；每 rollout 单次优化步={bool(single)}",
            f"单次优化步配置下应恒为 0：|ppo_kl| > {T['ppo_kl_warn']} 或 clipfrac > {T['pg_cf_warn']} WARN"
            + ("" if single else "（本 run 非单步配置，只报告）"))

    # 8 train-rollout mismatch
    sts, vals = [], []
    for i in isl:
        rs = by[i]
        d = [r["train_rollout_logprob_abs_diff"] for r in rs if num(r["train_rollout_logprob_abs_diff"])]
        n = [x for x in (lpn(r) for r in rs) if num(x)]
        if not d:
            sts.append("N/A"); vals.append(f"l{i}: 无"); continue
        rn = max(n) / n[0] if n and n[0] > 0 else None
        s = worst("FAIL" if max(d) > T["lp_abs_fail"] else "WARN" if max(d) > T["lp_abs_warn"] else "PASS",
                  "PASS" if rn is None else "FAIL" if rn > T["lpn_ratio_fail"] else "WARN" if rn > T["lpn_ratio_warn"] else "PASS")
        sts.append(s)
        vals.append(f"l{i}: 首步 {d[0]:.4f} max {max(d):.4f} (原始 x{max(d) / d[0]:.2f}，除以熵代理后 x{rn:.2f})"
                    if rn else f"l{i}: 首步 {d[0]:.4f} max {max(d):.4f}")
    add("lp_diff", "训练端-rollout 端 logprob 差 train/train_rollout_logprob_abs_diff", worst(*sts), "；".join(vals),
        f"绝对值 > {T['lp_abs_warn']} WARN, > {T['lp_abs_fail']} FAIL；(差/熵代理) 的 max/首步 > {T['lpn_ratio_warn']} WARN, "
        f"> {T['lpn_ratio_fail']} FAIL")
    sts, vals = [], []
    for i in isl:
        k = [r["train_rollout_kl"] for r in by[i] if num(r["train_rollout_kl"])]
        c = [r["tis_clipfrac"] for r in by[i] if num(r["tis_clipfrac"])]
        if not k:
            sts.append("N/A"); vals.append(f"l{i}: 无"); continue
        rk = max(k) / k[0] if k[0] > 0 else 0
        mc = max(c) if c else 0
        sts.append(worst("FAIL" if rk > T["trkl_ratio_fail"] else "WARN" if rk > T["trkl_ratio_warn"] else "PASS",
                         "FAIL" if mc > T["tis_cf_fail"] else "WARN" if mc > T["tis_cf_warn"] else "PASS"))
        vals.append(f"l{i}: kl 首步 {k[0]:.4f} max {max(k):.4f} (x{rk:.2f})；tis_clipfrac max {mc:.4f}")
    add("tr_kl", "KL(rollout||train)=train/train_rollout_kl，tis_clipfrac", worst(*sts), "；".join(vals),
        f"kl max/首步 > {T['trkl_ratio_warn']} WARN, > {T['trkl_ratio_fail']} FAIL；tis_clipfrac > {T['tis_cf_warn']} WARN, "
        f"> {T['tis_cf_fail']} FAIL")

    # 9 grad norm / loss
    sts, vals = [], []
    clip = float(cfg.get("clip_grad") or 1.0)
    for i in isl:
        g = [r["grad_norm"] for r in by[i] if num(r["grad_norm"])]
        if not g:
            sts.append("N/A"); vals.append(f"l{i}: 无"); continue
        md = statistics.median(g)
        sp = max(g) / md if md > 0 else 0
        sts.append(worst("FAIL" if sp > T["gn_spike_fail"] else "WARN" if sp > T["gn_spike_warn"] else "PASS",
                         "WARN" if md > clip else "PASS"))
        vals.append(f"l{i}: 中位 {md:.3f} max {max(g):.3f} (x{sp:.1f})")
    add("grad_norm", "train/grad_norm", worst(*sts), "；".join(vals),
        f"max/岛中位 > {T['gn_spike_warn']} WARN, > {T['gn_spike_fail']} FAIL；岛中位 > clip_grad({clip}) WARN")
    lm = max((abs(r["loss"]) for r in t if num(r["loss"])), default=None)
    add("loss", "train/loss", "N/A" if lm is None else "WARN" if lm > T["loss_abs_warn"] else "PASS",
        f"|loss| max {lm:.2g}" if lm is not None else "无", f"|loss| > {T['loss_abs_warn']} WARN（GRPO 优势零均值，loss 应接近 0）")

    # 10 outer gnorm
    g = [o["gnorm"] for o in p["outer"] if num(o["gnorm"])]
    if not g:
        add("outer_gnorm", "外层合并 gnorm", "N/A" if not p["outer"] else "FAIL", f"outer step {len(p['outer'])} 次", "-")
    else:
        md = statistics.median(g)
        sp = max(g) / md if md > 0 else 0
        one = sum(1 for o in p["outer"] if len(o["responders"]) < 2)
        add("outer_gnorm", "外层合并 gnorm（syncer outer step）",
            "FAIL" if sp > T["ogn_spike_fail"] else "WARN" if sp > T["ogn_spike_warn"] else "PASS",
            f"{len(g)} 步，首 {g[0]:.4f} 末 {g[-1]:.4f} 中位 {md:.4f} max {max(g):.4f} (x{sp:.2f})；单岛响应 {one} 步",
            f"max/中位 > {T['ogn_spike_warn']} WARN, > {T['ogn_spike_fail']} FAIL")

    # 11 jumps at merge boundaries
    jumps, sts = [], []
    for i in isl:
        rows = {r["rollout"]: r for r in by[i] if r["att"] == 0}
        for b in sorted(rows):
            r = rows[b]
            if not r["merges_before"] or (b - 1) not in rows:
                continue
            a = rows[b - 1]
            before, after, source = batch_reward_pair(a, r)
            da = after - before if source else None
            dl = lpn(r) / lpn(a) - 1 if lpn(r) and lpn(a) else None
            dk = (r["train_rollout_kl"] / a["train_rollout_kl"] - 1
                  if num(r["train_rollout_kl"]) and num(a["train_rollout_kl"]) and a["train_rollout_kl"] > 0 else None)
            s = worst("PASS" if da is None else "FAIL" if -da > T["jump_acc_fail"] else "WARN" if -da > T["jump_acc_warn"] else "PASS",
                      "PASS" if dl is None else "FAIL" if dl > T["jump_lpn_fail"] else "WARN" if dl > T["jump_lpn_warn"] else "PASS",
                      "PASS" if dk is None else "FAIL" if dk > T["jump_trkl_fail"] else "WARN" if dk > T["jump_trkl_warn"] else "PASS")
            sts.append(s)
            jumps.append({"island": i, "rollout": b, "merges": r["merges_before"], "acc": (before, after),
                          "acc_source": source, "half_batch_acc": (a["acc"], r["acc"]),
                          "d_acc": da, "lp_diff": (a["train_rollout_logprob_abs_diff"], r["train_rollout_logprob_abs_diff"]),
                          "rel_lpn": dl, "rel_trkl": dk, "status": s})
    # baseline: the same deltas at boundaries without a merge
    base = {"acc": [], "lpn": [], "trkl": []}
    for i in isl:
        rows = {r["rollout"]: r for r in by[i] if r["att"] == 0}
        for b in sorted(rows):
            r = rows[b]
            if r["merges_before"] or (b - 1) not in rows:
                continue
            a = rows[b - 1]
            before, after, _ = batch_reward_pair(a, r)
            if before is not None:
                base["acc"].append(abs(after - before))
            if lpn(r) and lpn(a):
                base["lpn"].append(abs(lpn(r) / lpn(a) - 1))
            if num(r["train_rollout_kl"]) and num(a["train_rollout_kl"]) and a["train_rollout_kl"] > 0:
                base["trkl"].append(abs(r["train_rollout_kl"] / a["train_rollout_kl"] - 1))

    def md(xs):
        return f"{statistics.median(xs):.3f}" if xs else "-"
    sources = {k: sum(1 for j in jumps if j["acc_source"] == k) for k in ("raw_reward", "acc")}
    cmp_txt = (f"奖励来源：全批次 raw_reward {sources['raw_reward']} 个边界，半批次 acc {sources['acc']} 个边界；"
               f"中位|Δ| 合并边界 vs 非合并边界: 奖励 {md([abs(j['d_acc']) for j in jumps if num(j['d_acc'])])} vs {md(base['acc'])}，"
               f"rel(差/熵) {md([abs(j['rel_lpn']) for j in jumps if num(j['rel_lpn'])])} vs {md(base['lpn'])}，"
               f"rel(tr_kl) {md([abs(j['rel_trkl']) for j in jumps if num(j['rel_trkl'])])} vs {md(base['trkl'])}"
               f"（非合并边界 {len(base['acc'])} 个）")
    add("merge_jump", "合并前后跳变（奖励降 / logprob 差升 / KL 升）", worst(*sts) if sts else "N/A",
        f"{len(jumps)} 个合并边界，WARN {sts.count('WARN')}，FAIL {sts.count('FAIL')}；" + cmp_txt,
        f"全批次奖励（rollout/raw_reward；缺失时用半批次 acc）降 > {T['jump_acc_warn']} WARN, > {T['jump_acc_fail']} FAIL；"
        f"(差/熵代理) 相对升 > {T['jump_lpn_warn']} WARN, "
        f"> {T['jump_lpn_fail']} FAIL；train_rollout_kl 相对升 > {T['jump_trkl_warn']} WARN, > {T['jump_trkl_fail']} FAIL")
    p["jumps"] = jumps
    return out


def summarize(checks):
    s = worst(*[c["status"] for c in checks])
    return s


def print_checks(p, checks):
    print("\n== merge boundaries (rollout b-1 -> b; merges = outer steps applied at boundary b)")
    for j in p.get("jumps", []):
        f = lambda x, n=4: fmt(x, n)
        src = {"raw_reward": "raw", "acc": "half"}.get(j.get("acc_source"), "-")
        print(f"  l{j['island']} r{j['rollout']:>2} merges {','.join(map(str, j['merges'])):<14} reward({src}) {f(j['acc'][0],2)}->{f(j['acc'][1],2)} "
              f"lp_diff {f(j['lp_diff'][0])}->{f(j['lp_diff'][1])} rel(lp/ent) {f(j['rel_lpn'],3)} rel(tr_kl) {f(j['rel_trkl'],3)}  {j['status']}")
    print("\n== health checks")
    for c in checks:
        print(f"[{c['status']:<4}] {c['name']}\n        值: {c['value']}\n        阈值: {c['threshold']}")
    nr = {i: sum(1 for r in p["table"] if r["island"] == i and r["has_train"]) for i in sorted({r["island"] for r in p["table"]})}
    print(f"\nOVERALL: {summarize(checks)}  (每岛训练轮 {nr}；只判定\"未见异常\"，不判定\"在学习\"；"
          f"N/A 不计入)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--json", help="write result JSON here (refused inside the run dir)")
    ap.add_argument("--no-table", action="store_true", help="skip the per-step table")
    ap.add_argument("--ref", help="reference run dir with the same prompt order (paired accuracy check)")
    a = ap.parse_args(argv)
    run = os.path.abspath(a.run_dir)
    p = parse(run)
    print(f"run {run}  rc: {p['rc']}")
    print("config: " + ", ".join(f"{k}={p['cfg'][k]}" for k in sorted(p["cfg"]) if k != "hf_checkpoint"))
    print(f"model: {p['cfg'].get('hf_checkpoint')}  miles line kinds: {p['miles_kinds']}\n")
    if not a.no_table:
        print_table(p)
    checks = check(p, parse(os.path.abspath(a.ref)) if a.ref else None)
    print_checks(p, checks)
    if a.json:
        jp = os.path.abspath(a.json)
        if jp.startswith(run + os.sep) or jp == run:
            sys.exit("refusing to write inside the run dir")
        with open(jp, "w") as f:
            json.dump({"parse": p, "checks": checks, "overall": summarize(checks), "thresholds": T}, f, indent=1, default=str)


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    main()

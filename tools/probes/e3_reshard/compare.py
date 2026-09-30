"""A8 judgement (plan-v3 §2.2/§2.3): G1-G6 and go / no-go / inconclusive from the harness evidence.

Tolerances are the pre-registered ones of plan-v3 and are constants here; do
not edit them after a run. ``merge`` merges the per-rank named optimizer
states of one (tp, pp) (fork-M5 ``merge_named_optimizer_states`` in the
container; a test stand-in on CPU).

usage (container): python compare.py <work> [--gbs 16 --mbs 1] -> <work>/RESULT.json
usage (offline):   python compare.py <retrieved work dir> --offline   (events + packed/*.pt)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from harness import read_events  # noqa: E402

GRAD_NORM_REL = 1e-3
UPDATE_REL_L2 = 1e-3
SIGN_AGREEMENT = 0.999
SIGN_MIN_ABS = 1e-8
MOMENT_REL_L2 = 1e-3
SHORT_RUN_REL = 1e-2


def _load(path: str) -> dict:
    import torch

    from yeto.rl.engine.miles_adapter.cut_plugin import from_safe

    return from_safe(torch.load(path, map_location="cpu", weights_only=True))


class Unavailable(LookupError):
    """The state files a check needs were not retrieved (offline run on events only)."""


def gathered(work: Path, arm: str, tag: str, merge: Callable, load: Callable = _load) -> dict:
    packed = Path(work) / "packed" / f"{arm}_{tag}.pt"
    if packed.is_file():  # pack_states.py output (container) or retrieved copy (offline)
        return load(str(packed))
    events = read_events(Path(work) / "arms" / arm)
    dump = next((e for e in events if e["kind"] == "dump" and e["tag"] == tag), None)
    if dump is None:
        raise KeyError(f"{arm}: no dump {tag}")
    if not all(Path(r["path"]).is_file() for r in dump["ranks"]):
        raise Unavailable(f"{arm}/{tag}: neither packed nor per-rank state files are present")
    states = [load(r["path"]) for r in dump["ranks"]]
    import torch

    first = states[0]
    for s in states[1:]:
        for key in ("adapter",):
            for n, v in first[key].items():
                if not torch.equal(v, s[key][n]):
                    raise AssertionError(f"{arm}/{tag}: DP ranks hold different {key} {n}")
    return {"adapter": first["adapter"], "optimizer": merge([s["optimizer_named"] for s in states]),
            "scheduler": first["scheduler"],
            "counters": {"megatron": first.get("megatron_counters") or {},
                         "weight_version": first.get("weight_version")},
            "rank_rng": {json.dumps(s["coord"], sort_keys=True): s["rng_digest"] for s in states}}


def _flat(state: dict, key: str):
    import torch

    return torch.cat([state["optimizer"][n]["tensors"][key].double().reshape(-1) for n in sorted(state["optimizer"])])


def bitwise_problems(a: dict, b: dict) -> list[str]:
    import torch

    out = []
    if set(a["adapter"]) != set(b["adapter"]) or set(a["optimizer"]) != set(b["optimizer"]):
        return ["parameter names differ"]
    for n in a["adapter"]:
        if not torch.equal(a["adapter"][n], b["adapter"][n]):
            out.append(f"adapter {n}")
    for n, ea in a["optimizer"].items():
        eb = b["optimizer"][n]
        for k in ea["tensors"]:
            if not torch.equal(ea["tensors"][k], eb["tensors"][k]):
                out.append(f"{n}.{k}")
        for k in ea["scalars"]:
            if not torch.equal(torch.as_tensor(ea["scalars"][k]), torch.as_tensor(eb["scalars"][k])):
                out.append(f"{n}.{k} (scalar)")
    if a["scheduler"] != b["scheduler"]:
        out.append("scheduler")
    if a.get("counters") != b.get("counters"):
        out.append(f"counters {a.get('counters')} != {b.get('counters')}")
    return out


def _rel(a, b) -> float:
    denom = float(a.norm())
    return float((a - b).norm()) / denom if denom else (0.0 if float(b.norm()) == 0 else math.inf)


def _train(work: Path, arm: str, step: int) -> dict:
    return next(e for e in read_events(Path(work) / "arms" / arm) if e["kind"] == "train" and e["step"] == step)


def _losses(train_event: dict) -> dict[int, str]:
    out = {}
    for rank in train_event["probe"]:
        for r in rank:
            if r["kind"] == "loss":
                for i in r["sample_indices"]:
                    out[i] = r["loss_hex"]
    return out


def _shards(train_event: dict) -> list[dict]:
    return [r for rank in train_event["probe"] for r in rank if r["kind"] == "shard"]


def _ids(shard: dict) -> list[int]:
    """Sample ids of a shard. The fork's scheduled shard carries ``sample_indices`` (global ids) and no
    ``partition`` key (DEV-GATHER run 7); positions are ranks within the step's sorted ids."""
    return list(shard.get("partition") or shard.get("sample_indices") or [])


def _micro_batches(shards: list[dict]) -> set[tuple[int, ...]]:
    out = set()
    for s in shards:
        part = _ids(s)
        for mb in s.get("micro_batch_indices") or []:
            out.add(tuple(sorted(part[i] for i in mb)))
    return out


def g2(work: Path, a: str, b: str, *, gbs: int, mbs: int, dp: dict[str, int]) -> list[str]:
    from yeto.rl.engine.miles_adapter.reshard import scheduled_partitions

    out = []
    ta, tb = _train(work, a, 3), _train(work, b, 3)
    for arm, t in ((a, ta), (b, tb)):
        shards = _shards(t)
        if not shards:
            out.append(f"{arm}: no shard read back")
        for s in shards:
            if not (s.get("has_micro_batch_indices") and s.get("has_num_rollouts")):
                out.append(f"{arm}: shard without micro_batch_indices/num_rollouts (unscheduled split)")
            elif s["num_rollouts"] != [gbs] and s["num_rollouts"] != gbs:
                out.append(f"{arm}: num_rollouts {s['num_rollouts']} != {gbs}")
        ids = sorted(i for s in shards for i in _ids(s))
        pos = {sid: k for k, sid in enumerate(ids)}
        predicted = scheduled_partitions(list(range(len(ids))), dp=dp[arm], global_batch_size=gbs,
                                         micro_batch_size=mbs) if ids else {"partitions": []}
        actual = sorted(([pos[i] for i in _ids(s)] for s in shards), key=lambda p: p[:1])
        if sorted(predicted["partitions"], key=lambda p: p[:1]) != actual:
            out.append(f"{arm}: rank partitions differ from the scheduled prediction")
        for rank in t["probe"]:
            for r in rank:
                if r["kind"] == "normalizer":
                    want = (gbs // mbs // dp[arm], gbs, dp[arm])
                    got = (r["num_microbatches"], r["num_rollouts"], r["loss_parallel_size"])
                    if got != want:
                        out.append(f"{arm}: normalizer {got} != {want}")
                        break
    sa, sb = _shards(ta), _shards(tb)
    if sorted(i for s in sa for i in _ids(s)) != sorted(i for s in sb for i in _ids(s)):
        out.append("consumed sample sets differ")
    if _micro_batches(sa) != _micro_batches(sb):
        out.append("micro-batch composition differs")
    return out


def numeric(ref_before: dict, ref_after: dict, new_before: dict, new_after: dict) -> dict[str, Any]:
    import torch

    da = _flat(ref_after, "param") - _flat(ref_before, "param")
    db = _flat(new_after, "param") - _flat(new_before, "param")
    mask = da.abs() > SIGN_MIN_ABS
    agree = float((torch.sign(da[mask]) == torch.sign(db[mask])).double().mean()) if int(mask.sum()) else 1.0
    return {"update_rel_l2": _rel(da, db), "sign_agreement": agree,
            "exp_avg_rel_l2": _rel(_flat(ref_after, "exp_avg"), _flat(new_after, "exp_avg")),
            "exp_avg_sq_rel_l2": _rel(_flat(ref_after, "exp_avg_sq"), _flat(new_after, "exp_avg_sq"))}


def _grad_norm(work: Path, arm: str, step: int) -> float:
    return float(_train(work, arm, step)["metrics"]["grad_norm"])


def _mean_loss(work: Path, arm: str, step: int) -> float:
    values = [float.fromhex(h) for h in _losses(_train(work, arm, step)).values()]
    return sum(values) / len(values) if values else math.nan


def _restore_rng_problems(work: Path) -> list[str]:
    def rng(arm):
        restore = next(e for e in read_events(Path(work) / "arms" / arm) if e["kind"] == "restore")
        return sorted(r["rng_digest"] for r in restore["ranks"])

    return [] if rng("B1") == rng("B1p") else ["restored RNG digests differ (restore events)"]


def judge(work: Path, *, gbs: int, mbs: int, merge: Callable, load: Callable = _load) -> dict[str, Any]:
    work = Path(work)
    G = lambda arm, tag: gathered(work, arm, tag, merge, load)  # noqa: E731
    dp = {"A1": 1, "A2": 2, "B1": 2, "B1p": 2, "B2": 1, "RT": 1}
    res: dict[str, Any] = {}
    unavailable: list[str] = []

    def state(arm: str, tag: str):
        try:
            return G(arm, tag)
        except Unavailable as exc:
            unavailable.append(str(exc))
            return None

    a1s2, a2s2 = state("A1", "s2"), state("A2", "s2")
    b1r, b2r, rtr = state("B1", "restored"), state("B2", "restored"), state("RT", "restored")
    if None in (a1s2, a2s2, b1r, b2r, rtr):
        res["G1"] = {"pass": None, "unavailable": True}
    else:
        res["G1"] = {"B1_vs_C1": bitwise_problems(a1s2, b1r), "B2_vs_C2": bitwise_problems(a2s2, b2r),
                     "RT_vs_C1": bitwise_problems(a1s2, rtr)}
        res["G1"]["pass"] = not any(res["G1"].values())
    res["G2"] = {"A1_B1": g2(work, "A1", "B1", gbs=gbs, mbs=mbs, dp=dp),
                 "A2_B2": g2(work, "A2", "B2", gbs=gbs, mbs=mbs, dp=dp)}
    res["G2"]["pass"] = not res["G2"]["A1_B1"] and not res["G2"]["A2_B2"]
    b1s3, b1ps3 = state("B1", "s3"), state("B1p", "s3")
    b1pr = state("B1p", "restored")
    g3 = bitwise_problems(b1s3, b1ps3) if None not in (b1s3, b1ps3) else []
    if _losses(_train(work, "B1", 3)) != _losses(_train(work, "B1p", 3)):
        g3.append("per-sample losses differ")
    if _grad_norm(work, "B1", 3) != _grad_norm(work, "B1p", 3):
        g3.append("grad_norm differs")
    if None in (b1r, b1pr):
        g3 += _restore_rng_problems(work)  # same fact from the restore events
    elif b1r["rank_rng"] != b1pr["rank_rng"]:
        g3.append("restored RNG digests differ")
    res["G3"] = {"problems": g3, "pass": not g3, "states_checked": None not in (b1s3, b1ps3)}
    g4: dict[str, Any] = {}
    for ref, new, ref_before, new_before in (("A1", "B1", a1s2, b1r), ("A2", "B2", a2s2, b2r)):
        after_ref, after_new = state(ref, "s3"), state(new, "s3")
        states_ok = None not in (ref_before, new_before, after_ref, after_new)
        n = numeric(ref_before, after_ref, new_before, after_new) if states_ok else {}
        problems = []
        if _losses(_train(work, ref, 3)) != _losses(_train(work, new, 3)):
            problems.append("per-sample loss not bitwise equal")
        for name, ea in (after_ref["optimizer"].items() if states_ok else ()):
            import torch

            if not torch.equal(torch.as_tensor(ea["scalars"].get("step")),
                               torch.as_tensor(after_new["optimizer"][name]["scalars"].get("step"))):
                problems.append(f"step differs for {name}")
                break
        if states_ok and after_ref["scheduler"] != after_new["scheduler"]:
            problems.append("scheduler differs")
        gr, gn = _grad_norm(work, ref, 3), _grad_norm(work, new, 3)
        n["grad_norm_rel"] = abs(gr - gn) / abs(gr) if gr else math.inf
        if not n["grad_norm_rel"] <= GRAD_NORM_REL:
            problems.append(f"grad_norm rel {n['grad_norm_rel']}")
        if states_ok:
            if not n["update_rel_l2"] <= UPDATE_REL_L2:
                problems.append(f"update rel L2 {n['update_rel_l2']}")
            if not n["sign_agreement"] >= SIGN_AGREEMENT:
                problems.append(f"sign agreement {n['sign_agreement']}")
            for key in ("exp_avg_rel_l2", "exp_avg_sq_rel_l2"):
                if not n[key] <= MOMENT_REL_L2:
                    problems.append(f"{key} {n[key]}")
        g4[f"{ref}_{new}"] = {**n, "problems": problems, "states_checked": states_ok}
    g4["pass"] = not any(v["problems"] for v in g4.values() if isinstance(v, dict))
    res["G4"] = g4
    g5 = []
    for arm in ("B1", "B1p", "B2"):
        restore = next(e for e in read_events(work / "arms" / arm) if e["kind"] == "restore")
        if len(restore["rng_mapping"]) != dp[arm] or any(m["source"] != "fresh" for m in restore["rng_mapping"]):
            g5.append(f"{arm}: RNG mapping missing or not fresh")
    seeds = {}
    for arm in ("B1", "B1p"):
        info = next(e for e in read_events(work / "arms" / arm) if e["kind"] == "rank_info" and e["when"] == "start")
        seeds[arm] = sorted((r.get("torch_initial_seed"), r.get("cuda_initial_seed"),
                             r.get("megatron_tracker_digest")) for r in info["ranks"])
    if seeds["B1"] != seeds["B1p"]:
        g5.append("fresh seeds differ between B1 and B1p (not reproducible)")
    if None in (b1r, b1pr):
        g5 += _restore_rng_problems(work)
    elif b1r["rank_rng"] != b1pr["rank_rng"]:
        g5.append("restored RNG digests differ between B1 and B1p")
    res["G5"] = {"problems": g5, "seeds": seeds, "pass": not g5}
    g6 = {}
    for ref, new in (("A1", "B1"), ("A2", "B2")):
        problems = []
        s8r, s8n = state(ref, "s8"), state(new, "s8")
        rel = _rel(_flat(s8r, "param"), _flat(s8n, "param")) if None not in (s8r, s8n) else None
        if rel is not None and not rel <= SHORT_RUN_REL:
            problems.append(f"master rel L2 at step 8: {rel}")
        for step in range(3, 9):
            lr, ln = _mean_loss(work, ref, step), _mean_loss(work, new, step)
            if not (math.isfinite(lr) and math.isfinite(ln)):
                problems.append(f"step {step}: non-finite loss")
            elif lr and abs(lr - ln) / abs(lr) > SHORT_RUN_REL:
                problems.append(f"step {step}: loss rel {abs(lr - ln) / abs(lr)}")
        g6[f"{ref}_{new}"] = {"master_rel_l2": rel, "problems": problems}
    g6["pass"] = not any(v["problems"] for v in g6.values() if isinstance(v, dict))
    res["G6"] = g6
    res["unavailable"] = sorted(set(unavailable))
    if res["unavailable"]:
        # Offline run on events only: report every check that could run, never a decision.
        res["decision"] = "incomplete (state files missing)"
    elif not res["G3"]["pass"]:
        res["decision"] = "inconclusive"
    elif all(res[g]["pass"] for g in ("G1", "G2", "G4", "G5", "G6")):
        res["decision"] = "go"
    else:
        res["decision"] = "no-go"
    return res


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("--gbs", type=int, default=16)
    ap.add_argument("--mbs", type=int, default=1)
    ap.add_argument("--offline", action="store_true",
                    help="retrieved artifacts only (events + packed states): no miles import needed")
    a = ap.parse_args(argv)
    if a.offline:
        def merge(_states):
            raise Unavailable("offline: per-rank dumps are not merged here; use the packed states")
    else:
        from miles.backends.megatron_utils.lora.dp_invariant_state import merge_named_optimizer_states as merge

    result = judge(Path(a.work), gbs=a.gbs, mbs=a.mbs, merge=merge)
    (Path(a.work) / "RESULT.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=repr))
    print(json.dumps({"decision": result["decision"], "unavailable": result["unavailable"],
                      **{g: result[g]["pass"] for g in ("G1", "G2", "G3", "G4", "G5", "G6")}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Pre-registered criteria of each capability (S25 T-EVAL; review SESSION25/gpu-review/T-EVAL.md).

One judge per capability, the same criteria for both backends; only where the
evidence lives differs (:mod:`.evidence`). Each judge fills a
:class:`~.verdict.Verdict` with criteria (pass/fail + evidence) and metrics.
The criteria come from the GPU acceptance of the capability's S25 task: E2 from
T-C1 (C1-C6), E3 from T-C3/T-C3V (1-5), E4 from T-C2 (off-policy) plus a forced
over-age drop, E5 from T-C4 / S24 L5.4 (SCHEDULED stop, resume checks).
"""

from __future__ import annotations

import re
import statistics
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from .evidence import Evidence, finite, gpu_ok, numeric_sum
from .verdict import Verdict, find_all, jsonl

GPU_WANT = {"A10G": "A10", "H100!": "H100", "L4": "L4"}
CONTROL = "island-c"  # verl E5: the continuous control island (not part of the resumed run)


# ------------------------------------------------------------------ shared criteria
def _rc_ok(ev: Evidence, v: Verdict, ignore: tuple[str, ...] = ()) -> None:
    rcs = {k: r for k, r in ev.rc.items() if k not in ignore}
    v.check("rc", "every launch / container exited 0", bool(rcs) and all(r == 0 for r in rcs.values()), rcs)


def _gpu(ev: Evidence, v: Verdict, cfg: dict) -> None:
    want = GPU_WANT[cfg["gpu"]]
    v.check("gpu", f"every container ran on the requested card ({want})", gpu_ok(ev, want), ev.gpu_names)


def _round_metrics(ev: Evidence, trainer: str | None = None) -> dict[str, Any]:
    rounds = ev.of(trainer, "rl_local_round") if trainer else ev.all("rl_local_round")
    walls = ev.of(trainer, "rl_round_wall") if trainer else ev.all("rl_round_wall")
    return {
        "rounds": len(rounds),
        "reward_mean": [round(r.get("reward_mean", float("nan")), 4) for r in rounds],
        "loss": [r.get("loss") for r in rounds],
        "grad_norm": [r.get("grad_norm") for r in rounds],
        "rollout_s": [r.get("rollout_seconds") for r in rounds],
        "train_s": [r.get("train_seconds") for r in rounds],
        "round_wall_s_median": (statistics.median(w["seconds"] for w in walls if "seconds" in w)
                                if any("seconds" in w for w in walls) else None),
    }


def _trained_rounds(ev: Evidence, trainer: str) -> list[int]:
    return sorted({int(e["rollout_id"]) for e in ev.of(trainer, "rl_round_trained") if "rollout_id" in e})


def _released(ev: Evidence, trainer: str) -> list[int]:
    return [int(e["policy_version"]) for e in ev.of(trainer, "rl_policy_released")]


# ------------------------------------------------------------------ E1
def judge_e1(ev: Evidence, v: Verdict, cfg: dict) -> None:
    """Island rollout + trainer: R rounds trained in one island, finite numbers."""
    trainers = ev.trainers()
    v.check("island", "one island trained (rollout and trainer in the same island)", len(trainers) == 1, trainers)
    t = trainers[0] if trainers else ""
    _rc_ok(ev, v)
    rounds = _trained_rounds(ev, t)
    v.check("rounds", f"rounds 0..{cfg['rounds'] - 1} trained once each", rounds == list(range(cfg["rounds"])), rounds)
    local = ev.of(t, "rl_local_round")
    v.check("finite", "loss, grad norm and reward finite every round",
            len(local) == cfg["rounds"] and finite(r.get("loss") for r in local)
            and finite(r.get("grad_norm") for r in local) and finite(r.get("reward_mean") for r in local),
            {"loss": [r.get("loss") for r in local], "reward_mean": [r.get("reward_mean") for r in local]})
    bases = [r.get("base_policy_version") for r in local]
    v.check("versions", "the policy version advances by one per round", bases == list(range(len(bases))) and bases,
            bases)
    _gpu(ev, v, cfg)
    v.metrics.update(_round_metrics(ev, t))


# ------------------------------------------------------------------ E2
def _reclaim(ev: Evidence) -> tuple[int, float | None, str]:
    log = ev.run_dir / ("raw/reclaim.log" if ev.backend == "miles" else "reclaim.log")
    text = log.read_text() if log.is_file() else ""
    m = re.search(r"stop rc=(\d+) \S+ (\d+\.\d+)", text)  # Miles script
    if m:
        return int(m.group(1)), float(m.group(2)), text
    m = re.search(r"rc=(\d+)", text)  # verl driver: the stop time is the line's ISO time
    if m:
        import datetime

        stamp = datetime.datetime.strptime(text.split()[0], "%Y-%m-%dT%H:%M:%SZ")
        return int(m.group(1)), stamp.replace(tzinfo=datetime.timezone.utc).timestamp(), text
    return -1, None, text


def judge_e2(ev: Evidence, v: Verdict, cfg: dict) -> None:
    """A pure sampler island is reclaimed once and replaced; the trainer keeps training."""
    rc, stop_t, text = _reclaim(ev)
    v.check("reclaim", "one reclaim of the sampler island's container, stop rc 0",
            rc == 0 and text.count("stop") >= 1, text.strip().splitlines()[-3:])
    samplers = ev.samplers()
    target = cfg.get("reclaim_component") or (samplers[0] if samplers else "")
    if ev.backend == "verl":  # each sampler container writes its own component dir
        tapes = [ev.islands.get(s, []) for s in samplers]
        merged = [e for t in tapes for e in t]
    else:
        merged = ev.islands.get(target, [])
    leaves = [e for e in merged if e.get("event") == "rl_sampler_leave"]
    joins = [e for e in merged if e.get("event") == "rl_sampler_join"]
    pre = [e for e in leaves if e.get("reason") == "PREEMPTED"]
    v.check("leave", "the reclaimed incarnation wrote rl_sampler_leave reason=PREEMPTED within 30 s of the stop",
            bool(pre) and stop_t is not None and float(pre[0]["time_unix"]) - stop_t <= 30.0,
            {"leaves": [(e.get("incarnation"), e.get("reason")) for e in leaves], "stop_t": stop_t})
    rounds = [e for e in merged if e.get("event") == "rl_sampler_round"]
    ok = False
    detail: dict[str, Any] = {"joins": len(joins)}
    if len(joins) >= 2:
        old, new = joins[0], joins[-1]
        written = [int(e["rollout_id"]) for e in rounds if e["time_unix"] < new["time_unix"] and e.get("groups", 0)]
        ok = (new.get("incarnations") == [old.get("incarnation")]
              and written and new.get("next_rollout_id") == max(written) + 1)
        detail.update(old_written=written, next_rollout_id=new.get("next_rollout_id"),
                      incarnations=new.get("incarnations"))
        if stop_t:
            v.metrics["replacement_join_s"] = round(float(new["time_unix"]) - stop_t, 1)
    v.check("join", "a new incarnation joined, named the reclaimed one and resumed at the next round", ok, detail)
    groups = ev.entries()
    ids = [e.get("sample_id") or e.get("group_id") or e.get("payload") for e in groups]
    per_round: dict[Any, set] = defaultdict(set)
    for e in groups + [r for rows in ev.shards.values() for r in rows if not r.get("end")]:
        sr = e.get("sampler_round") or {}
        if sr:
            per_round[(sr.get("sampler"), sr.get("rollout_id"))].add(sr.get("incarnation"))
    v.check("buffer", "buffer groups unique; each sampler round written by one incarnation only",
            bool(ids) and len(ids) == len(set(ids)) and bool(per_round) and all(len(s) == 1 for s in per_round.values()),
            {"groups": len(ids), "unique": len(set(ids)), "rounds_with_incarnation": len(per_round)})
    trainers = ev.trainers()
    t = trainers[0] if trainers else ""
    trained = _trained_rounds(ev, t)
    during = [e for e in ev.of(t, "rl_round_trained")
              if stop_t and joins and len(joins) >= 2 and stop_t < float(e["time_unix"]) < float(joins[-1]["time_unix"])]
    v.check("trainer", f"the trainer trained all {cfg['rounds']} rounds", len(trained) == cfg["rounds"], trained)
    v.metrics.update(trainer_rounds_during_replacement=len(during), sampler_rounds=len(rounds),
                     buffer_groups=len(ids))
    _rc_ok(ev, v)
    _gpu(ev, v, cfg)


# ------------------------------------------------------------------ E3
def judge_e3(ev: Evidence, v: Verdict, cfg: dict) -> None:
    """N synced trainer islands; learner 0 releases the merged version to a sampler island."""
    trainers = ev.trainers()
    n = cfg["learners"]
    v.check("trainers", f"{n} synced trainer islands trained", len(trainers) == n, trainers)
    applied = [{e["sync/outer_version"]: e["sync/global_policy_hash"] for e in ev.of(t, "rl_policy_apply")
                if "sync/outer_version" in e} for t in trainers]
    v.check("merged_hash", "every outer version has the same merged hash on every trainer (>= rounds versions)",
            len(applied) == n and applied[0] and all(a == applied[0] for a in applied)
            and len(applied[0]) >= cfg["rounds"], {"outer_versions": sorted(applied[0]) if applied else []})
    publisher = next((t for t in trainers if any(e.get("publisher") for e in ev.of(t, "rl_policy_release_role"))), None)
    released = ev.of(publisher, "rl_policy_released") if publisher else []
    others = [t for t in trainers if t != publisher]
    sampler_tapes = [e for s in ev.samplers() for e in ev.islands[s]]
    loaded = [e for e in sampler_tapes if e.get("event") == "rl_sampler_policy_loaded"]
    merged = applied[0] if applied else {}
    rel_versions = [int(e["policy_version"]) for e in released]
    load_versions = [int(e["policy_version"]) for e in loaded]
    v.check("sampler_hash", "the sampler loads released versions in order; loaded hash = merged hash; "
            "releases come from learner 0",
            bool(loaded) and load_versions == sorted(set(load_versions)) and all(x in rel_versions for x in load_versions)
            and all(e.get("policy_hash") == merged.get(int(e["policy_version"])) for e in loaded + released)
            and all((e.get("origin") or {"learner_id": 0}).get("learner_id") == 0 for e in released),
            {"loaded": load_versions, "released": rel_versions})
    gaps = [e for t in trainers for e in ev.of(t, "rl_policy_release_gap")]
    contiguous = rel_versions == list(range(rel_versions[0], rel_versions[0] + len(rel_versions))) if rel_versions else False
    v.check("one_publisher", "one publisher, contiguous release versions (>= 3), no gap; others release nothing",
            publisher is not None and contiguous and len(rel_versions) >= 3 and not gaps
            and not any(ev.of(t, "rl_policy_released") for t in others),
            {"publisher": publisher, "gaps": len(gaps)})
    positions = {m: [p for row in rows for p in row.get("positions", [])] for m, rows in ev.lanes.items()}
    flat = [p for ps in positions.values() for p in ps]
    v.check("lanes", "lane read positions disjoint, no duplicate, p % N == m",
            len(positions) == n and bool(flat) and len(flat) == len(set(flat))
            and all(p % n == m for m, ps in positions.items() for p in ps),
            {m: len(ps) for m, ps in positions.items()})
    steps = [e for e in ev.all("outer_step")] or _head_outer_steps(ev)
    v.check("outer_steps", "the syncer ran >= 3 OuterSteps", len(steps) >= 3, len(steps))
    _rc_ok(ev, v, ignore=("sampler_launch", "sampler"))
    _gpu(ev, v, cfg)
    v.metrics.update(release_seconds=[e.get("seconds") for e in released], buffer_groups=len(ev.entries()),
                     outer_steps=len(steps))


def _head_outer_steps(ev: Evidence) -> list[dict]:
    rows = [r for p in find_all(ev.run_dir / "raw" / "head", "*tape*.jsonl") for r in jsonl(p)]
    return [r for r in rows if r.get("kind") == "outer_step"]


# ------------------------------------------------------------------ E4
def judge_e4(ev: Evidence, v: Verdict, cfg: dict) -> None:
    """Asynchronous rollout (a sampler island ahead of the trainer) with off-policy
    correction; the trainer drops groups older than max_policy_age at least once."""
    trainers = ev.trainers()
    t = trainers[0] if trainers else ""
    age = int(cfg["max_policy_age"])
    trained = _trained_rounds(ev, t)
    v.check("rounds", f"the trainer trained all {cfg['rounds']} rounds", len(trained) == cfg["rounds"], trained)
    rt = ev.of(t, "rl_round_trained")
    tis = [((e.get("mismatch") or {}).get("tis")) for e in rt]
    # verl: per-source importance ratio (offpolicy_source); Miles: the round's TIS mean
    src = ev.of(t, "offpolicy_source")
    ratio = [s.get("ratio_p50") for e in src for s in e.get("sources") or []] or [x for x in tis if x is not None]
    v.check("correction", "off-policy correction metrics present and finite every round",
            len(rt) == cfg["rounds"] and finite(ratio) and finite(
                (e.get("mismatch") or {}).get("abs_diff", (e.get("mismatch") or {}).get("tis_abs")) for e in rt),
            {"ratio_or_tis": ratio})
    # two distinct drop points, each counted once: the round collector's over-age discard
    # (rl_rollout_carry_over, both backends) and the Miles buffer reader passing over
    # stale groups (rl_buffer_stale_skipped, one per round)
    carried = numeric_sum(ev.of(t, "rl_rollout_carry_over"), "over_age_discarded_groups")
    skipped = ev.of(t, "rl_buffer_stale_skipped")
    passed_over = numeric_sum(skipped, "stale_groups")
    # cross-check (the reading before the event existed): buffer positions consumed (the
    # last release's buffer_selected) minus groups trained (round labels; S25 C1 C6 count)
    legacy = None
    releases = ev.of(t, "rl_policy_released")
    if ev.backend == "miles" and releases and releases[-1].get("buffer_selected") is not None:
        legacy = float(releases[-1]["buffer_selected"]) - numeric_sum(ev.of(t, "rl_round_labels"), "rl/groups")
    if ev.backend == "miles":
        v.check("stale_event", "the Miles buffer reader taped rl_buffer_stale_skipped every round",
                len(skipped) == cfg["rounds"], len(skipped))
        if legacy is not None:
            v.check("stale_cross_check", "taped passed-over groups = consumed positions - trained groups",
                    passed_over == legacy, {"event": passed_over, "consumed_minus_trained": legacy})
    dropped = carried + passed_over
    v.check("over_age_drop", "at least one group older than max_policy_age was dropped, not trained", dropped >= 1,
            dropped)
    over = _over_age_trained(ev, t, age)
    v.check("no_over_age_trained", "no trained group is older than max_policy_age", over == 0, over)
    loads = [e for s in ev.samplers() for e in ev.of(s, "rl_sampler_policy_loaded")]
    v.check("sampler_follows", "the sampler island loaded >= 2 trainer releases while it ran ahead",
            len({e.get("policy_version") for e in loads}) >= 2, sorted({e.get("policy_version") for e in loads}))
    _rc_ok(ev, v, ignore=("sampler_launch", "sampler-0", "sampler-1"))
    _gpu(ev, v, cfg)
    v.metrics.update(over_age_discarded_groups=carried, stale_passed_over_groups=passed_over,
                     stale_consumed_minus_trained=legacy, ratio_p50=ratio,
                     **_round_metrics(ev, t))


def _over_age_trained(ev: Evidence, trainer: str, age: int) -> int:
    over = 0
    if ev.backend == "verl":
        for path in find_all(ev.files.get(trainer, ev.run_dir), "verl-fa-rollout-*.jsonl"):
            for r in jsonl(path):
                for versions in (r.get("versions") or {}).values():
                    over += min(versions) < r["rollout_id"] - age
        return over
    # Miles: one rl_group_policies record per trained round, in round order: [group, token,
    # [versions]]; the round trains at its base policy version (rl_local_round)
    bases = [r.get("base_policy_version") for r in ev.of(trainer, "rl_local_round")]
    for record, base in zip(ev.of(trainer, "rl_group_policies"), bases):
        for group in record.get("groups") or []:
            versions = group[2] if isinstance(group, (list, tuple)) and len(group) > 2 else []
            if base is not None and versions:
                over += min(versions) < int(base) - age
    return over


# ------------------------------------------------------------------ E5
def judge_e5(ev: Evidence, v: Verdict, cfg: dict) -> None:
    """A SCHEDULED stop cuts the trainer; a new incarnation resumes from the cut
    and the resume assertions pass."""
    life = ev.all("rl_trainer_lifecycle")
    stopped = [e for e in life if e.get("phase") == "stopped" and (e.get("cut") or {}).get("trigger") == "scheduled"]
    if not stopped:  # verl: the cut record carries the trigger
        stopped = [e for e in ev.all("rl_round_cut") + ev.all("rl_cut_saved")
                   if (e.get("consistent_cut") or e).get("trigger") == "scheduled"]
    v.check("scheduled_stop", "the trainer stopped on a scheduler command with a SCHEDULED cut", bool(stopped),
            [(e.get("incarnation"), (e.get("cut") or {}).get("cut_id", e.get("cut_id"))) for e in stopped])
    resumes = ev.all("rl_resume")
    checks_ok = bool(resumes) and all(
        all(str(val) in ("same", "ok", "outer_recorded") for key, val in (r.get("checks") or {}).items()
            if key in ("run_fingerprint", "consumed_prompt_ids_digest", "cut_shards_sha256", "policy_hash",
                       "ledger_last_round"))
        for r in resumes)
    v.check("resume", "a new incarnation resumed from the cut; fingerprint, ledger, prompts digest, shards and "
            "policy hash all match", checks_ok,
            [{"rollout_id": r.get("rollout_id"), "checks": r.get("checks")} for r in resumes])
    first = [e for e in ev.all("rl_resume_check")]
    v.check("first_step", "the resume checks after the first resumed round pass", bool(first)
            and all(e.get("ok") for e in first), [(e.get("check"), e.get("ok")) for e in first])
    records = [e for t in ev.trainers() if t != CONTROL for e in ev.of(t, "rl_round_trained")]
    trained = sorted({int(e["rollout_id"]) for e in records})
    v.check("rounds", f"rounds 0..{cfg['rounds'] - 1} trained exactly once across incarnations",
            trained == list(range(cfg["rounds"])) and len(records) == cfg["rounds"],
            {"rounds": trained, "records": len(records)})
    if cfg.get("compare_prompt_sets"):  # verl: the resumed rounds read the same prompts as the control
        v.check("data_cursor", "stopped + resumed rounds read the same prompt sets as the continuous control",
                _prompt_sets_match(ev), None)
    _rc_ok(ev, v)
    _gpu(ev, v, cfg)
    restore = [r.get("restore_s") for r in resumes]
    v.metrics.update(restore_s=restore, scheduled_stops=len(stopped), resumes=len(resumes))


def _prompt_sets_match(ev: Evidence) -> bool:
    """verl: each round's prompt set (``verl-rollout-0.jsonl``, sorted-row hash; S25 C4)."""
    def sets(component: str) -> dict[int, str]:
        rows = [r for p in find_all(ev.files.get(component, ev.run_dir / "none"), "verl-rollout-*.jsonl")
                for r in jsonl(p)]
        return {i: r["prompt_set_sha256"] for i, r in enumerate(rows) if r.get("prompt_set_sha256")}

    control = sets(CONTROL)
    a, b = sets("island-a"), sets("island-b")
    resumed = {**a, **{len(a) + i: h for i, h in b.items()}}  # b continues where a stopped
    return bool(control) and len(resumed) == len(control) and resumed == control


JUDGES: dict[str, Callable[[Evidence, Verdict, dict], None]] = {
    "E1": judge_e1, "E2": judge_e2, "E3": judge_e3, "E4": judge_e4, "E5": judge_e5}

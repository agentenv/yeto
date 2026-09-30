#!/usr/bin/env python3
"""Evaluate a C3 rebuild run against the no-rebuild baseline B1 (plan-v6 G-4.4 + E1
plan-3.8-4.4-v2 §3 supplements) from pulled evidence only.

usage: e2_c3_analyze.py <run dir> <baseline run dir> [--expect-outcome RESTORED|REBUILD_OLD]

Inputs per run dir: runs/*/events/*.jsonl (launcher tape), state/elastic-state/{reconfig,ledger}
(unpacked from pulled/state.tgz.b64), state/elastic-state/cuts/*/manifest.json.
Prints JSON: {criteria: {name: {pass, detail}}, pass}.
"""
from __future__ import annotations

import argparse
import base64
import glob
import io
import json
import tarfile
from pathlib import Path


def _jsonl(path: str | Path) -> list[dict]:
    out = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def _complete_sections(text: str) -> dict | None:
    """Top-level sections of a pretty-printed manifest that are complete in a truncated
    copy (a section counts only when the next top-level key follows it)."""
    import re

    keys = [(m.start(), m.group(1)) for m in re.finditer(r'^ "([a-z_]+)": ', text, re.M)]
    out = {}
    for (start, key), (nxt, _) in zip(keys, keys[1:]):
        body = text[start:nxt].rstrip().rstrip(",")
        try:
            out.update(json.loads("{" + body + "}"))
        except ValueError:
            continue
    if not {"progress", "outer", "ledger"} <= set(out):
        return None
    out["_truncated_copy"] = True
    return out


def load(run: Path) -> dict:
    state = run / "state"
    tgz, b64 = run / "pulled" / "state.tgz", run / "pulled" / "state.tgz.b64"
    if not (state / "elastic-state").exists():
        state.mkdir(exist_ok=True)
        try:
            raw = tgz.read_bytes() if tgz.exists() else base64.b64decode(b64.read_text()) if b64.exists() else b""
            if raw:
                with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
                    tf.extractall(state, filter="data")
        except (tarfile.TarError, EOFError, ValueError):
            pass
    tapes = sorted(glob.glob(str(run / "runs" / "*" / "events" / "*.jsonl*")))
    events = _jsonl(tapes[0]) if tapes else []
    es = state / "elastic-state"
    journal = _jsonl(es / "reconfig" / "journal.jsonl") if (es / "reconfig" / "journal.jsonl").exists() else []
    ledger = _jsonl(es / "ledger" / "journal.jsonl") if (es / "ledger" / "journal.jsonl").exists() else []
    manifests = []
    for p in glob.glob(str(es / "cuts" / "*" / "manifest.json")) + glob.glob(str(run / "pulled" / "cut-manifest*.json")) + glob.glob(
            str(run / "pulled" / "elastic-state__cuts__*__manifest.json")):
        text = Path(p).read_text()
        try:
            manifests.append(json.loads(text))
        except ValueError:
            part = _complete_sections(text)  # 8 KiB-truncated pull: only fully present sections
            if part:
                manifests.append(part)
    manifests = list({m.get("cut_id"): m for m in manifests}.values())
    return {"events": events, "journal": journal, "ledger": ledger, "manifests": manifests}


def rounds(events: list[dict]) -> dict[int, dict]:
    return {int(e["rollout_id"]): e for e in events if e.get("event") == "rl_round_trained"}


def analyze(run: Path, base: Path, expect_outcome: str = "RESTORED") -> dict:
    r, b = load(run), load(base)
    crit: dict[str, dict] = {}

    def c(name: str, ok: bool, **detail) -> None:
        crit[name] = {"pass": bool(ok), **detail}

    phases = [j.get("phase") for j in r["journal"] if j.get("kind") == "phase" and j.get("phase")]
    want = ["VALIDATING", "WAIT_SAFE", "REBUILDING_TRAINER", "SUCCEEDED"]
    it = iter(phases)
    c("journal VALIDATING->WAIT_SAFE->REBUILDING_TRAINER->SUCCEEDED", all(p in it for p in want), phases=phases)
    done = [j for j in r["journal"] if j.get("phase") == "SUCCEEDED" and isinstance(j.get("rebuild"), dict)]
    rebuild = done[-1]["rebuild"] if done else {}
    outcome, attempts, generation = rebuild.get("outcome"), rebuild.get("attempts"), rebuild.get("generation")
    c(f"rebuild outcome == {expect_outcome}", outcome == expect_outcome, outcome=outcome)
    c("SwappableActor.generation == 1", generation == 1, generation=generation)
    if expect_outcome == "REBUILD_OLD":
        stages = [a.get("stage") for a in attempts or []]
        c("attempt stages create_training_models, done", stages == ["create_training_models", "done"], stages=stages)
    c("journal rebuild.local_step == 3", rebuild.get("local_step") == 3, local_step=rebuild.get("local_step"))
    m = r["manifests"][0] if len(r["manifests"]) == 1 else None
    c("exactly one cut manifest", m is not None, count=len(r["manifests"]))
    if m:
        c("cut progress.local_step == 3", m["progress"]["local_step"] == 3, local_step=m["progress"]["local_step"])
        c("cut outer.settled", m["outer"].get("settled") is True)
        c("cut ledger carried_over == 0 and ready_unconsumed == 0",
          m["ledger"].get("carried_over") == 0 and m["ledger"].get("ready_unconsumed") == 0,
          ledger={k: m["ledger"].get(k) for k in ("carried_over", "ready_unconsumed")})
    ev = r["events"]
    starts = [e for e in ev if e.get("event") == "rl_driver_start"]
    rebuilt = [e for e in ev if e.get("event") == "rl_trainer_rebuilt"]
    c("rl_driver_start == 1 (no second initialize)", len(starts) == 1, count=len(starts))
    c("rl_trainer_rebuilt == 1", len(rebuilt) == 1, count=len(rebuilt))
    if rebuilt:
        e = rebuilt[0]
        c("rl_trainer_rebuilt policy_version == 3", e.get("policy_version") == 3, policy_version=e.get("policy_version"))
        pubs = [p for p in ev if p.get("event") == "rl_publication"]
        members = set(pubs[0].get("sync/publication_members") or []) if pubs else set()
        c("re-publication to every member", set(e.get("sync/publication_members") or []) == members and bool(members),
          rebuilt_members=e.get("sync/publication_members"), members=sorted(members))
        if m:
            token = str(e.get("rl/policy_token") or "")
            c("re-published policy hash == cut policy hash", token.endswith(m["progress"]["policy_hash"]),
              token=token, cut=m["progress"]["policy_hash"])
    rr, br = rounds(ev), rounds(b["events"])
    c("rounds not replayed (rollout ids 0..5 once each)", sorted(rr) == list(range(6)) and len(
        [e for e in ev if e.get("event") == "rl_round_trained"]) == 6, rounds=sorted(rr))
    c("round 3 trained_sample_ids_sha256 == B1",
      3 in rr and 3 in br and rr[3].get("trained_sample_ids_sha256") == br[3].get("trained_sample_ids_sha256"),
      run=(rr.get(3) or {}).get("trained_sample_ids_sha256"), b1=(br.get(3) or {}).get("trained_sample_ids_sha256"))
    c("data cursor before round 3 (after round 2) == B1",
      2 in rr and 2 in br and rr[2].get("data_cursor") == br[2].get("data_cursor"),
      run=(rr.get(2) or {}).get("data_cursor"), b1=(br.get(2) or {}).get("data_cursor"))
    applied = [j for j in r["ledger"] if j.get("kind") == "optimizer_applied"]
    ids = [int(j.get("rollout_id", -1)) for j in applied]
    c("optimizer_applied never twice (ledger)", len(ids) == len(set(ids)) and bool(ids), rollout_ids=ids,
      note="pulled ledger can lag the last round; completeness is judged from the tape (rounds 0..5 once)")
    return {"criteria": crit, "pass": all(v["pass"] for v in crit.values())}


def analyze_fault(run: Path, *, bound_s: float = 600.0) -> dict:
    """G-4.5 rows 1/2/3/5: bounded RECOVERY_REQUIRED (or REBUILD_OLD success), no consumption
    after the rebuild request, optimizer_applied never twice, cursor never rewound."""
    r = load(run)
    crit: dict[str, dict] = {}

    def c(name: str, ok: bool, **detail) -> None:
        crit[name] = {"pass": bool(ok), **detail}

    phases = [j for j in r["journal"] if j.get("kind") == "phase"]
    start = next((j for j in phases if j.get("phase") == "REBUILDING_TRAINER"), None)
    end = next((j for j in phases if j.get("phase") in ("RECOVERY_REQUIRED", "SUCCEEDED", "REBUILT_OLD",
                                                          "CANCELLED")), None)
    c("rebuild reached REBUILDING_TRAINER", start is not None)
    terminal = (end or {}).get("phase")
    c("terminal RECOVERY_REQUIRED (or a successful REBUILD_OLD)", terminal in ("RECOVERY_REQUIRED", "SUCCEEDED"),
      terminal=terminal, error=(end or {}).get("error"))
    if start and end:
        took = float(end["wall_time"]) - float(start["wall_time"])
        c(f"bounded (<= {bound_s:.0f} s)", took <= bound_s, seconds=round(took, 1))
    ev = r["events"]
    after = [e for e in ev if e.get("event") == "rl_round_trained" and int(e.get("rollout_id", -1)) >= 3]
    if terminal == "RECOVERY_REQUIRED":
        c("no consumption after RECOVERY_REQUIRED", not after, rounds_after=[e.get("rollout_id") for e in after])
    applied = [int(j.get("rollout_id", -1)) for j in r["ledger"] if j.get("kind") == "optimizer_applied"]
    c("optimizer_applied never twice", len(applied) == len(set(applied)), rollout_ids=applied)
    cursors = [e.get("data_cursor", {}).get("sample_offset") for e in ev
               if e.get("event") == "rl_round_trained" and e.get("data_cursor")]
    c("data cursor never rewound", all(b > a for a, b in zip(cursors, cursors[1:])), offsets=cursors)
    return {"criteria": crit, "pass": all(v["pass"] for v in crit.values())}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("baseline", type=Path, nargs="?")
    p.add_argument("--expect-outcome", default="RESTORED")
    p.add_argument("--fault", action="store_true", help="G-4.5 fault-row evaluation (no baseline)")
    ns = p.parse_args()
    res = analyze_fault(ns.run) if ns.fault else analyze(ns.run, ns.baseline, ns.expect_outcome)
    print(json.dumps(res, indent=1, default=repr))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""watchdog case (plan-3.8-4.4-v2 §4 criteria 1-5 + §6) over one run dir.

Usage: analyze_wd.py <dir>   (dir: elastic-state/, gpu.txt, gpu_samples.jsonl, probe_after.txt,
launch.log.gz, ps.txt.gz, compute-apps.txt)
"""
import gzip
import json
import sys
from pathlib import Path


def main(d):
    d = Path(d)
    j = [json.loads(l) for l in (d / "elastic-state/reconfig/journal.jsonl").read_text().splitlines() if l.strip()]
    res = {}
    wd = [r for r in j if r["kind"] == "watchdog"]
    wa = [r for r in j if r["kind"] == "watchdog_action"]
    add = next((r for r in j if r["kind"] == "add_intent"), {})
    new_cells = sorted(add.get("members") or [])
    res["1"] = {"watchdog": wd, "watchdog_action": wa, "new_cells": new_cells,
                "pass": bool(wd) and bool(wa) and any(r.get("killed") for r in wa)}
    ph = {r["phase"]: r for r in j if r["kind"] == "phase"}
    t_wd = wd[0]["wall_time"] if wd else None
    t_term = ph.get("REBUILT_OLD", {}).get("wall_time")
    log = gzip.open(d / "launch.log.gz", "rt").read()
    probe_failed = "liveness probe failed" in log
    res["2"] = {"terminal": [p for p in ph if p in ("SUCCEEDED", "REBUILT_OLD", "CANCELLED", "RECOVERY_REQUIRED")],
                "watchdog_to_terminal_s": (t_term - t_wd) if (t_wd and t_term) else None,
                "liveness_probe_failed_in_log": probe_failed,
                "pass": "REBUILT_OLD" in ph and "SUCCEEDED" not in ph and t_wd is not None
                and t_term is not None and t_term - t_wd <= 60 and not probe_failed}
    ep = json.loads((d / "elastic-state/reconfig/epochs.json").read_text())
    res["3"] = {"members_after": ep.get("members"), "config_epoch": ep.get("config_epoch")}
    # GPU of killed cells: new cells bind the standby bundles G6/G7 (placement map c2/c3)
    g = {}
    for l in (d / "gpu.txt").read_text().splitlines():
        p = [x.strip() for x in l.split(",")]
        g[int(p[0])] = p[1]
    killed_gpus = {g[6], g[7]}
    old_gpus = {g[4], g[5]}
    samples = [json.loads(l) for l in (d / "gpu_samples.jsonl").read_text().splitlines() if l.strip()]
    before = [s for s in samples if t_wd and s["t"] < t_wd]
    old_pids_before = {tuple(a) for s in before[-3:] for a in s["apps"] if a and a[0] in old_gpus}
    after = [s for s in samples if t_term and s["t"] >= t_term]
    old_pids_after = {tuple(a) for s in after for a in s["apps"] if a and a[0] in old_gpus}
    res["3"]["old_engine_pids_before"] = sorted(old_pids_before)
    res["3"]["old_engine_pids_after"] = sorted(old_pids_after)
    res["3"]["pass"] = (ep.get("config_epoch") == 0 and old_pids_before == old_pids_after
                        and bool(old_pids_before))
    clean = [s["t"] - t_term for s in after if not any(a and a[0] in killed_gpus for a in s["apps"])]
    res["4"] = {"killed_gpus": sorted(killed_gpus), "first_clean_after_terminal_s": min(clean) if clean else None,
                "pass": bool(clean) and min(clean) <= 60}
    probe = (d / "probe_after.txt").read_text()
    try:
        pj = json.loads([l for l in probe.splitlines() if l.startswith("{")][-1])
        st = pj.get("cell_statuses", {})
    except Exception:  # noqa: BLE001
        st = None
    bad = None if st is None else {k: v for k, v in st.items()
                                   if any(c.split(":", 1)[-1] in k for c in new_cells) and "Serving" in v}
    res["5"] = {"cell_statuses": st, "killed_cells_serving": bad, "pass": st is not None and not bad}
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1])

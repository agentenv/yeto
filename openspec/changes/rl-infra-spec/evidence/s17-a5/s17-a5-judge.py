"""S17 N7 A5 (rl-infra-spec 3.8 X6) judge. Reads only files already pulled to disk.
usage: python3 s17-a5-judge.py <base run dir> <merged run dir> [out.json]
Each criterion gets its own verdict; dependencies between them are stated, never folded into one PASS/FAIL.
Criteria: gpu-plan-v2 §3 A5 1-5 + plan-3.8-4.4-v2 §2 observations (see S17-I2-A5-PRELAUNCH-REVIEW.md §7 for the merged run)."""
import glob, json, os, re, sys
from collections import defaultdict


def jl(p):
    out = []
    if not os.path.exists(p):
        return out
    for line in open(p, errors="replace"):
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def load(run):
    td = glob.glob(f"{run}/tape-direct/*")[0]
    isl = {i: jl(f"{td}/l{i}/rank0/rl-island-{i}.jsonl") for i in (0, 1)}
    jr = {i: jl(f"{td}/l{i}/rank0/elastic-state/reconfig/journal.jsonl") for i in (0, 1)}
    ep = {}
    for i in (0, 1):
        p = f"{td}/l{i}/rank0/elastic-state/reconfig/epochs.json"
        ep[i] = json.load(open(p)) if os.path.exists(p) else None
    syn = jl(f"{run}/head/yeto-output/yeto-tape.jsonl")
    log = open(f"{run}/launch.log", errors="replace").read() if os.path.exists(f"{run}/launch.log") else ""
    return dict(isl=isl, jr=jr, ep=ep, syn=syn, log=log, td=td)


def ev(L, name):
    return [e for e in L if e.get("event") == name]


def rounds(L):
    by = defaultdict(list)
    for e in ev(L, "rl_round_trained"):
        by[e["rollout_id"]].append(e)
    return by


def tx_terminal(J, req):
    ph = [r for r in J if r.get("kind") == "phase" and r.get("request_id") == req]
    return (ph[-1]["phase"] if ph else None), ph


def main():
    base, merged = load(sys.argv[1]), load(sys.argv[2])
    res = {}
    # ---- C1 ---------------------------------------------------------------------------------------------
    c1 = {}
    J0 = merged["jr"][0]
    up_term, _ = tx_terminal(J0, "up1"); dn_term, _ = tx_terminal(J0, "dn1")
    c1["up1_terminal"], c1["dn1_terminal"] = up_term, dn_term
    c1["a_both_succeeded"] = up_term == "SUCCEEDED" and dn_term == "SUCCEEDED"
    same, detail = True, {}
    for i in (0, 1):
        rb, rm = rounds(base["isl"][i]), rounds(merged["isl"][i])
        for rid in sorted(set(rb) | set(rm)):
            b = rb.get(rid, []); m = rm.get(rid, [])
            key = lambda e: (e.get("trained_sample_ids_sha256"), e.get("trained_groups"), e.get("trained_samples"))
            ok = len(b) == 1 and len(m) == 1 and key(b[0]) == key(m[0])
            detail[f"island{i}_r{rid}"] = {"ok": ok, "base": [key(x) for x in b], "merged": [key(x) for x in m]}
            same &= ok
    c1["b_samples_equal_per_round"] = same
    c1["b_detail"] = detail
    step_ok = True
    for i in (0, 1):
        rm = rounds(merged["isl"][i])
        steps = [rm[r][0].get("train_step") for r in sorted(rm) if len(rm[r]) == 1]
        one_each = all(len(v) == 1 for v in rm.values())
        inc = all(b - a == 1 for a, b in zip(steps, steps[1:])) if all(s is not None for s in steps) else None
        c1[f"c_island{i}_one_step_per_round"] = {"one_round_trained_each": one_each, "train_steps": steps, "increment_by_1": inc}
        step_ok &= one_each and inc is not False
    c1["c_step_unchanged"] = step_ok
    hashes = defaultdict(dict); chain_ok = True
    for i in (0, 1):
        L = merged["isl"][i]
        for a in ev(L, "rl_policy_apply"):
            hashes[a["policy_version"]][i] = a.get("sync/global_policy_hash")
        for a in ev(L, "rl_policy_apply"):
            v, h = a["policy_version"], a.get("sync/global_policy_hash")
            pubs = [p for p in ev(L, "rl_publication") if p.get("policy_version") == v and p["time_unix"] >= a["time_unix"]]
            if not pubs or pubs[0].get("rl/policy_token") != f"yeto:{v}:{h}":
                chain_ok = False
    c1["d_global_hash_equal_both_islands"] = all(len(set(d.values())) == 1 and len(d) == 2 for d in hashes.values())
    c1["d_publication_token_follows_apply"] = chain_ok
    rost = lambda S: [sorted(r.get("responded") or [] if isinstance(r.get("responded"), list) else []) for r in S]
    c1["e_syncer_roster_merged"] = [sorted(m["id"] for m in r.get("expected_members", [])) for r in merged["syn"]]
    c1["e_syncer_roster_base"] = [sorted(m["id"] for m in r.get("expected_members", [])) for r in base["syn"]]
    c1["e_roster_always_both_and_equal_base"] = (c1["e_syncer_roster_merged"] == c1["e_syncer_roster_base"]
                                                 and all(r == [0, 1] for r in c1["e_syncer_roster_merged"]))
    c1["verdict"] = all(c1[k] for k in ("a_both_succeeded", "b_samples_equal_per_round", "c_step_unchanged",
                                        "d_global_hash_equal_both_islands", "d_publication_token_follows_apply",
                                        "e_roster_always_both_and_equal_base"))
    res["C1_switch_samples_step_policy_roster"] = c1
    # ---- C2 ---------------------------------------------------------------------------------------------
    c2 = {}
    resend = {i: ev(merged["isl"][i], "rl_pull_resend") for i in (0, 1)}
    c2["pull_resend_events"] = {i: [{k: e.get(k) for k in ("time_unix", "global_step", "round_attempt", "pulls_received") if k in e}
                                    for e in v] for i, v in resend.items()}
    up_ph = [r for r in J0 if r.get("kind") == "phase" and r.get("request_id") == "up1"]
    lo = min((r["wall_time"] for r in up_ph), default=None); hi = max((r["wall_time"] for r in up_ph), default=None)
    c2["up1_window_wall"] = [lo, hi]
    in_win = {i: [e for e in v if lo is not None and lo - 5 <= e["time_unix"] <= hi + 300] for i, v in resend.items()}
    c2["resends_in_up_window"] = {i: [e.get("global_step") for e in v] for i, v in in_win.items()}
    c2["a_at_least_one_resend"] = sum(len(v) for v in in_win.values()) >= 1
    c2["b_no_strict_failure"] = not any(ev(merged["isl"][i], "rl_strict_failure") for i in (0, 1)) and "rl_strict_failure" not in merged["log"]
    c2["c_island1_finalized"] = bool(ev(merged["isl"][1], "rl_learner_finalized"))
    c2["d_roster_unchanged"] = c1["e_roster_always_both_and_equal_base"]
    up_round = None
    req = [r for r in J0 if r.get("kind") == "request" and r.get("request_id") == "up1"]
    pd = [r for r in J0 if r.get("kind") == "pause_decision" and req and r.get("tx_id") == req[0].get("tx_id")]
    if pd:
        up_round = pd[0].get("rollout_id")
    c2["up_pause_rollout_id"] = up_round
    c2["e_that_round_completed"] = up_round is not None and all(
        len(rounds(merged["isl"][i]).get(up_round, [])) == 1 for i in (0, 1))
    c2["injection_seen_in_log"] = "TEST INJECTION" in merged["log"]
    c2["verdict"] = all(c2[k] for k in ("a_at_least_one_resend", "b_no_strict_failure", "c_island1_finalized",
                                        "d_roster_unchanged", "e_that_round_completed"))
    res["C2_quorum_pull_resend"] = c2
    # ---- C3 ---------------------------------------------------------------------------------------------
    c3 = {}
    fin_term, fin_ph = tx_terminal(J0, "fin1")
    fin_req = [r for r in J0 if r.get("request_id") == "fin1"]
    c3["fin1_records"] = fin_req[-6:]
    td = merged["td"]
    status = glob.glob(f"{td}/l0/rank0/elastic-state/inbox/fin1.status.json")
    c3["fin1_status_file"] = json.load(open(status[0])) if status else None
    txt = json.dumps(fin_req) + json.dumps(c3["fin1_status_file"])
    c3["a_rejected_or_cancelled"] = fin_term in ("CANCELLED", None) and ("finaliz" in txt.lower())
    c3["finalization_records"] = [r for r in J0 if r.get("kind") == "finalization"]
    ep = merged["ep"][0] or {}
    c3["b_config_epoch_end"] = ep.get("config_epoch")
    c3["b_config_epoch_unchanged"] = ep.get("config_epoch") == 2
    c3["verdict"] = c3["a_rejected_or_cancelled"] and c3["b_config_epoch_unchanged"]
    res["C3_finalization_refuses"] = c3
    # ---- plan-3.8-4.4-v2 §2 observations ----------------------------------------------------------------
    obs = {}
    pds = [r for r in J0 if r.get("kind") == "pause_decision"]
    obs["pause_decisions"] = [{k: r.get(k) for k in ("request_id", "tx_id", "rollout_id", "outer_phase", "allowed", "stalls_peers",
                                                      "budget_s", "quorum_timeout_s", "margin", "idle_flow_timeout_s")} for r in pds]
    executed = {r.get("tx_id") for r in J0 if r.get("kind") == "phase" and r.get("phase") in ("SUCCEEDED", "REBUILT_OLD")}
    obs["executed_tx_pause_ok"] = all(r.get("outer_phase") == "round-boundary-published" and r.get("allowed") is True
                                      and r.get("stalls_peers") is True and r.get("budget_s") == 480.0
                                      for r in pds if r.get("tx_id") in executed)
    obs["no_conflicting_or_invalid_pull_permit"] = not re.search(r"conflicting PULL permits|invalid PULL permit", merged["log"])
    pushes = defaultdict(int)
    for r in merged["syn"]:
        for m in r.get("responded", []) if isinstance(r.get("responded"), list) else []:
            pushes[(r.get("step"), m if not isinstance(m, dict) else m.get("id"))] += 1
    obs["syncer_responded_per_step"] = {f"{k[0]}:{k[1]}": v for k, v in sorted(pushes.items(), key=str)}
    res["obs_plan_3.8_4.4_v2_s2"] = obs
    res["C4_idle_flow_probe"] = "not run separately (head on Nebius public IP; see review §2); if C2 shows a dropped flow -> environment-blocked"
    res["C5_report_note"] = "rollout capability only (no trainer edges)"
    out = sys.argv[3] if len(sys.argv) > 3 else f"{sys.argv[2]}/judgment.json"
    json.dump(res, open(out, "w"), indent=1, default=str)
    for k in ("C1_switch_samples_step_policy_roster", "C2_quorum_pull_resend", "C3_finalization_refuses"):
        print(k, "verdict:", res[k]["verdict"])
    print("obs executed_tx_pause_ok:", obs["executed_tx_pause_ok"], "| no bad permits:", obs["no_conflicting_or_invalid_pull_permit"])


if __name__ == "__main__":
    main()

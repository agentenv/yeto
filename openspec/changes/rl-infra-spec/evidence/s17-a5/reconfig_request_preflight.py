"""Local preflight for elastic reconfiguration requests (S17 lesson: two A5 runs lost to request/flag mistakes).
Checks every request body that a run will submit, in order, against the same rules the island controller applies
at plan time, BEFORE any GPU is rented:
  1. epoch chain: request k carries expected_config_epoch == number of earlier requests expected to commit;
  2. deadline_s <= pause budget (strict two-island: quorum_timeout_s x pause_margin, default margin 0.5;
     pause_audit.py uses the request deadline as the expected pause); no quorum flag -> no budget limit;
  3. the target config exists in the --rl-elastic-resources file and the edge (current -> target) is declared there;
  4. an --rl-elastic-attestation file is given, and it certifies that edge (else: "no capability attestation").
Requests listed in --expect-refused (e.g. a finalization probe) must still pass 1-4; only their commit is not assumed.
usage: python3 reconfig_request_preflight.py --triggers JSON --resources FILE [--attestation FILE]
          [--quorum-timeout-s S] [--pause-margin M] [--initial-config C] [--expect-refused fin1,...] [--out FILE]
exit 0 = all pass; 1 = some request fails (details printed and written to --out)."""
import argparse, json, sys


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--triggers", required=True, help='{"triggers": [[phase, rollout_id, request_id, body], ...]}')
    ap.add_argument("--resources", required=True)
    ap.add_argument("--attestation")
    ap.add_argument("--quorum-timeout-s", type=float)
    ap.add_argument("--pause-margin", type=float, default=0.5)
    ap.add_argument("--initial-config", default="T1R1S1")
    ap.add_argument("--expect-refused", default="")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    trig = json.loads(a.triggers)["triggers"]
    res = json.load(open(a.resources))
    configs = set(res.get("configs", {}))
    declared = {(e["source"], e["target"]) for e in res.get("edges", [])}
    att = json.load(open(a.attestation)) if a.attestation else None
    attested = {(e["source"], e["target"]) for e in (att or {}).get("certified_edges", [])}
    budget = a.quorum_timeout_s * a.pause_margin if a.quorum_timeout_s else None
    refused = {x for x in a.expect_refused.split(",") if x}
    cfg, epoch, rows, ok_all = a.initial_config, 0, [], True
    if cfg not in configs:
        rows.append({"request_id": None, "fail": [f"initial config {cfg} not in resources"]}); ok_all = False
    for t in trig:
        req, body = t[2], t[3]; verb = t[4] if len(t) > 4 else "request"
        tgt = body.get("target") if verb == "request" else cfg
        fails = []
        if verb not in ("request", "rebuild"):
            fails.append(f"unknown verb {verb}")
        if verb == "rebuild" and body.get("kind") != "trainer-rebuild":
            fails.append("rebuild body needs kind=trainer-rebuild")
        if body.get("expected_config_epoch") != epoch:
            fails.append(f"epoch chain: expected_config_epoch {body.get('expected_config_epoch')} != {epoch}")
        if budget is not None and float(body.get("deadline_s", 0)) > budget:
            fails.append(f"deadline {body.get('deadline_s')} s > pause budget {budget} s")
        if tgt not in configs:
            fails.append(f"target {tgt} not in resources configs {sorted(configs)}")
        if verb == "request":  # a same-shape trainer rebuild is not an edge and needs no attestation
            if (cfg, tgt) not in declared:
                fails.append(f"edge {cfg}->{tgt} not declared in resources")
            if att is None:
                fails.append("no --rl-elastic-attestation: every transition is refused")
            elif (cfg, tgt) not in attested:
                fails.append(f"edge {cfg}->{tgt} not certified by the attestation")
        rows.append({"request_id": req, "trigger": t[:2], "from": cfg, "to": tgt, "epoch": epoch,
                     "deadline_s": body.get("deadline_s"), "budget_s": budget, "fail": fails})
        ok_all &= not fails
        if req not in refused and verb == "request":
            cfg, epoch = tgt, epoch + 1
    out = {"ok": ok_all, "budget_s": budget, "fingerprint": (att or {}).get("runtime_fingerprint"), "requests": rows}
    for r in rows:
        print(f"preflight {r['request_id']}: {r.get('from')}->{r.get('to')} epoch {r.get('epoch')} deadline {r.get('deadline_s')} "
              f"budget {budget} -> {'ok' if not r['fail'] else 'FAIL: ' + '; '.join(r['fail'])}")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())

"""Summarise s29 phase-timer JSONL. Usage: python3 phase_report.py <dir-with-jsonl>..."""
import glob, json, sys, statistics as st
recs = []
for d in sys.argv[1:]:
    for f in glob.glob(d + "/*.jsonl"):
        recs += [json.loads(l) for l in open(f) if l.strip()]
# index calls per rank
by = {}
for r in sorted(recs, key=lambda r: r["t"]):
    by.setdefault((r["rank"], r["call"]), []).append(r)
ncall = max(len(v) for v in by.values())
for call in ("compute_log_prob", "update_actor"):
    for i in range(ncall):
        rows = [v[i] for (rk, c), v in by.items() if c == call and len(v) > i]
        if not rows: continue
        print(f"\n=== {call} #{i}  level={rows[0]['level']}  ranks={len(rows)}")
        for pp in sorted({r.get("pp") for r in rows}):
            rr = [r for r in rows if r.get("pp") == pp]
            keys = sorted({k for r in rr for k in r["acc"]}, key=lambda k: -st.median([r["acc"].get(k, [0, 0])[0] for r in rr]))
            print(f"  pp{pp} ({len(rr)} ranks)")
            for k in keys:
                vals = [r["acc"].get(k, [0, 0]) for r in rr]
                s = [v[0] for v in vals]
                print(f"    {k:42s} med {st.median(s):9.2f}s  min {min(s):9.2f}  max {max(s):9.2f}  n={vals[0][1]}")
            ls = rr[0].get("lists") or {}
            for k, v in ls.items():
                if v: print(f"    list {k}: n={len(v)} sum={sum(v):.1f} mean={st.mean(v):.3f} p50={st.median(v):.3f} max={max(v):.3f}")

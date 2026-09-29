"""Byte-compare local vs remote argv of every case; exit 0 only if all identical."""
import json, sys
local = {r["case"]: r for r in json.load(open(sys.argv[1]))}
remote = {r["case"]: r for r in json.load(open(sys.argv[2]))}
report, ok = [], set(local) == set(remote)
for case in sorted(local):
    a = json.dumps(local[case].get("argv"), separators=(",", ":")).encode()
    b = json.dumps(remote.get(case, {}).get("argv"), separators=(",", ":")).encode()
    same = a == b and local[case].get("argv") is not None
    ok &= same and remote.get(case, {}).get("ok", False)
    report.append({"case": case, "argv_identical": same, "remote_parse_ok": remote.get(case, {}).get("ok")})
json.dump({"all_ok": ok, "cases": report}, open(sys.argv[3], "w"), indent=1)
print("COMPARE_OK" if ok else "COMPARE_FAILED", sum(r["argv_identical"] for r in report), "/", len(report))
sys.exit(0 if ok else 1)

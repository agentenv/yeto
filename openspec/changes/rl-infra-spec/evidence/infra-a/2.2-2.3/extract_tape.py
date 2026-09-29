"""Rebuild an island tape from YETO_RL_EVENT lines in a head launch log (dedup, order kept)."""
import json, sys
seen, out = set(), []
for line in open(sys.argv[1], errors="replace"):
    i = line.find("YETO_RL_EVENT ")
    if i < 0:
        continue
    raw = line[i + len("YETO_RL_EVENT "):].strip()
    try:
        e = json.loads(raw)
    except json.JSONDecodeError:
        continue
    if raw not in seen:
        seen.add(raw); out.append(e)
with open(sys.argv[2], "w") as f:
    for e in out:
        f.write(json.dumps(e, sort_keys=True) + "\n")
print(len(out))

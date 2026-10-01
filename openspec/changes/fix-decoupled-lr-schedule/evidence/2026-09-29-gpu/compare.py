import json,glob,sys
def get(root):
    out={}
    for i in (0,1):
        f=glob.glob(f"{root}/work/seed-17/*/island-{i}/events.jsonl")[0]
        ev=[json.loads(l) for l in open(f)]
        rr=[e for e in ev if e.get("event")=="rl_local_round"]
        out[i]=dict(lrs=[e.get("applied_lrs") for e in rr],hash=[e["rl/policy_hash"] for e in ev if "rl/policy_hash" in e][-1],delta=[e.get("delta_l2_norm") for e in rr])
    s=[json.loads(l) for l in open(glob.glob(f"{root}/work/seed-17/*/syncer.jsonl")[0])]
    out["syncer"]=[(x["step"],x.get("sync/global_delta_norm")) for x in s if "step" in x]
    return out
L,P=get("dec-legacy"),get("dec-ports")
for n,x in (("legacy",L),("ports",P)):
    print(n, x)
ok=True
for i in (0,1):
    same = [repr(a) for a in L[i]["lrs"]]==[repr(a) for a in P[i]["lrs"]]
    allcfg = all(v==1e-05 for x in (L,P) for l in x[i]["lrs"] for v in l)
    print(f"island {i}: legacy==ports applied_lrs (repr) {same}; all==1e-05 {allcfg}"); ok&=same and allcfg
for n,x in (("legacy",L),("ports",P)):
    nz=all(g not in (None,0,0.0) for _,g in x["syncer"]); h=x[0]["hash"]==x[1]["hash"]
    print(f"{n}: syncer global_delta_norm all nonzero {nz} ({len(x['syncer'])} steps); final hash equal {h} {x[0]['hash'][:12]}"); ok&=nz and h
print("PASS" if ok else "FAIL")

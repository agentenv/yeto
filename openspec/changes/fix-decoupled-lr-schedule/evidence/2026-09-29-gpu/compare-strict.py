# 3.3: compare post-fix strict2 applied LR with pre-fix rerun-strict2 evidence.
import json,glob,re
pre="../../rl-engine-ports/evidence/2026-09-29-rerun-strict2/work/seed-17/yeto-federated-m2"
post=glob.glob("2026-09-29-gpu/strict2-ports/work/seed-17/*")[0]
def logged(p):  # one value per optimizer step (model.py line; log_utils duplicates it)
    return [float(x) for x in re.findall(r"model\.py:\d+ - step \d+: \{.*?'train/lr-pg_0': ([0-9.e-]+)", open(p).read())]
ok=True
for i in (0,1):
    lp, lq = logged(f"{pre}/island-{i}/miles.log"), logged(f"{post}/island-{i}/miles.log")
    flat=[v for l in open(f"{post}/island-{i}/events.jsonl") if '"rl_local_round"' in l for v in json.loads(l)["applied_lrs"]]
    # Miles logs lr after scheduler.step, so applied at step k = configured lr (k=0) or logged value of step k-1.
    pre_applied=[1e-4]+lp[:-1]
    s1=[repr(x) for x in pre_applied]==[repr(x) for x in flat]; s2=[repr(x) for x in lp]==[repr(x) for x in lq]
    print(f"island {i}: logged pre {lp} post {lq} identical={s2}")
    print(f"island {i}: applied pre(derived) {pre_applied} post(applied_lrs) {flat} bitwise_equal={s1} last>0={flat[-1]>0}")
    ok&=s1 and s2 and flat[-1]>0
rc=open("2026-09-29-gpu/strict2-ports/rc").read().strip()
print("post-fix rc",rc,"(a zero-LR invariant trip fails the round => rc!=0)"); ok&=rc=="0"
print("PASS" if ok else "FAIL")

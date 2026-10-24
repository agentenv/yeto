"""S26 E0.3: reward of two runs on the prompts both trained in the same rounds (default 7-9).

Under fully_async the trained prompt set of a round depends on completion order and on the
stale-group filter, so two runs train different prompts; this compares them on the common ones.

    python tools/probes/s26_e03/common_prompts.py RUNS_DIR RUN_A RUN_B [ROUNDS]
"""
import sys,json,collections
def load(r):
    by=collections.defaultdict(lambda: collections.defaultdict(list))
    for l in open(f"{sys.argv[1]}/s26-e03-{r}/launch.ts.log",errors="replace"):
        if '"event":"rl_trajectory_reward"' not in l: continue
        e=json.loads(l.split("YETO_RL_EVENT ",1)[1]); by[e['rollout_id']][e['task_id']].append(e['reward'])
    return by
a,b=load(sys.argv[2]),load(sys.argv[3]); sa=sb=n=0; tasks=0
for k in (tuple(int(x) for x in sys.argv[4].split(",")) if len(sys.argv) > 4 else (7, 8, 9)):
    for t in set(a[k])&set(b[k]):
        sa+=sum(a[k][t]); sb+=sum(b[k][t]); n+=len(a[k][t]); tasks+=1
print(sys.argv[2],sys.argv[3],f"last3 common tasks={tasks} A={sa/n:.3f} B={sb/n:.3f} B-A={(sb-sa)/n:+.3f}")

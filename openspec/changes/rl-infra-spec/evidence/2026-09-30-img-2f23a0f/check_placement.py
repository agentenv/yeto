import json, subprocess
import miles.ray.placement_group as pg
print("miles from", pg.__file__)
print("git HEAD", subprocess.check_output(["git","--git-dir=/tmp/img-verify-2f23a0f/root/miles/.git","rev-parse","HEAD"],text=True).strip())
raw = {"trainer":[0,1],"rollout":[2,3],"standby":[4,5],
       "rollout_cells":[{"name":"r0","bundles":[2]},{"name":"r1","bundles":[3]},
                        {"name":"s0","bundles":[4,5],"start":False},{"name":"u0","start":False}]}
pm = pg.parse_placement_map(json.dumps(raw))
pg.validate_placement_map(pm, trainer_num_gpus=2, rollout_num_gpus=2)
print("OK", pm)
for c in pm.rollout_cells: print("  binding", c.name, pg.rollout_cell_binding(pm, c))
bad = [
 ("started unbound", {**raw, "rollout_cells":[{"name":"x"}]}),
 ("stopped before started", {**raw, "rollout_cells":[{"name":"a","bundles":[2],"start":False},{"name":"b","bundles":[3]}]}),
 ("dup name", {**raw, "rollout_cells":[{"name":"a","bundles":[2]},{"name":"a","bundles":[3]}]}),
 ("trainer bundle", {**raw, "rollout_cells":[{"name":"a","bundles":[0]}]}),
 ("unknown key", {**raw, "rollout_cells":[{"name":"a","bundles":[2],"gpu":1}]}),
]
for label, r in bad:
    try:
        pg.validate_placement_map(pg.parse_placement_map(json.dumps(r)), trainer_num_gpus=2, rollout_num_gpus=2); print("UNEXPECTED ACCEPT", label)
    except AssertionError as e: print("rejected as expected:", label, "->", str(e)[:120])
print("no-cells map:", pg.parse_placement_map('{"trainer":[0,1],"rollout":[2,3]}').rollout_cells)

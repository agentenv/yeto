# in-container fork probe (run with the learner's interpreter/env): prints one JSON line.
# usage: fork_probe.py status | stale <url> | oldepoch
import json, sys, time, urllib.request
import ray
mode = sys.argv[1] if len(sys.argv) > 1 else "status"
res = {"t": time.time(), "mode": mode}
import os
ray.init(address=os.environ.get("RAY_ADDRESS", "auto"), logging_level="ERROR")
named = ray.util.list_named_actors(all_namespaces=True)
ns = next((a["namespace"] for a in named if a["name"] == "ray_worker_manager"), None)
mgr = ray.get_actor("ray_worker_manager", namespace=ns)
cells = ray.get(mgr.get_cell_infos.remote(pool_ids=["inference-controller"]))
w = ray.get(mgr.get_worker_infos.remote(next(iter(cells))))[0]
ic = ray.get(mgr.get_actor_handle.remote(w.name, expected_generation=w.generation))
st = ray.get(ic.get_membership_status.remote()); res["membership"] = st
res["versions"] = ray.get(ic.get_cells_weight_versions.remote())
res["cell_statuses"] = {k: str(v) for k, v in ray.get(ic.get_cell_statuses.remote()).items()}
if mode == "stale":
    try:
        req = urllib.request.Request(sys.argv[2].rstrip("/") + "/update_weight_version",
                                     data=json.dumps({"new_version": "stale-ack-probe"}).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=5) as r:
            res["stale_ack"] = {"status": r.status}
    except Exception as e:  # noqa: BLE001
        res["stale_ack"] = {"error": repr(e)[:300]}
    res["versions_after"] = ray.get(ic.get_cells_weight_versions.remote())
    res["versions_equal"] = res["versions"] == res["versions_after"]
if mode == "oldepoch":
    ep = int(st.get("epoch", 0))
    try:
        ray.get(ic.start_update_weights.remote(members=sorted(res["versions"])[:1], expected_epoch=ep - 1))
        res["refused"] = False
    except Exception as e:  # noqa: BLE001
        res["refused"] = "MembershipEpochMismatch" in repr(e); res["error"] = repr(e)[:300]
print(json.dumps(res, default=str))

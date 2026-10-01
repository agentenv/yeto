# in-container fork probe (run with the learner's interpreter/env): prints one JSON line.
# usage: fork_probe.py status | stale <url> | oldepoch
# Self-attestation (written into the output JSON under "attest" and enforced; exit 3 if any fails):
#   ray_version_ok   : ray.__version__ != 2.9.3 (the old system ray must not be the one we talk to)
#   gcs_6379_ok      : the connected GCS address (runtime context) ends with :6379 AND equals the address we asked for
#   miles_present_ok : the Miles actor "ray_worker_manager" (and its namespace) exists in this Ray
import json, os, sys, time, urllib.request


def attest(ray, asked):
    a = {"ray_version": getattr(ray, "__version__", None), "asked_address": asked}
    try:
        a["gcs_address"] = ray.get_runtime_context().gcs_address
    except Exception as e:  # noqa: BLE001
        a["gcs_address"] = None; a["gcs_error"] = repr(e)[:200]
    try:
        named = ray.util.list_named_actors(all_namespaces=True)
        a["miles_namespace"] = next((x["namespace"] for x in named if x["name"] == "ray_worker_manager"), None)
        a["named_actors"] = sorted(x["name"] for x in named)[:20]
    except Exception as e:  # noqa: BLE001
        a["miles_namespace"] = None; a["named_error"] = repr(e)[:200]
    a["ray_version_ok"] = a["ray_version"] is not None and a["ray_version"] != "2.9.3"
    a["gcs_6379_ok"] = bool(a["gcs_address"]) and str(a["gcs_address"]).endswith(":6379") and a["gcs_address"] == asked
    a["miles_present_ok"] = a["miles_namespace"] is not None
    a["attested"] = a["ray_version_ok"] and a["gcs_6379_ok"] and a["miles_present_ok"]
    return a


def main(argv, ray=None, out=print):
    if ray is None:
        import ray  # noqa: PLW0127
    mode = argv[1] if len(argv) > 1 else "status"
    res = {"t": time.time(), "mode": mode}
    asked = os.environ.get("RAY_ADDRESS", "")
    ray.init(address=asked or "auto", logging_level="ERROR")
    res["attest"] = attest(ray, asked)
    if not res["attest"]["attested"]:
        res["probe_attested"] = False
        out(json.dumps(res, default=str)); return 3
    res["probe_attested"] = True
    named = ray.util.list_named_actors(all_namespaces=True)
    ns = res["attest"]["miles_namespace"]
    mgr = ray.get_actor("ray_worker_manager", namespace=ns)
    cells = ray.get(mgr.get_cell_infos.remote(pool_ids=["inference-controller"]))
    w = ray.get(mgr.get_worker_infos.remote(next(iter(cells))))[0]
    ic = ray.get(mgr.get_actor_handle.remote(w.name, expected_generation=w.generation))
    st = ray.get(ic.get_membership_status.remote()); res["membership"] = st
    res["versions"] = ray.get(ic.get_cells_weight_versions.remote())
    res["cell_statuses"] = {k: str(v) for k, v in ray.get(ic.get_cell_statuses.remote()).items()}
    if mode == "stale":
        try:
            req = urllib.request.Request(argv[2].rstrip("/") + "/update_weight_version",
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
    out(json.dumps(res, default=str)); return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

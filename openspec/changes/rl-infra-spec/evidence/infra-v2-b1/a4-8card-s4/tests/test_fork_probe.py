"""CPU test of fork_probe self-attestation with a stub ray module. run: python3 tests/test_fork_probe.py"""
import io, json, os, sys, types, unittest
sys.path[:0] = [os.path.join(os.path.dirname(__file__), ".."), os.path.join(os.path.dirname(__file__), "..", "scripts")]
import fork_probe


class _R:
    def __init__(self, v): self._v = v
    def remote(self, *a, **k): return self._v


def make_ray(version="2.47.1", gcs="10.0.0.5:6379", actors=({"name": "ray_worker_manager", "namespace": "miles"},)):
    ray = types.SimpleNamespace(__version__=version)
    ray.init = lambda **k: None
    ray.util = types.SimpleNamespace(list_named_actors=lambda all_namespaces=True: list(actors))
    ray.get_runtime_context = lambda: types.SimpleNamespace(gcs_address=gcs)
    w = types.SimpleNamespace(name="w", generation=1)
    ic = types.SimpleNamespace(get_membership_status=_R({"epoch": 1}), get_cells_weight_versions=_R({"c0": "v1"}),
                               get_cell_statuses=_R({"c0": "READY"}))
    mgr = types.SimpleNamespace(get_cell_infos=_R({"c0": 1}), get_worker_infos=_R([w]), get_actor_handle=_R(ic))
    ray.get_actor = lambda name, namespace=None: mgr
    ray.get = lambda x: x
    return ray


class T(unittest.TestCase):
    def run_probe(self, ray, addr="10.0.0.5:6379"):
        os.environ["RAY_ADDRESS"] = addr; buf = []
        rc = fork_probe.main(["fork_probe.py", "status"], ray=ray, out=buf.append)
        return rc, json.loads(buf[-1])

    def test_ok(self):
        rc, r = self.run_probe(make_ray())
        self.assertEqual(rc, 0); self.assertTrue(r["probe_attested"]); self.assertIn("membership", r)
        a = r["attest"]
        self.assertEqual((a["ray_version"], a["gcs_address"], a["miles_namespace"]), ("2.47.1", "10.0.0.5:6379", "miles"))
        self.assertTrue(a["ray_version_ok"] and a["gcs_6379_ok"] and a["miles_present_ok"])

    def test_old_ray_rejected(self):
        rc, r = self.run_probe(make_ray(version="2.9.3"))
        self.assertEqual(rc, 3); self.assertFalse(r["probe_attested"]); self.assertFalse(r["attest"]["ray_version_ok"]); self.assertNotIn("membership", r)

    def test_wrong_port_rejected(self):
        rc, r = self.run_probe(make_ray(gcs="10.0.0.5:6380"), addr="10.0.0.5:6380")
        self.assertEqual(rc, 3); self.assertFalse(r["attest"]["gcs_6379_ok"])

    def test_address_mismatch_rejected(self):
        rc, r = self.run_probe(make_ray(gcs="10.0.0.9:6379"))
        self.assertEqual(rc, 3); self.assertFalse(r["attest"]["gcs_6379_ok"])

    def test_no_miles_actor_rejected(self):
        rc, r = self.run_probe(make_ray(actors=({"name": "other", "namespace": "x"},)))
        self.assertEqual(rc, 3); self.assertFalse(r["attest"]["miles_present_ok"])


if __name__ == "__main__":
    unittest.main()

"""Scenarios for yeto.sky_patches.verda against sky's REAL Verda provisioner.

Run by tests/test_verda_provider.py in a Python that has SkyPilot 0.13
installed (the yeto test venv has none): the real
``sky.provision.verda.instance`` module is imported, its Verda client is
replaced by an in-memory fake, the patch is applied, and each scenario
drives sky's own code paths. Prints one JSON object with the results.

    python tests/verda_sky_scenarios.py <repo root>
"""

from __future__ import annotations

import json
import os
import sys
import tempfile


class FakeInstance:
    def __init__(self, d):
        self.instance_id = d["id"]
        self.status = d["status"]
        self.hostname = d["hostname"]
        self.ip = d.get("ip", "198.51.100.1")


class FakeVerda:
    """In-memory Verda: `create_fails` makes instance_create raise 503."""

    def __init__(self, instances, create_fails=False, list_empty=False):
        self.instances = {d["id"]: dict(d) for d in instances}
        self.create_fails = create_fails
        self.list_empty = list_empty
        self.deleted: list[str] = []
        self.created: list[str] = []
        self._n = 0

    def instances_get(self):
        if self.list_empty:
            return []
        return [FakeInstance(d) for d in self.instances.values()]

    def instance_get(self, iid):
        if iid not in self.instances:
            raise RuntimeError("HTTP 404 not_found")
        return FakeInstance(self.instances[iid])

    def ssh_keys_get(self):
        class K:
            id, public_key = "k1", "ssh-ed25519 AAAA test"

        return [K()]

    def instance_create(self, payload):
        if self.create_fails:
            raise RuntimeError("503 Service Unavailable: no capacity for this instance type")
        self._n += 1
        iid = f"new-{self._n}"
        self.instances[iid] = {"id": iid, "status": "running", "hostname": payload["hostname"]}
        self.created.append(iid)
        return FakeInstance(self.instances[iid])

    def instance_action(self, instance_id, action):
        assert action == "delete"
        self.deleted.append(instance_id)
        self.instances[instance_id]["status"] = "deleted"


def main(repo: str) -> dict:
    sys.path.insert(0, repo)
    os.environ["YETO_VERDA_IDS_PATH"] = os.path.join(tempfile.mkdtemp(), "ids.json")
    import time

    time.sleep = lambda s: None  # sky polls every 5 s
    import sky
    from sky.provision import common
    from sky.provision.verda import instance as m
    from sky.utils import status_lib

    import yeto.sky_patches as patches
    from yeto.sky_patches import verda as vp

    out: dict = {"sky_version": sky.__version__}
    orig = {n: getattr(m, n) for n in ("run_instances", "query_instances", "terminate_instances")}

    # Version guard first: a non-verified version leaves sky untouched.
    ok, why = vp.apply(m, version="0.99.0")
    out["guard_rejects_other_version"] = (not ok) and all(getattr(m, n) is f for n, f in orig.items())

    status = patches.install()
    out["install_status"] = status
    out["patched"] = vp.is_patched(m)
    import sky.provision.verda as pkg

    out["package_reexports_patched"] = pkg.query_instances is m.query_instances

    def config(count):
        return common.ProvisionConfig(
            provider_config={"zones": None}, authentication_config={}, docker_config={},
            node_config={"InstanceType": "1L40S.20V", "PublicKey": "ssh-ed25519 AAAA test"},
            count=count, tags={}, resume_stopped_nodes=True, ports_to_open_on_launch=None,
        )

    def provision(cluster, count):
        """What sky's provisioner does: run, and on failure terminate."""
        try:
            m.run_instances("FIN-01", cluster, cluster, config(count))
            return "ok"
        except Exception as exc:  # noqa: BLE001
            m.terminate_instances(cluster, {})
            return f"failed: {type(exc).__name__}"

    # 1. Same-prefix sibling survives teardown / status of another cluster.
    fake = FakeVerda([
        {"id": "a", "status": "running", "hostname": "vfix-x-head"},
        {"id": "b", "status": "running", "hostname": "vfix-xb-head"},
    ])
    m.verda = fake
    q = m.query_instances("vfix-x", "vfix-x", {}, True)
    m.terminate_instances("vfix-x", {})
    out["sibling_prefix"] = {"query": sorted(q), "deleted": fake.deleted}

    # 2. Relaunch while the node still runs: reused, nothing created/deleted.
    fake = FakeVerda([{"id": "r1", "status": "running", "hostname": "vfix-r-head"}], create_fails=True)
    m.verda = fake
    out["relaunch_running"] = {"result": provision("vfix-r", 1), "created": fake.created,
                               "deleted": fake.deleted, "r1": fake.instances["r1"]["status"]}

    # 3. Creation fails (503) with a running node present: it is kept.
    fake = FakeVerda([{"id": "k1", "status": "running", "hostname": "vfix-k-head"}], create_fails=True)
    m.verda = fake
    out["create_fails_keeps_running"] = {"result": provision("vfix-k", 2), "deleted": fake.deleted,
                                         "k1": fake.instances["k1"]["status"]}

    # 4. Unknown / new Verda states map to INIT instead of KeyError.
    fake = FakeVerda([
        {"id": "u1", "status": "ordered", "hostname": "vfix-u-head"},
        {"id": "u2", "status": "some_future_state", "hostname": "vfix-u-worker"},
    ])
    m.verda = fake
    q = m.query_instances("vfix-u", "vfix-u", {}, True)
    out["unknown_status"] = {k: str(v[0]) for k, v in sorted(q.items())}

    # 5. Empty listing but a recorded id: re-checked by id, still UP.
    vp.remember_ids("vfix-e", ["e1"])
    fake = FakeVerda([{"id": "e1", "status": "running", "hostname": "vfix-e-head"}], list_empty=True)
    m.verda = fake
    q = m.query_instances("vfix-e", "vfix-e", {}, True)
    out["empty_recheck"] = {k: str(v[0]) for k, v in q.items()}

    # 6. The dispatcher's argument order (cluster_name, name_on_cloud, ...).
    from sky import provision as provision_lib

    fake = FakeVerda([{"id": "d1", "status": "running", "hostname": "vfix-d-head"}])
    m.verda = fake
    q = provision_lib.query_instances("verda", "VFIX-D-display", "vfix-d", {}, non_terminated_only=True)
    out["dispatcher"] = {k: str(v[0]) for k, v in q.items()}

    # 7. Fresh launch: creates exactly one node and reports it.
    fake = FakeVerda([])
    m.verda = fake
    out["fresh_launch"] = {"result": provision("vfix-f", 1), "created": fake.created, "deleted": fake.deleted}
    out["status_up"] = str(status_lib.ClusterStatus.UP)

    # 8. yeto's own Verda view uses sky's REAL cluster_name_on_cloud.
    from sky import clouds
    from sky.utils import common_utils

    from yeto.verda_ops import VerdaInstanceGuard, instances_on_cloud, verify_teardown

    display = "vfix-l0-fin-01"
    on_cloud = common_utils.make_cluster_name_on_cloud(display, clouds.Verda.max_cluster_name_length())
    fake = FakeVerda([])
    m.verda = fake
    provision(on_cloud, 1)

    class Api:  # yeto.verda_ops.VerdaApi surface over the same fake
        def list_instances(self):
            return [dict(d, os_volume_id=f"vol-{d['id']}") for d in fake.instances.values()]

        def get_instance(self, iid):
            d = fake.instances.get(iid)
            return None if d is None else dict(d, os_volume_id=f"vol-{iid}")

        def delete_instance(self, iid, volume_ids=None):
            fake.instance_action(iid, "delete")

        def list_volumes(self):
            return []

        def list_trash(self):
            return []

    api = Api()
    guard = VerdaInstanceGuard(api, [display], resolve_on_cloud=lambda c: None)
    by_display = guard.record(display, display)
    by_on_cloud = guard.record(display, on_cloud)
    m.terminate_instances(on_cloud, {})
    out["real_naming"] = {
        "on_cloud": on_cloud,
        "hostnames": sorted(d["hostname"] for d in fake.instances.values()),
        "display_name_matches": by_display,
        "on_cloud_matches": by_on_cloud,
        "verify_after_down": verify_teardown(api, by_on_cloud, on_cloud, sleep_fn=lambda s: None),
        "verify_without_ids": verify_teardown(api, [], on_cloud, sleep_fn=lambda s: None),
        "live_after_down": [i["id"] for i in instances_on_cloud(api, on_cloud)],
    }
    return out


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1]), default=str))

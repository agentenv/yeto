"""Runtime fix for SkyPilot's Verda provisioner (sky/provision/verda/instance.py).

What SkyPilot 0.13.0 gets wrong, and what this replaces (see
openspec/changes/fix-verda-provider/design.md, D2):

* ``query_instances`` has the pre-0.13 signature, so the dispatcher's
  ``(cluster_name, cluster_name_on_cloud, provider_config, ...)`` lands one
  slot off and the *display* name is matched against hostnames;
* existing instances are looked up with status ``'ACTIVE'`` (Verda says
  ``running``), so a relaunch never reuses the running node;
* hostnames are matched by substring (``xc929`` also matches ``xc929b``);
* a failed launch is cleaned up with ``terminate_instances(cluster)``,
  which deletes *every* matching instance — including a pre-existing,
  running one;
* the status map lacks ``ordered`` & co. (KeyError on refresh) and the wait
  loop wants exactly ``count`` running nodes;
* an empty listing is taken at face value (the record is dropped while the
  instance runs).

The patch is applied only on SkyPilot versions it was verified against
(``VERIFIED_SKY_VERSIONS``); otherwise nothing is touched and callers must
treat Verda as unpatched (yeto then disables Verda auto-recovery).
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading

TARGET_MODULE = "sky.provision.verda.instance"
PACKAGE_MODULE = "sky.provision.verda"
VERIFIED_SKY_VERSIONS = frozenset({"0.13.0"})
# Functions whose shape the patch depends on; checked before patching.
REQUIRED_ATTRS = (
    "verda", "run_instances", "terminate_instances", "query_instances",
    "get_cluster_info", "_filter_instances", "InstanceStatus",
)
PATCH_MARK = "_yeto_verda_patch"

# Verda status -> sky ClusterStatus name (None = gone). Unknown -> INIT.
STATUS_MAP = {
    "ordered": "INIT",
    "provisioning": "INIT",
    "new": "INIT",
    "validating": "INIT",
    "starting_hibernation": "INIT",
    "hibernating": "STOPPED",
    "restoring": "INIT",
    "error": "INIT",
    "running": "UP",
    "offline": "STOPPED",
    "deleting": None,
    "deleted": None,
    "discontinued": None,
}
GONE = frozenset({"deleted", "discontinued", "deleting"})
PENDING = frozenset({"ordered", "provisioning", "new", "validating", "restoring"})

# In-process: cluster -> ids this process created in a launch that failed.
# The provisioner's teardown that follows deletes only these.
_FAILED_LAUNCH: dict[str, list[str]] = {}
_LOCK = threading.Lock()


def ids_path() -> str:
    """Cross-process record of instance ids per cluster (R2 defence)."""
    return os.path.expanduser(os.environ.get("YETO_VERDA_IDS_PATH", "~/.yeto/verda-instance-ids.json"))


def load_ids() -> dict[str, list[str]]:
    try:
        with open(ids_path(), encoding="utf-8") as f:
            data = json.load(f)
        return {str(k): [str(i) for i in v] for k, v in data.items()} if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_ids(data: dict[str, list[str]]) -> None:
    path = ids_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, path)
    except OSError:
        pass


def remember_ids(cluster: str, ids) -> None:
    with _LOCK:
        data = load_ids()
        cur = data.get(cluster, [])
        for i in ids:
            if i not in cur:
                cur.append(i)
        data[cluster] = cur
        _save_ids(data)


def forget_ids(cluster: str) -> None:
    with _LOCK:
        data = load_ids()
        if data.pop(cluster, None) is not None:
            _save_ids(data)


def hostname_matches(cluster: str, hostname: str) -> bool:
    """Exact node names only: '<cluster>-head' or '<cluster>-worker[-N]'."""
    if not hostname:
        return False
    return hostname == f"{cluster}-head" or re.fullmatch(re.escape(cluster) + r"-worker(-\d+)?", hostname) is not None


def _sky_version() -> str | None:
    sky = sys.modules.get("sky")
    return getattr(sky, "__version__", None)


def check_version(module, version: str | None = None) -> tuple[bool, str]:
    version = version if version is not None else _sky_version()
    if version not in VERIFIED_SKY_VERSIONS:
        return False, f"sky {version} not in verified {sorted(VERIFIED_SKY_VERSIONS)}"
    missing = [a for a in REQUIRED_ATTRS if not hasattr(module, a)]
    if missing:
        return False, f"sky verda module lacks {missing}"
    return True, ""


def is_patched(module=None) -> bool:
    module = module if module is not None else sys.modules.get(TARGET_MODULE)
    return bool(module is not None and getattr(module, PATCH_MARK, False))


def verified_for_current_sky() -> tuple[bool, str]:
    """Would the patch apply to the sky importable here? (no import side
    effects beyond importing sky's version string)."""
    try:
        import sky  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return False, f"sky not importable ({exc})"
    version = _sky_version()
    if version not in VERIFIED_SKY_VERSIONS:
        return False, f"sky {version} not in verified {sorted(VERIFIED_SKY_VERSIONS)}"
    return True, ""


def apply(module, version: str | None = None) -> tuple[bool, str]:
    """Patch `module` (sky.provision.verda.instance) in place."""
    if is_patched(module):
        return True, "already applied"
    ok, why = check_version(module, version)
    if not ok:
        print(f"[yeto] NOT patching sky Verda provisioner: {why}", file=sys.stderr)
        return False, why
    fns = _build(module)
    for name, fn in fns.items():
        setattr(module, name, fn)
    pkg = sys.modules.get(PACKAGE_MODULE)
    if pkg is not None:
        for name in ("run_instances", "terminate_instances", "query_instances", "get_cluster_info"):
            if hasattr(pkg, name):
                setattr(pkg, name, fns[name])
    setattr(module, PATCH_MARK, True)
    return True, ""


def _build(m):
    """The replacement functions, closed over the target module `m` (its
    client, status enum, logger and sky helpers are looked up on `m` at
    call time, so tests can drive them with a fake module)."""

    def _status(inst) -> str:
        return str(getattr(inst, "status", "") or "").lower()

    def _filter_instances(cluster_name_on_cloud, status_filters=None):
        wanted = None if status_filters is None else {str(s).lower() for s in status_filters}
        out = {}
        for inst in m.verda.instances_get():
            if not hostname_matches(cluster_name_on_cloud, inst.hostname):
                continue
            st = _status(inst)
            if wanted is None:
                if st in GONE:
                    continue
            elif st not in wanted:
                continue
            out[inst.instance_id] = inst
        return out

    def _delete(ids):
        for iid in ids:
            try:
                m.verda.instance_action(instance_id=iid, action="delete")
            except Exception as exc:  # noqa: BLE001 - report, keep going
                m.logger.warning(f"[yeto] delete of Verda instance {iid} failed: {exc}")

    orig_run = m.run_instances

    def run_instances(region, cluster_name, cluster_name_on_cloud, config):
        name = cluster_name_on_cloud
        before = set(_filter_instances(name))
        with _LOCK:
            _FAILED_LAUNCH.pop(name, None)
        try:
            record = orig_run(region, cluster_name, name, config)
        except BaseException:
            after = set(_filter_instances(name))
            created = sorted(after - before)
            with _LOCK:
                _FAILED_LAUNCH[name] = created
            if created:
                m.logger.warning(f"[yeto] Verda launch of {name} failed; deleting only new instance(s) {created}")
                _delete(created)
            raise
        remember_ids(name, [i for i in _filter_instances(name)])
        return record

    def terminate_instances(cluster_name_on_cloud, provider_config=None, worker_only=False):
        name = cluster_name_on_cloud
        with _LOCK:
            failed = _FAILED_LAUNCH.pop(name, None)
        if failed is not None:
            # Teardown right after a failed launch: only what that launch
            # created (already deleted above) may go; pre-existing nodes stay.
            m.logger.info(f"[yeto] {name}: failed-launch teardown limited to {failed or 'nothing'}")
            return
        instances = _filter_instances(name)
        targets = [
            iid for iid, inst in instances.items()
            if not (worker_only and inst.hostname.endswith("-head"))
        ]
        if not targets:
            m.logger.info(f"No instances found for cluster {name}")
            return
        _delete(targets)
        for _ in range(getattr(m, "MAX_POLLS_FOR_UP_OR_TERMINATE", 1)):
            remaining = _filter_instances(name)
            if not any(i in remaining for i in targets):
                break
            m.time.sleep(getattr(m, "POLL_INTERVAL", 5))
        if not worker_only:
            forget_ids(name)

    def _to_cluster_status(st):
        key = STATUS_MAP.get(st, "INIT") if st in STATUS_MAP else "INIT"
        return None if key is None else getattr(m.status_lib.ClusterStatus, key)

    def query_instances(cluster_name, cluster_name_on_cloud, provider_config=None,
                        non_terminated_only=True, retry_if_missing=False):
        del cluster_name, provider_config, retry_if_missing
        name = cluster_name_on_cloud
        found = {}
        for inst in m.verda.instances_get():
            if hostname_matches(name, inst.hostname):
                found[inst.instance_id] = _status(inst)
        if not any(st not in GONE for st in found.values()):
            # R2: an empty/terminated-only listing while ids are on record —
            # ask for each id before letting sky drop the cluster.
            for iid in load_ids().get(name, []):
                if iid in found and found[iid] not in GONE:
                    continue
                try:
                    inst = m.verda.instance_get(iid)
                except Exception:  # noqa: BLE001 - 404 == gone
                    continue
                found[iid] = _status(inst)
        out = {}
        for iid, st in found.items():
            cs = _to_cluster_status(st)
            if non_terminated_only and cs is None:
                continue
            out[iid] = (cs, None)
        return out

    orig_info = m.get_cluster_info

    def get_cluster_info(region, cluster_name_on_cloud, provider_config=None):
        return orig_info(region, cluster_name_on_cloud, provider_config)

    _COUNT = threading.local()

    def _sky_filter(cluster_name_on_cloud, status_filters=None):
        """What sky's own run_instances body calls: exact hostnames, the
        'ACTIVE' lookup read as 'running', and at most the requested number
        of running nodes (so its `== count` wait behaves as `>=`)."""
        if status_filters is not None:
            status_filters = ["running" if str(s).upper() == "ACTIVE" else s for s in status_filters]
        res = _filter_instances(cluster_name_on_cloud, status_filters)
        count = getattr(_COUNT, "value", None)
        if count and status_filters is not None and [str(s).lower() for s in status_filters] == ["running"] and len(res) > count:
            ordered = sorted(res.items(), key=lambda kv: not kv[1].hostname.endswith("-head"))
            res = dict(ordered[:count])
        return res

    def run_with_count(region, cluster_name, cluster_name_on_cloud, config):
        _COUNT.value = int(getattr(config, "count", 1) or 1)
        try:
            return run_instances(region, cluster_name, cluster_name_on_cloud, config)
        finally:
            _COUNT.value = None

    return {
        "_filter_instances": _sky_filter,
        "run_instances": run_with_count,
        "terminate_instances": terminate_instances,
        "query_instances": query_instances,
        "get_cluster_info": get_cluster_info,
    }

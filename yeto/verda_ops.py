"""Verda-side truth for yeto's launcher: instance ids, recovery guard, teardown proof.

sky's Verda provisioner cannot be trusted to say whether a node exists
(openspec change fix-verda-provider), so yeto asks Verda itself — with
its own HTTP client (yeto.shape.providers._verda_request) — before
relaunching an island and after tearing one down.
"""

from __future__ import annotations

import re
import sys
import time
from typing import Callable

from .shape import providers
from .sky_patches.verda import GONE, hostname_matches

# Verda statuses that mean "the node is (still / about to be) there".
ALIVE = frozenset(
    {"running", "provisioning", "ordered", "new", "validating", "offline",
     "starting_hibernation", "hibernating", "restoring", "error"}
)


class VerdaApi:
    """Tiny authenticated client. `request` is injectable for tests."""

    def __init__(self, request: Callable | None = None):
        self._request = request or providers._verda_request
        self._token: str | None = None

    def _auth(self) -> str:
        if self._token is None:
            creds = providers.verda_credentials()
            if creds is None:
                raise RuntimeError("no Verda credentials")
            resp = self._request(
                "POST", "/oauth2/token",
                body={"grant_type": "client_credentials", "client_id": creds[0], "client_secret": creds[1]},
            )
            token = (resp or {}).get("access_token") if isinstance(resp, dict) else None
            if not token:
                raise RuntimeError("Verda token response carried no access_token")
            self._token = str(token)
        return self._token

    def call(self, method: str, path: str, body: dict | None = None):
        return self._request(method, path, token=self._auth(), body=body)

    def list_instances(self) -> list[dict]:
        return list(self.call("GET", "/instances") or [])

    def get_instance(self, instance_id: str) -> dict | None:
        """The instance, or None when Verda answers 404."""
        try:
            return self.call("GET", f"/instances/{instance_id}")
        except RuntimeError as exc:
            if getattr(exc, "status", None) == 404 or "HTTP 404" in str(exc):
                return None
            raise

    def delete_instance(self, instance_id: str, volume_ids: list[str] | None = None) -> None:
        body = {"id": instance_id, "action": "delete"}
        if volume_ids:
            body.update(volume_ids=list(volume_ids), delete_permanently=True)
        self.call("PUT", "/instances", body=body)

    def list_volumes(self) -> list[dict]:
        return list(self.call("GET", "/volumes") or [])

    def list_trash(self) -> list[dict]:
        return list(self.call("GET", "/volumes/trash") or [])

    def delete_volume_permanently(self, volume_id: str) -> None:
        self.call("DELETE", f"/volumes/{volume_id}", body={"is_permanent": True})


def instances_for_cluster(api: VerdaApi, cluster: str) -> list[dict]:
    """Exact-hostname match; deleted/discontinued instances excluded."""
    return [
        i for i in api.list_instances()
        if hostname_matches(cluster, str(i.get("hostname") or ""))
        and str(i.get("status") or "").lower() not in GONE
    ]


def next_cluster_name(name: str) -> str:
    """l0-fin-03 -> l0-fin-03-r1 -> l0-fin-03-r2 ... (lower case)."""
    m = re.fullmatch(r"(.*)-r(\d+)", name)
    base, k = (m.group(1), int(m.group(2))) if m else (name, 0)
    return f"{base}-r{k + 1}".lower()


class VerdaInstanceGuard:
    """Before relaunching a Verda island, look its recorded ids up at Verda.

    check(name) -> None (not a Verda island), ("alive", why), ("gone",
    new cluster name) or ("unknown", why). Ids are recorded at launch
    (`record`) and saved into the run record by the caller's hook."""

    def __init__(self, api: VerdaApi, verda_clusters, on_ids: Callable | None = None):
        self.api = api
        self.clusters = set(verda_clusters)
        self.ids: dict[str, list[str]] = {}
        self.on_ids = on_ids

    def record(self, cluster: str) -> list[str]:
        """Look the cluster's instances up by hostname and remember their ids."""
        ids = [str(i["id"]) for i in instances_for_cluster(self.api, cluster)]
        if ids:
            self.ids[cluster] = ids
            if self.on_ids is not None:
                try:
                    self.on_ids(cluster, ids)
                except Exception as exc:  # noqa: BLE001 - bookkeeping only
                    print(f"[launcher] recording Verda ids for {cluster} failed: {exc}", file=sys.stderr)
        return ids

    def check(self, name: str):
        if name not in self.clusters:
            return None
        ids = self.ids.get(name)
        if not ids:
            try:
                ids = self.record(name)
            except Exception as exc:  # noqa: BLE001
                return ("unknown", f"Verda instance lookup for {name} failed: {exc}")
        states: dict[str, str] = {}
        for iid in ids:
            try:
                inst = self.api.get_instance(iid)
            except Exception as exc:  # noqa: BLE001
                return ("unknown", f"Verda lookup of instance {iid} failed: {exc}")
            states[iid] = "absent" if inst is None else str(inst.get("status") or "").lower()
        alive = {i: s for i, s in states.items() if s in ALIVE or (s not in GONE and s != "absent")}
        if alive:
            return ("alive", f"Verda still has instance(s) {alive} for {name}; the job failed but the "
                             "node was not lost — inspect it before deleting")
        new = next_cluster_name(name)
        self.clusters.add(new)
        self.ids.pop(name, None)
        print(f"[launcher] {name}: Verda instance(s) {states or '(none on record)'} gone; relaunching as {new}")
        return ("gone", new)


def verify_teardown(api: VerdaApi, cluster: str, ids: list[str] | None = None,
                    *, attempts: int = 6, sleep_fn=time.sleep, purge_trash: bool = True):
    """After a down: prove at Verda that the cluster's instances and their
    OS volumes are gone for good. Returns (ok, remaining descriptions).

    Instances are looked up by id (plus exact hostname, for ids we never
    saw); a still-live one is deleted again. Volumes of those instances
    that sit in the trash are deleted permanently."""
    ids = list(ids or [])
    volume_ids: set[str] = set()
    remaining: list[str] = []
    for attempt in range(attempts):
        remaining = []
        for inst in instances_for_cluster(api, cluster):
            if str(inst["id"]) not in ids:
                ids.append(str(inst["id"]))
        for iid in ids:
            inst = api.get_instance(iid)
            if inst is None:
                continue
            volume_ids.update(str(v) for v in (inst.get("volume_ids") or []))
            if inst.get("os_volume_id"):
                volume_ids.add(str(inst["os_volume_id"]))
            st = str(inst.get("status") or "").lower()
            if st in {"deleted", "discontinued"}:
                continue
            remaining.append(f"instance {iid} ({inst.get('hostname')}) status={st}")
            if st != "deleting":
                try:
                    api.delete_instance(iid, sorted(volume_ids) or None)
                except Exception as exc:  # noqa: BLE001
                    remaining[-1] += f" (delete failed: {exc})"
        if not remaining:
            break
        sleep_fn(min(30, 5 * (attempt + 1)))
    if remaining:
        return False, remaining
    # Volumes: none may be live, none may linger in the trash.
    vol_left: list[str] = []
    for attempt in range(attempts):
        vol_left = []
        live = {str(v.get("id")): v for v in api.list_volumes()}
        trash = {str(v.get("id")): v for v in api.list_trash()}
        for vid in sorted(volume_ids):
            if vid in live and str(live[vid].get("status") or "").lower() not in {"deleted"}:
                vol_left.append(f"volume {vid} still attached/live ({live[vid].get('status')})")
                continue
            t = trash.get(vid)
            if t is not None and not t.get("is_permanently_deleted"):
                if purge_trash:
                    try:
                        api.delete_volume_permanently(vid)
                    except Exception as exc:  # noqa: BLE001
                        vol_left.append(f"volume {vid} in trash (permanent delete failed: {exc})")
                        continue
                vol_left.append(f"volume {vid} in trash (permanent delete requested)")
        if not vol_left:
            return True, []
        sleep_fn(min(30, 5 * (attempt + 1)))
    return False, vol_left

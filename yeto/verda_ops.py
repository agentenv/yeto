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
    """Tiny authenticated client. `request` and `clock` are injectable for
    tests. The token is refreshed before it expires and once more on a 401."""

    REFRESH_MARGIN_S = 60.0

    def __init__(self, request: Callable | None = None, clock: Callable[[], float] = time.monotonic):
        self._request = request or providers._verda_request
        self._clock = clock
        self._token: str | None = None
        self._expires_at = 0.0

    def _auth(self, force: bool = False) -> str:
        if force or self._token is None or self._clock() >= self._expires_at - self.REFRESH_MARGIN_S:
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
            try:
                ttl = float(resp.get("expires_in") or 3600)
            except (TypeError, ValueError):
                ttl = 3600.0
            self._token = str(token)
            self._expires_at = self._clock() + ttl
        return self._token

    def call(self, method: str, path: str, body: dict | None = None):
        try:
            return self._request(method, path, token=self._auth(), body=body)
        except RuntimeError as exc:
            if getattr(exc, "status", None) != 401:
                raise
            return self._request(method, path, token=self._auth(force=True), body=body)

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


def instances_on_cloud(api: VerdaApi, name_on_cloud: str) -> list[dict]:
    """Instances whose hostname is exactly `<name_on_cloud>-head|-worker[-N]`.

    `name_on_cloud` is sky's cluster_name_on_cloud (lower-cased display
    name + '-<user hash>', see sky.utils.common_utils.make_cluster_name_on_cloud),
    never the display name. Deleted/discontinued instances excluded."""
    return [
        i for i in api.list_instances()
        if hostname_matches(name_on_cloud, str(i.get("hostname") or ""))
        and str(i.get("status") or "").lower() not in GONE
    ]


def sky_name_on_cloud(cluster: str) -> str | None:
    """sky's cluster_name_on_cloud for a display name: from sky's record
    (the handle), else None. Never guessed from the display name."""
    try:
        from sky import global_user_state

        rec = global_user_state.get_cluster_from_name(cluster)
        handle = (rec or {}).get("handle")
        return getattr(handle, "cluster_name_on_cloud", None)
    except Exception:  # noqa: BLE001 - sky absent / remote API server
        return None


def next_cluster_name(name: str) -> str:
    """l0-fin-03 -> l0-fin-03-r1 -> l0-fin-03-r2 ... (lower case)."""
    m = re.fullmatch(r"(.*)-r(\d+)", name)
    base, k = (m.group(1), int(m.group(2))) if m else (name, 0)
    return f"{base}-r{k + 1}".lower()


class VerdaInstanceGuard:
    """Before relaunching a Verda island, look its recorded ids up at Verda.

    Ids are recorded right after each (re)launch from the launch handle's
    cluster_name_on_cloud (`record(cluster, name_on_cloud)`); the caller's
    hook saves them in the run record. check(name) -> None (not a Verda
    island), ("alive", why), ("gone", new cluster name) or ("unknown", why).
    Without recorded ids the answer is "unknown": a blind relaunch could
    delete a node we simply failed to see."""

    def __init__(self, api: VerdaApi, verda_clusters, on_ids: Callable | None = None,
                 resolve_on_cloud: Callable[[str], str | None] = sky_name_on_cloud):
        self.api = api
        self.clusters = set(verda_clusters)
        self.ids: dict[str, list[str]] = {}
        self.on_cloud: dict[str, str] = {}
        self.on_ids = on_ids
        self.resolve_on_cloud = resolve_on_cloud

    def record(self, cluster: str, name_on_cloud: str | None = None) -> list[str]:
        """Remember the instance ids behind `cluster` (exact hostname match
        on sky's cluster_name_on_cloud)."""
        name_on_cloud = name_on_cloud or self.resolve_on_cloud(cluster)
        if not name_on_cloud:
            print(f"[launcher] WARNING: no cluster_name_on_cloud for {cluster}; Verda ids not recorded",
                  file=sys.stderr)
            return []
        self.on_cloud[cluster] = name_on_cloud
        ids = [str(i["id"]) for i in instances_on_cloud(self.api, name_on_cloud)]
        if not ids:
            print(f"[launcher] WARNING: Verda lists no instance named {name_on_cloud}-*; ids not recorded",
                  file=sys.stderr)
            return []
        self.ids[cluster] = ids
        if self.on_ids is not None:
            try:
                self.on_ids(cluster, ids)
            except Exception as exc:  # noqa: BLE001 - bookkeeping only
                print(f"[launcher] recording Verda ids for {cluster} failed: {exc}", file=sys.stderr)
        return ids

    def after_relaunch(self, cluster: str) -> list[str]:
        """Ids of the node a relaunch just created (sky's record now holds
        the new cluster_name_on_cloud)."""
        self.ids.pop(cluster, None)
        return self.record(cluster)

    def check(self, name: str):
        if name not in self.clusters:
            return None
        ids = self.ids.get(name)
        if not ids:
            return ("unknown", f"no Verda instance ids on record for {name}")
        states: dict[str, str] = {}
        for iid in ids:
            try:
                inst = self.api.get_instance(iid)
            except Exception as exc:  # noqa: BLE001
                return ("unknown", f"Verda lookup of instance {iid} failed: {exc}")
            states[iid] = "absent" if inst is None else str(inst.get("status") or "").lower()
        alive = {i: s for i, s in states.items() if s not in GONE and s != "absent"}
        if alive:
            return ("alive", f"Verda still has instance(s) {alive} for {name}; the job failed but the "
                             "node was not lost — inspect it before deleting")
        new = next_cluster_name(name)
        self.clusters.add(new)
        print(f"[launcher] {name}: Verda instance(s) {states} gone; relaunching as {new}")
        return ("gone", new)


def verify_teardown(api: VerdaApi, ids: list[str] | None, name_on_cloud: str | None = None,
                    *, attempts: int = 6, sleep_fn=time.sleep, purge_trash: bool = True):
    """After a down: prove at Verda that THIS run's instances are gone and
    their OS volumes deleted for good. Returns (ok, remaining).

    Scope: only the recorded instance ids, and only their OS volume
    (`os_volume_id`; extra attached volumes are not created by yeto and are
    left alone). An empty id list is never success. Instances found by
    hostname (`name_on_cloud`) that are not on record are reported as
    suspected leftovers — not deleted — and also make the result not ok."""
    ids = [str(i) for i in (ids or [])]
    if not ids:
        return False, ["no Verda instance ids on record; cannot prove the teardown"]
    volume_ids: set[str] = set()
    remaining: list[str] = []
    for attempt in range(attempts):
        remaining = []
        for iid in ids:
            inst = api.get_instance(iid)
            if inst is None:
                continue
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
    suspects = []
    if name_on_cloud:
        suspects = [
            f"suspected leftover (not on record, not deleted): instance {i.get('id')} ({i.get('hostname')}) "
            f"status={i.get('status')}"
            for i in instances_on_cloud(api, name_on_cloud) if str(i.get("id")) not in ids
        ]
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
            break
        sleep_fn(min(30, 5 * (attempt + 1)))
    left = vol_left + suspects
    return (not left), left

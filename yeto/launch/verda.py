"""Verda islands: in-VM docker preparation, launch candidates and cluster renames."""

from __future__ import annotations

import os
import sys

from . import REPO_ROOT


# Verda islands of this process's run: cluster name -> VerdaInstanceGuard
# (holds their instance ids). terminate_and_verify proves their teardown
# at Verda by id instead of trusting sky.
VERDA_ISLANDS: dict = {}


def prepare_verda_islands(names: list[str], on_instance_ids=None) -> dict:
    """Per-run Verda setup on this machine: the full local sky catalog, the
    patch status (unpatched -> no auto-recovery for these islands, D4) and
    the instance-id guard used by recovery and teardown."""
    from ..shape.providers import write_verda_sky_catalog
    from ..sky_patches import verda as verda_patch
    from ..verda_ops import VerdaApi, VerdaInstanceGuard

    try:
        print(f"[launcher] verda: full sky catalog written to {write_verda_sky_catalog()}")
    except Exception as e:  # noqa: BLE001 - sky's own catalog still works, just thinner
        print(f"[launcher] WARNING: could not write the Verda sky catalog: {e}", file=sys.stderr)
    ok, why = verda_patch.verified_for_current_sky()
    pth_written = None
    if ok and remote_sky_api_server():
        # Provisioning runs in a server we cannot patch from here.
        ok, why = False, f"sky API server is remote ({remote_sky_api_server()})"
    if ok:
        from ..sky_patches import ensure_local_pth

        try:
            pth, fresh = ensure_local_pth(str(REPO_ROOT))
            if fresh:
                # A server already running keeps the unpatched code: treat
                # the patch as not in effect for this run (2.5).
                pth_written = pth
                ok, why = False, (
                    f"patch hook {pth} was only now installed; a sky API server started "
                    "before it runs unpatched (install it persistently with "
                    "`python -m yeto.sky_patches install`, then `sky api stop`)"
                )
        except OSError as e:
            ok, why = False, f"cannot install the patch hook for sky's API server ({e})"
    no_recover = set() if ok else set(names)
    if not ok:
        print(
            f"[launcher] WARNING: sky's Verda provisioner is NOT patched ({why}); "
            f"auto-recovery is disabled for {', '.join(names)} (recover_timeout=0) — "
            "a same-name relaunch there can delete the running node",
            file=sys.stderr,
        )
    guard = VerdaInstanceGuard(VerdaApi(), names, on_ids=on_instance_ids)
    for n in names:
        VERDA_ISLANDS[n] = guard
    return {"guard": guard, "no_recover": no_recover, "pth_written": pth_written}


def remote_sky_api_server() -> str | None:
    """The sky API server endpoint when it is not on this machine, else None."""
    endpoint = os.environ.get("SKYPILOT_API_SERVER_ENDPOINT")
    try:
        from sky.server import common as server_common

        endpoint = server_common.get_server_url()
        if server_common.is_api_server_local(endpoint):
            return None
        return endpoint
    except Exception:  # noqa: BLE001 - sky absent or API drift: judge the env var
        if not endpoint:
            return None
        from urllib.parse import urlparse

        host = urlparse(endpoint).hostname or ""
        return None if host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"} else endpoint


def verda_launch_candidates(spec, args, availability, demoted) -> list[dict]:
    from ..shape.providers import VerdaSignals, verda_any_of, verda_candidates

    types = VerdaSignals(cache=None)._fetch_types()
    per_node = spec.gpus_per_node
    cands = verda_candidates(
        spec.gpu, per_node, types, availability,
        regions=[spec.region] if spec.region else None,
        use_spot=bool(getattr(args, "spot", False)), demoted=demoted,
    )
    return verda_any_of(cands, spec.gpu, per_node)


def _verda_copy_override(sky, cand: dict) -> dict:
    """Resources.copy() keeps the base's cloud/region; an `infra=` override
    on top of those raises "Cannot specify both infra and cloud, region, or
    zone". Translate the candidate's infra into explicit cloud/region/zone."""
    out = dict(cand)
    infra = out.pop("infra", None)
    if infra:
        cloud, _, rest = infra.partition("/")
        region, _, zone = rest.partition("/")
        out["cloud"] = sky.CLOUD_REGISTRY.from_str(cloud)
        out["region"] = region or None
        out["zone"] = zone or None
    return out


def launch_verda_island(sky, task, name: str, spec, args, *, sleep=None):
    """Launch one Verda island over live-stock candidates (D5)."""
    from ..shape.providers import VerdaSignals, launch_with_verda_candidates

    sig = VerdaSignals(cache=None, use_spot=bool(getattr(args, "spot", False)))
    base = next(iter(task.resources))

    def launch(cands):
        task.set_resources([base.copy(**_verda_copy_override(sky, c)) for c in cands])
        print(f"[launcher] {name}: Verda candidates {[c['instance_type'] + '@' + c['infra'] for c in cands]}")
        return sky.stream_and_get(sky.launch(task, cluster_name=name, retry_until_up=False))

    return launch_with_verda_candidates(
        launch,
        sig._fetch_availability,
        lambda avail, demoted: verda_launch_candidates(spec, args, avail, demoted),
        sleep=sleep,
    )


def _rename_hook(clusters: list, on_clusters, fixed: list):
    """FleetController on_rename: keep the teardown list, the Verda id guard
    and the run registry on the island's new cluster name."""

    def hook(old: str, new: str) -> None:
        if old in clusters:
            clusters[clusters.index(old)] = new
        else:
            clusters.append(new)
        if old in VERDA_ISLANDS:
            VERDA_ISLANDS[new] = VERDA_ISLANDS[old]
        if on_clusters is not None:
            on_clusters(fixed + [c for c in clusters if c not in fixed])

    return hook

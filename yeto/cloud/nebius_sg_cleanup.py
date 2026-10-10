"""Delete a run's leaked SkyPilot security groups on Nebius (S19 real-spot A2).

sky names each cluster's group ``sky-sg-<cluster_name_on_cloud>`` and deletes it on
``sky down``. Groups leak when an instance is removed some other way (yeto's
delete-by-name fallback, a head that died first) or when sky's delete gives up;
the network quota is 128 groups and S19 hit it ("max-security-groups-count 129 of
128") with 125 leaked groups and 0 instances.

:func:`delete_run_security_groups` deletes only groups whose name starts with
``sky-sg-<run>-`` and that no instance network interface uses, through sky's own
``delete_security_group`` (drains rules first; a group with rules cannot be
deleted). Best effort: never raises.
"""

from __future__ import annotations

from typing import Any, Callable

SG_PREFIX = "sky-sg-"


def run_sg_prefix(run: str) -> str:
    return f"{SG_PREFIX}{run}-"


def orphans(groups: list[tuple[str, str]], used_ids: set[str], run: str) -> list[tuple[str, str]]:
    """(id, name) of this run's groups that no instance uses."""
    prefix = run_sg_prefix(run)
    return [(gid, name) for gid, name in groups if name.startswith(prefix) and gid not in used_ids]


def _list_groups(utils: Any, project_id: str) -> list[tuple[str, str]]:
    nebius = utils.nebius
    service = nebius.vpc().SecurityGroupServiceClient(nebius.sdk())
    out, token = [], ""
    while True:
        res = nebius.sync_call(service.list(nebius.vpc().ListSecurityGroupsRequest(
            parent_id=project_id, page_size=100, page_token=token)))
        out.extend((g.metadata.id, g.metadata.name) for g in res.items)
        token = res.next_page_token
        if not token:
            return out


def delete_run_security_groups(run: str, region: str = "eu-north1", *, utils: Any = None,
                               log: Callable[[str], Any] = print) -> list[str]:
    """Delete run ``run``'s unused sky security groups; returns the ids deleted."""
    try:
        if utils is None:
            from sky.provision.nebius import utils as utils  # noqa: PLW0127
        project_id = utils.get_project_by_region(region)
        used = {sg for info in utils.list_instances(project_id).values()
                for sg in info.get("security_group_ids") or []}
        todo = orphans(_list_groups(utils, project_id), used, run)
    except Exception as exc:  # noqa: BLE001
        log(f"[yeto] Nebius security groups of {run}: not checked ({type(exc).__name__}: {exc})")
        return []
    done = []
    for gid, name in todo:
        try:
            utils.delete_security_group(gid)  # sky logs and returns when it gives up
            if utils.get_security_group_by_name(project_id, name) is not None:
                raise RuntimeError("still present after delete")
            done.append(gid)
            log(f"[yeto] Nebius security group {name}: deleted")
        except Exception as exc:  # noqa: BLE001
            log(f"[yeto] Nebius security group {name}: NOT deleted ({exc})")
    return done

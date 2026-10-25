"""Let two sky clusters on Nebius reach each other (S27 G-DI two-cluster island).

sky gives each Nebius cluster its own security group ``sky-sg-<cluster_name_on_cloud>``.
Its only inbound rules are "any protocol from this same group" and TCP 22 / 10022 from
anywhere (``sky.provision.nebius.utils.ensure_default_sg_rules``). The two clusters of
one island (trainer cluster + engine cluster) therefore cannot reach each other's Ray,
NCCL or SGLang ports.

:func:`link` adds to each group one inbound rule "any protocol from the other group".
The rule lives inside the cluster's own group, and sky drains a group's rules before it
deletes the group on ``sky down``, so the rule goes away with the cluster. :func:`unlink`
removes the two rules first, so that sky never deletes a group that a rule of the other
group still names. Nothing is opened to the internet.
"""

from __future__ import annotations

from typing import Any, Callable

SG_PREFIX = "sky-sg-"
LINK_RULE_PREFIX = "yeto-island-peer"


def group_name(cluster_name_on_cloud: str) -> str:
    return f"{SG_PREFIX}{cluster_name_on_cloud}"


def _utils(utils: Any) -> Any:
    if utils is None:
        from sky.provision.nebius import utils as sky_utils

        return sky_utils
    return utils


def _peer_spec(utils: Any, source_group: str) -> Any:
    vpc = utils.nebius.vpc()
    return vpc.SecurityRuleSpec(
        access=vpc.RuleAccessAction.ALLOW,
        protocol=vpc.RuleProtocol.ANY,
        ingress=vpc.RuleIngress(source_security_group_id=source_group),
    )


def _group_ids(utils: Any, region: str, names: tuple[str, str]) -> tuple[str, str]:
    project = utils.get_project_by_region(region)
    ids = []
    for name in names:
        gid = utils.get_security_group_by_name(project, group_name(name))
        if gid is None:
            raise RuntimeError(f"Nebius security group {group_name(name)} not found in {region}")
        ids.append(gid)
    return ids[0], ids[1]


def link(region: str, cluster_a: str, cluster_b: str, *, utils: Any = None,
         log: Callable[[str], Any] = print) -> dict[str, str]:
    """Admit each cluster's group into the other's (idempotent). ``cluster_a`` and
    ``cluster_b`` are sky's ``cluster_name_on_cloud``. Raises when a group is missing
    or a rule cannot be created: the island cannot work without the rule."""
    utils = _utils(utils)
    a, b = _group_ids(utils, region, (cluster_a, cluster_b))
    for dst, src in ((a, b), (b, a)):
        spec = _peer_spec(utils, src)
        have = {utils._rule_signature(r.spec) for r in utils.list_security_rules(dst)}
        if utils._rule_signature(spec) in have:
            log(f"[launcher] Nebius security group {dst} already admits {src}")
            continue
        utils._create_security_rule(dst, spec, LINK_RULE_PREFIX)
        log(f"[launcher] Nebius security group {dst} now admits {src} (any protocol)")
    return {cluster_a: a, cluster_b: b}


def unlink(region: str, cluster_a: str, cluster_b: str, *, utils: Any = None,
           log: Callable[[str], Any] = print) -> int:
    """Delete the rules :func:`link` made (best effort: never raises). Returns the
    number of rules deleted."""
    try:
        utils = _utils(utils)
        a, b = _group_ids(utils, region, (cluster_a, cluster_b))
    except Exception as exc:  # noqa: BLE001 - a missing group has no rule left to delete
        log(f"[launcher] Nebius peer rules of {cluster_a} / {cluster_b}: not checked ({exc})")
        return 0
    deleted = 0
    for dst, src in ((a, b), (b, a)):
        try:
            for rule in utils.list_security_rules(dst):
                ingress = getattr(rule.spec, "ingress", None)
                if ingress is not None and (ingress.source_security_group_id or "") == src:
                    utils._delete_security_rule(rule.metadata.id)
                    deleted += 1
        except Exception as exc:  # noqa: BLE001
            log(f"[launcher] Nebius peer rule in {dst}: not deleted ({exc})")
    return deleted

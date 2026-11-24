"""The control plane's side of an assignment: what a pool writes for one machine.

A pool member (``yeto.launch.pool.island_pool.Member``: ticket, shard, the shard
it continues, its data slot) on a node with a profile becomes
``assignments/<ticket>.json`` (:func:`assign_member`): the spec rendered as the
pool renders it for every route, the code hash (published once per tree), the
model plan, the profile, the cards. The node runs ``python -m yeto.node run
--assignment <ticket>`` with the returned token; nothing else reaches it.

A machine opened by hand joins the same way (:func:`assign_manual`): the pool
config and a member made for it (a ``manual-...`` ticket, the next shard name),
so the pool's fence (``pool-fence:<ticket>``) and the handoff of a replacement
(``continues_shard``) act on it as on an ordered one. The operator gets the
command to run there (:func:`node_command`).
"""

from __future__ import annotations

import shlex
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from yeto.launch.node_inventory import NodeProfile

from .profile import profile_doc
from .records import Assignment, CodeRef, ModelPlan, assign

MANUAL_PREFIX = "manual"


def model_plan(cfg: Mapping[str, Any], profile: NodeProfile) -> ModelPlan:
    """The pool config's ``model_pull`` (``src_key``, ``dest``, ``include``, ``max_mb_s``)
    as a plan, by the profile's model source (no ``model_pull``: nothing to do)."""
    pull = cfg.get("model_pull") or {}
    source = profile.model_source or ("store" if pull else "none")
    if source == "store" and not pull:
        return ModelPlan(source="none")
    return ModelPlan(source=source, key=pull.get("src_key"), dest=pull.get("dest"), include=pull.get("include"),
                     max_mb_s=int(pull.get("max_mb_s", 2500)), timeout_s=float(pull.get("timeout_s", 3600)),
                     repo=pull.get("repo"), revision=pull.get("revision"))


def assign_member(objects: Any, cfg: Mapping[str, Any], member: Any, shards: Sequence[str], profile: NodeProfile,
                  code: CodeRef, *, source: str | None = None, spec: Mapping[str, Any] | None = None,
                  render: Callable[..., Mapping[str, Any]] | None = None, store_uri: str | None = None,
                  token: str | None = None, **extra: Any) -> tuple[Assignment, str]:
    """Write the assignment of pool member ``member`` and return it with its token.
    ``spec``: the rendered spec (default: the pool's ``render_spec`` of the member);
    every generation keeps its own node records (the next generation is past all of them)."""
    if spec is None:
        if render is None:
            from yeto.launch.nc.cli import render_spec as render  # noqa: PLC0415
        spec = render(dict(cfg), member, list(shards))
    spec = {"start_on_latest": True, **dict(spec)}
    assignment, token = assign(objects, ticket=member.ticket, shard=member.shard, spec=spec, code=code, token=token,
                               run=str(cfg.get("run_prefix") or "").strip("/") or None,
                               pool=cfg.get("name") or cfg.get("run"), source=source,
                               continues_shard=getattr(member, "replaces_shard", None),
                               data_slot=getattr(member, "data_slot", None), model=model_plan(cfg, profile),
                               profile=profile_doc(profile), store_uri=store_uri, **extra)
    return assignment, token


MANUAL_TOKEN_TTL_S = 24 * 3600.0  # a hand-opened machine reads its assignment within a day


def manual_ticket(run_id: str, name: str, clock: Callable[[], float] = time.time) -> str:
    return f"{MANUAL_PREFIX}-{run_id}-{name}-{int(clock())}"


def assign_manual(objects: Any, cfg: Mapping[str, Any], *, shard: str, shards: Sequence[str], profile: NodeProfile,
                  code: CodeRef, run_id: str, continues_shard: str | None = None, data_slot: str | None = None,
                  member_factory: Callable[..., Any] | None = None, token_ttl_s: float | None = MANUAL_TOKEN_TTL_S,
                  clock: Callable[[], float] = time.time, **kw: Any) -> tuple[Any, Assignment, str]:
    """A member for a machine opened by hand and its assignment: ``(member, assignment,
    token)``. The member is the pool's own record type, so the pool fences and hands
    off its shard by the same rules. The token is refused after ``token_ttl_s`` (None:
    never): a token handed to an operator does not stay valid forever. The token binds
    the machine to this ticket; it is not an authentication (``yeto.node.records``)."""
    if member_factory is None:
        from yeto.launch.pool.island_pool import Member as member_factory  # noqa: PLC0415,N813
    member = member_factory(account=-1, ticket=manual_ticket(run_id, profile.name), shard=shard,
                            replaces_shard=continues_shard, data_slot=data_slot)
    if token_ttl_s is not None:
        kw.setdefault("expires_at", float(clock()) + float(token_ttl_s))
    assignment, token = assign_member(objects, cfg, member, shards, profile, code, source="manual", **kw)
    return member, assignment, token


def node_command(assignment: Assignment, store: str, *, env_file: str | None = None,
                 python: str = "python3") -> str:
    """The command an operator (or a transport) runs on the node; the token goes in
    ``YETO_NODE_TOKEN`` or a ``--token-file``, never in this line."""
    argv = [python, "-m", "yeto.node", "run", "--assignment", assignment.ticket, "--store", store]
    if env_file:
        argv += ["--env-file", env_file]
    return shlex.join(argv)

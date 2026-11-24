"""The node runtime on an SSH + docker host of the pool's lease route (the sky route).

The SSH + docker pool (``yeto.launch.nc.cli.NcSshOps``) on ``provider.kind: sky`` used
the National Compute node layout and its MI355X container flags for every cloud VM,
so a CUDA VM could not start a sampler. On the node runtime it starts, over the
lease's ssh alias:

1. the node's profile from the offer (``H100:8`` -> family cuda, 8 cards; the pool
   config's ``node_profile`` section overrides any field) and its directories under
   the ssh user's home (``~/yeto/{work,models,cache}``, no sudo needed for them);
2. the image: the pool's ``sampler_image``, else the engine image catalog's choice
   for (profile, model, engine) - refused here, before ``docker run``, when none fits;
3. the assignment in the run store, the code tree, the store env file, the token
   and the profile (as the runtime in the container sees it) on the node, the
   prompts file checked by size and sha256;
4. ``sudo docker run -d`` of the sampler image with the family's device flags and
   the node's directories at ``/work``, ``/models``, ``/root/.cache``, running
   ``python -m yeto.node run``: the runtime pulls the model by its plan, writes the
   spec and starts the sampler as its child; ``docker stop`` reaches the sampler
   through it. The pool reads the node's state from the store.
"""

from __future__ import annotations

import dataclasses
import shlex
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from yeto.launch.node_inventory import NodePaths, NodeProfile

from .profile import CONTAINER_CACHE, CONTAINER_MODELS, CONTAINER_WORK, container_mounts, node_dirs
from .route import runtime_command
from .transport import check_copy, size_sha256_command

CONTAINER_PATHS = {"work": CONTAINER_WORK, "models": CONTAINER_MODELS, "cache": CONTAINER_CACHE}


class SshDockerError(RuntimeError):
    pass


def accelerators_of(provider: Any, ticket: str) -> tuple[str, int] | None:
    """``(card, count)`` of the offer the lease was ordered with (``"H100:8"``)."""
    key = getattr(provider, "offer_of", lambda t: None)(ticket)
    offers = list(getattr(provider, "offers", ()) or ())
    chosen = next((o for o in offers if getattr(o, "key", None) == key), offers[0] if offers else None)
    return _accelerators(getattr(chosen, "resources", None) or getattr(provider, "resources", None))


def check_offers(cfg: Mapping[str, Any], provider: Any, spec: Mapping[str, Any]) -> list[tuple[str, str]]:
    """``(offer, image)`` for every offer of the pool; :class:`SshDockerError` before anything
    is ordered when a machine of some offer has no image for the spec (each one named)."""
    offers = list(getattr(provider, "offers", ()) or ()) or [SimpleNamespace(
        key="resources", resources=getattr(provider, "resources", None) or {})]
    chosen, refused = [], []
    for offer in offers:
        try:
            profile = offer_profile(cfg, _accelerators(getattr(offer, "resources", None)), name=str(offer.key))
            chosen.append((str(offer.key), image_for(cfg, profile, spec)[0]))
        except SshDockerError as error:
            refused.append(f"offer {offer.key}: {error}")
    if refused:
        raise SshDockerError("no sampler image for every offer of the pool (refused before ordering): "
                             + "; ".join(refused))
    return chosen


def _accelerators(resources: Mapping[str, Any] | None) -> tuple[str, int] | None:
    accelerators = (resources or {}).get("accelerators")
    if isinstance(accelerators, Mapping):  # sky's {"H100": 8}
        accelerators = next((f"{k}:{v}" for k, v in accelerators.items()), "")
    card, _, count = str(accelerators or "").partition(":")
    return (card, int(float(count or 1))) if card else None


def node_profile(cfg: Mapping[str, Any], provider: Any, lease: Any) -> NodeProfile:
    """The lease's node profile (offer + the pool config's ``node_profile`` section)."""
    return offer_profile(cfg, accelerators_of(provider, lease.ticket), name=lease.ticket, host=lease.ip,
                         user=lease.user, gpus=lease.gpus)


def offer_profile(cfg: Mapping[str, Any], found: tuple[str, int] | None, *, name: str, host: str = "",
                  user: str = "ubuntu", gpus: int | None = None) -> NodeProfile:
    """The node profile of an offer's machine (``(card, count)`` of its accelerators)."""
    from yeto.gpu_spec import canonical_card  # noqa: PLC0415
    from yeto.hw.catalog import CARD_HBM_GIB  # noqa: PLC0415

    section = dict(cfg.get("node_profile") or {})
    card = section.pop("card", None) or (found[0] if found else None)
    if not card:
        raise SshDockerError(f"{name}: no card known (the offer names no accelerators): set the "
                             "pool config's node_profile.card")
    cards = int(section.pop("cards_per_node", None) or (found[1] if found else None) or gpus or 8)
    hbm = section.pop("hbm_gib", None) or next(iter(CARD_HBM_GIB.get(canonical_card(card), ())), 0.0)
    base = f"/home/{user}/yeto" if user != "root" else "/root/yeto"
    paths = dict(section.pop("paths", None) or {})
    paths = NodePaths(work=paths.get("work") or f"{base}/work", models=paths.get("models") or f"{base}/models",
                      cache=paths.get("cache") or f"{base}/cache")
    section.setdefault("container", "docker")
    section.setdefault("model_source", "store" if cfg.get("model_pull") else None)
    try:
        return NodeProfile.for_card(name, card, cards, hbm_gib=float(hbm), host=host, paths=paths, **section)
    except TypeError as error:
        raise SshDockerError(f"pool config node_profile: {error}") from None


def container_dest(dest: str | None) -> str:
    """``model_pull.dest`` as the container sees it: the route's host paths (default
    ``/mnt/nvme/models/Kimi-K3.1-Pro``) by the route's own mounts, so the spec's
    ``/models/...`` path names the pulled model on every route."""
    from yeto.launch.nc import ssh_sampler  # noqa: PLC0415

    host = str(dest or ssh_sampler.MODEL_HOST).rstrip("/")
    if host == CONTAINER_MODELS or host.startswith(CONTAINER_MODELS + "/"):
        return host
    if host in ssh_sampler.MOUNTS:
        return ssh_sampler.MOUNTS[host]
    root = f"{ssh_sampler.NVME}/models"
    if host.startswith(root + "/"):
        return CONTAINER_MODELS + host[len(root):]
    raise SshDockerError(f"model_pull.dest {dest!r}: a path under /models/ (or the route's {root}/)")


def image_for(cfg: Mapping[str, Any], profile: NodeProfile, spec: Mapping[str, Any]) -> tuple[str, str]:
    """``(image, python)``: the pool's sampler image, else the catalog's choice."""
    from yeto.launch import engine_images  # noqa: PLC0415
    from yeto.launch.nc.cli import pool_sampler_image  # noqa: PLC0415

    named = pool_sampler_image(dict(cfg)) or cfg.get("image")
    needs = engine_images.spec_needs(spec)
    model, pref = needs.pop("model"), needs.pop("engine_pref")
    try:
        chosen = engine_images.resolve(profile, model, pref, image=named, **needs)
    except engine_images.ImageIncompatible as error:
        raise SshDockerError(f"lease {profile.name}: {error}") from None
    return chosen.ref.removeprefix("docker:"), chosen.python or "python3"


def start(ops: Any, lease: Any, m: Any, shards: list[str], *, run: Callable[..., Any] = subprocess.run,
          prompts_file: str | None = None) -> str:
    """Start ``m``'s sampler on the lease's node through the node runtime; the container name."""
    from yeto.launch import agentic_setup  # noqa: PLC0415
    from yeto.launch.nc import node_pool, ssh_sampler  # noqa: PLC0415

    cfg, route = ops.cfg, ops.node_route
    prefix = str(cfg["run_prefix"]).strip("/")
    if prefix in frozenset(cfg.get("protected_prefixes", ())):
        raise ValueError(f"run prefix {prefix!r} is protected (a buffer in use)")
    spec = {"start_on_latest": True, **ops._spec(m, shards)}  # noqa: SLF001 - the pool's own render
    if agentic_setup.parse(spec) is not None:
        raise ValueError("agentic.setup is not prepared on the node runtime route: set node_runtime false")
    agentic_env_file = cfg.get("agentic_env_file")
    if agentic_env_file and not spec.get("agentic"):
        raise ValueError("an agentic env file needs a spec with an agentic section")
    alias, profile = node_pool.host_alias(lease.ticket), node_profile(cfg, getattr(ops, "provider", None), lease)
    image, python = image_for(cfg, profile, spec)
    dirs = node_dirs(profile)
    sampler_dir = f"{dirs['work']}/sampler"

    def ssh(command: str, *, input: str | None = None, timeout: float = 600) -> Any:
        out = ssh_sampler.ssh(alias, command, run=run, timeout=timeout, input=input)
        if out.returncode != 0:
            raise SshDockerError(f"{alias}: {command[:60]!r} exit {out.returncode}: {(out.stderr or '')[-300:]}")
        return out

    def put(text: str, remote: str, mode: str) -> None:
        q = shlex.quote(remote)
        ssh(f"umask 077; cat > {q}.tmp && chmod {mode} {q}.tmp && mv -f {q}.tmp {q}", input=text, timeout=120)

    ssh("mkdir -p " + " ".join(shlex.quote(d) for d in (sampler_dir, f"{dirs['work']}/logs", f"{dirs['work']}/stage",
                                                         dirs["models"], dirs["cache"])), timeout=60)
    remote_prompts = ((spec.get("prompts") or {}) if isinstance(spec.get("prompts"), Mapping) else {}).get("path")
    if prompts_file is not None:
        if not str(remote_prompts or "").startswith(CONTAINER_WORK + "/"):
            raise ValueError(f"prompts.path {remote_prompts or '(unset)'!r}: a prompts file needs a container path "
                             "under /work/")
        host = dirs["work"] + str(remote_prompts)[len(CONTAINER_WORK):]
        ssh(f"mkdir -p {shlex.quote(str(Path(host).parent))}", timeout=60)
        ssh_sampler.upload_file(alias, prompts_file, host, mode="644", run=run)
        out = ssh(size_sha256_command(host), timeout=120)
        check_copy(out.stdout or "", Path(prompts_file).expanduser().read_bytes(), node=alias, remote=host,
                   local=prompts_file)
    node_cfg = dict(cfg)
    if cfg.get("model_pull"):
        node_cfg["model_pull"] = {**cfg["model_pull"], "dest": container_dest(cfg["model_pull"].get("dest"))}
    assignment, token = route.assign(m, shards, profile, spec, cfg=node_cfg, image=image, python=python)
    yeto_dir = ssh_sampler.upload_code(alias, run=run, root=f"{dirs['work']}/yeto")
    if Path(yeto_dir).name != assignment.code.sha:
        raise SshDockerError(f"{alias}: the code tree {yeto_dir} is not the assignment's {assignment.code.sha}")
    env_file = f"{sampler_dir}/.store.env"
    ssh_sampler.upload_file(alias, cfg["r2_env_file"], env_file, mode="600", run=run)
    inside = f"{CONTAINER_WORK}/sampler"
    put(token, f"{sampler_dir}/{m.shard}.token", "600")
    put(route.profile_file_text(profile, container="process", paths=CONTAINER_PATHS),
        f"{sampler_dir}/{m.shard}.profile.json", "644")
    extra_env_files = []
    if agentic_env_file:  # HARBOR_*, DOCKER_HOST, ...: a second --env-file, as on the docker route
        extra_env_files.append(f"{sampler_dir}/{m.shard}.agentic.env")
        ssh_sampler.upload_file(alias, str(Path(str(agentic_env_file)).expanduser()), extra_env_files[-1],
                                mode="600", run=run)
    # the yeto tree in front of the image's own PYTHONPATH (-e PYTHONPATH= would drop it)
    tree = f"{CONTAINER_WORK}/yeto/{assignment.code.sha}"
    command = (f'export PYTHONPATH={shlex.quote(tree)}"${{PYTHONPATH:+:$PYTHONPATH}}"; '
               + runtime_command(python, assignment.ticket, route.store, env_file=f"{inside}/.store.env",
                                 token_file=f"{inside}/{m.shard}.token",
                                 profile_file=f"{inside}/{m.shard}.profile.json", log="-"))
    # the docker route's tuning (release download, start spread, run prefix); the runtime sets
    # the visible cards, the store settings and the per-shard variables itself
    env = {k: v for k, v in ssh_sampler.sampler_env(prefix).items() if k not in ("PYTHONPATH", "HIP_VISIBLE_DEVICES")}
    node = dataclasses.replace(profile, docker_args=tuple(profile.devices(None).args), image=image)
    name = ssh_sampler.container_name(m.shard)
    argv = ssh_sampler.docker_run_argv(
        node, name=name, command=["bash", "-c", command], yeto_dir=None, env_file=env_file, detach=True,
        env={**env, "PYTHONUNBUFFERED": "1", **profile.family_env()}, extra_env_files=extra_env_files,
        mounts=container_mounts(profile), ro_models=False)
    ssh(f"sudo docker rm -f {name} >/dev/null 2>&1; {shlex.join(argv)}", timeout=1800)
    return name

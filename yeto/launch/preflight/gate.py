"""The cluster preflight every launch of an ``ssh:`` trainer island runs first.

``yeto launch`` with an ``ssh:`` entry in ``--gpu`` (and ``yeto run up`` whose
trainer is one, before it orders anything) runs the checks of ``yeto preflight
--no-bench`` on the inventory's trainer nodes: the per-node checks and the TCP
ports between the nodes, no bandwidth benchmark. A FAIL refuses the launch, a
WARN is printed and the launch goes on. ``--skip-preflight`` skips it; a dry run
only prints the command it would run. The report goes to the run directory, as
``yeto preflight`` writes it.
"""

from __future__ import annotations

import shlex
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO, Any

from ..ssh_trainer_island import SshIslandError
from .cluster import (
    FAIL,
    PASS,
    SKIP,
    WARN,
    ClusterPreflight,
    Endpoint,
    Report,
    Steps,
    requirements_from_env,
    resolve_nodes,
    trainer_image,
)

SKIP_FLAG = "--skip-preflight"
DEFAULT_PREFIX = "baremetal"
# the gate: every node check and the TCP ports, no ib_write_bw and no NCCL benchmark
GATE_STEPS = Steps(ib=False, nccl=False)


def ssh_entries(gpu: str | None) -> list[str]:
    """The ``ssh:`` entries of a ``--gpu`` value."""
    return [e.strip() for e in str(gpu or "").split(",") if e.strip().startswith("ssh:")]


def argv_value(argv: Sequence[str], flag: str) -> str | None:
    """The value of ``flag`` in an argv (``--flag V`` or ``--flag=V``; the last wins)."""
    value = None
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            value = argv[i + 1]
        elif arg.startswith(flag + "="):
            value = arg[len(flag) + 1:]
    return value


def preflight_commands(gpu: str | None, inventory: str | None, prefix: str | None,
                       image: str | None = None) -> list[list[str]]:
    """``yeto preflight`` argv (after ``yeto``) of every ``ssh:`` entry of ``gpu``;
    ``image``: the run's ``--rl-image`` (the image the trainer runs)."""
    out = []
    for entry in ssh_entries(gpu):
        argv = ["preflight", "--gpu", entry]
        if inventory:
            argv += ["--rl-node-inventory", str(inventory)]
        if image:
            argv += ["--image", str(image)]
        out.append(argv + ["--no-bench", "--cluster-prefix", str(prefix or DEFAULT_PREFIX)])
    return out


def commands_for_launch_argv(argv: Sequence[str]) -> list[list[str]]:
    """The preflight commands of a ``yeto launch`` argv (``yeto run up``'s trainer)."""
    return preflight_commands(argv_value(argv, "--gpu"), argv_value(argv, "--rl-node-inventory"),
                              argv_value(argv, "--cluster-prefix"), argv_value(argv, "--rl-image"))


DOCKER_HUB = "registry-1.docker.io"
DOCKER_HUB_ALIASES = ("docker.io", "index.docker.io")  # what docker pulls from registry-1.docker.io


def registry_endpoint(image: str | None) -> Endpoint | None:
    """The registry an image is pulled from (``https://<host>/``; Docker Hub, for a
    reference without a registry host or one naming ``docker.io``, is
    ``registry-1.docker.io``); None for no image or a placeholder. The probe on the node
    takes ``http`` for a registry the docker daemon lists as insecure."""
    ref = str(image or "").strip().removeprefix("docker:")
    if not ref or "<" in ref:
        return None
    first, sep, _rest = ref.partition("/")
    host = first if sep and ("." in first or ":" in first or first == "localhost") else DOCKER_HUB
    if host in DOCKER_HUB_ALIASES:
        host = DOCKER_HUB
    return Endpoint("registry", f"https://{host}/", ref)


def s3_endpoint(url: str | None, region: str | None, store: str) -> Endpoint:
    """The object API of the ``s3://`` store ``store``: ``url``, else AWS S3 in ``region``."""
    if not url:
        url = f"https://s3.{region}.amazonaws.com" if region and region != "auto" else "https://s3.amazonaws.com"
    return Endpoint("store", url if "://" in url else f"https://{url}", store)


def store_endpoint(store: str | None, environ: Mapping[str, str]) -> Endpoint | None:
    """The object API the nodes use for the run store ``store``: an ``s3://`` store's
    S3-compatible endpoint (``YETO_RL_STORE_S3_ENDPOINT_URL`` / ``AWS_ENDPOINT_URL`` / the
    store profile's, else AWS S3 in the store's region), an ``oss://`` store's
    ``YETO_RL_STORE_OSS_ENDPOINT_URL``. None for any other store (a host path, a Modal
    Volume: nothing to reach over the network) or an endpoint not known here."""
    from ...cloud import oss_store, s3_store

    store = str(store or "")
    if store.startswith("s3://"):
        endpoint = s3_endpoint(s3_store.store_endpoint(environ), s3_store.store_region(environ), store)
    elif store.startswith("oss://"):
        url = environ.get(oss_store.ENDPOINT_ENV)
        endpoint = Endpoint("store", url if "://" in url else f"https://{url}", store) if url else None
    else:
        return None
    return None if endpoint is None or "<" in endpoint.url else endpoint


def launch_endpoints(argv: Sequence[str], environ: Mapping[str, str], nodes: Sequence[Any] = ()) -> list[Endpoint]:
    """What the trainer nodes of a ``yeto launch`` argv reach: the run store
    (``--rl-checkpoint-store``, at the endpoint the launch's store client takes:
    ``store_client_environ`` of ``--rl-object-store-endpoint`` / ``--rl-object-store-profile``
    and the environment) and, when a node of ``nodes`` (or none given) runs
    ``runtime: docker``, the registry of the trainer image (``--rl-image``, else the
    inventory's, else the pinned default). :func:`node_endpoints` keeps a node's own."""
    from types import SimpleNamespace

    from ..store_env import store_client_environ

    env = store_client_environ(SimpleNamespace(
        rl_object_store_endpoint=argv_value(argv, "--rl-object-store-endpoint"),
        rl_object_store_profile=argv_value(argv, "--rl-object-store-profile")), environ)
    found = [store_endpoint(argv_value(argv, "--rl-checkpoint-store"), env)]
    docker = [n for n in nodes if getattr(n, "runtime", "docker") == "docker"]
    if docker or not nodes:  # a rootfs node pulls no image
        image = argv_value(argv, "--rl-image") or next((n.image for n in docker if getattr(n, "image", None)), None)
        if not image:
            from ...rl import default_rl_image

            image = default_rl_image(recipe=argv_value(argv, "--rl-model-recipe"),
                                     tuning=argv_value(argv, "--tuning"))
        found.append(registry_endpoint(image))
    return [e for e in found if e is not None]


def node_endpoints(endpoints: Sequence[Endpoint], node: Any) -> tuple[Endpoint, ...]:
    """The endpoints of ``endpoints`` node ``node`` reaches itself: no registry for a
    node that does not run ``runtime: docker`` (it pulls no image)."""
    docker = getattr(node, "runtime", "docker") == "docker"
    return tuple(e for e in endpoints if docker or e.role != "registry")


def egress_lines(report: Report) -> list[str]:
    """The egress report in the preflight's line format (:func:`summary_lines`): one line per
    endpoint check of each node (PASS, WARN or FAIL, with its fix when not a PASS), the ssh
    check of a node only when it is not a PASS, then the counts."""
    lines = []
    for c in report.checks:
        if c.name.startswith("reach_") or c.status != PASS:
            lines.append(f"  [{c.status}] {c.scope} {c.name}: {c.detail}"
                         + (f" -> {c.fix}" if c.fix and c.status != PASS else ""))
    n = report.counts
    lines.append(f"  {n[PASS]} pass, {n[WARN]} warn, {n[FAIL]} fail, {n[SKIP]} skip")
    return lines


def _say(out: IO[str], line: str) -> None:
    print(line, file=out, flush=True)


def summary_lines(report: Report) -> list[str]:
    """One line per WARN / FAIL finding, with its fix."""
    lines = []
    for c in report.checks:
        if c.status in (WARN, FAIL):
            lines.append(f"  [{c.status}] {c.scope} {c.name}: {c.detail}" + (f" -> {c.fix}" if c.fix else ""))
    n = report.counts
    lines.append(f"  {n[PASS]} pass, {n[WARN]} warn, {n[FAIL]} fail, {n[SKIP]} skip")
    return lines


def launch_gate(args: Any, *, ops: Any = None, out: IO[str] | None = None,
                clock: Any = time.time) -> dict[str, Any] | None:
    """Run the cluster preflight of ``args`` (a ``yeto launch`` namespace) before the
    launch; raises ``ValueError`` on a FAIL. Returns the record (None: no ``ssh:`` island)."""
    out = out or sys.stderr
    gpu = getattr(args, "gpu", None)
    inventory = getattr(args, "rl_node_inventory", None)
    prefix = getattr(args, "cluster_prefix", None) or DEFAULT_PREFIX
    rl_image = getattr(args, "rl_image", None)
    commands = preflight_commands(gpu, inventory, prefix, rl_image)
    if not commands:
        return None
    shown = "; ".join("yeto " + shlex.join(c) for c in commands)
    if getattr(args, "skip_preflight", False):
        _say(out, f"[yeto] cluster preflight skipped ({SKIP_FLAG}): {shown}")
        return {"skipped": True, "commands": commands}
    if getattr(args, "dry_run", False):
        _say(out, f"[yeto] dry run: the launch would first run the cluster preflight: {shown}")
        return {"dry_run": True, "commands": commands}
    from ... import runs

    reports = []
    for entry in ssh_entries(gpu):
        nodes = resolve_nodes(entry, inventory)  # InventoryError is a ValueError: the launch is refused
        try:  # the image the trainer runs: --rl-image, an overlay-unsafe pin as its overlay-safe repack
            image = trainer_image(rl_image, nodes)
        except SshIslandError as exc:
            raise ValueError(str(exc)) from None
        out_dir = (runs.run_dir(prefix) / "preflight"
                   / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(clock())))
        _say(out, f"[yeto] cluster preflight on {len(nodes)} ssh node(s) of {entry} (node checks and TCP ports; "
                  f"{SKIP_FLAG} skips it) -> {out_dir}")
        report = ClusterPreflight(nodes, image=image, ops=ops, req=requirements_from_env(), steps=GATE_STEPS,
                                  out_dir=out_dir, log=lambda m: _say(out, m), clock=clock).run()
        for line in summary_lines(report):
            _say(out, line)
        reports.append({"gpu": entry, "ok": report.ok, "counts": report.counts, "report": str(out_dir)})
        if not report.ok:
            fails = [f"{c.scope} {c.name}" for c in report.checks if c.status == FAIL]
            raise ValueError(f"cluster preflight of {entry} failed ({', '.join(fails[:6])}"
                             f"{', ...' if len(fails) > 6 else ''}); fix the nodes (report "
                             f"{Path(out_dir) / 'preflight.txt'}) or pass {SKIP_FLAG}")
    return {"reports": reports}

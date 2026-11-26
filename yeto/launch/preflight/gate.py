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

import contextlib
import os
import shlex
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
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


@contextlib.contextmanager
def launch_environ(environ: Mapping[str, str] | None) -> Iterator[None]:
    """``os.environ`` as ``environ`` has it (its keys added or changed, none removed;
    nothing for None) inside the block: the environment a launch child of ``yeto run up``
    starts with (``Run.env``: run up's own plus the run config's ``env``), for resolving
    what that launch would. The overlay is the keys whose value differs from this
    process's now, applied as a scope of :func:`yeto.launch.pool.env_scope.scoped_environ`
    (the one mechanism of the process: it nests, raises ``EnvOverlayConflict`` when
    another overlay is active, and keeps a value something else wrote meanwhile). Inside
    a pool call of the same run (``Run.pool_environ`` applied: this process's env is
    ``Run.env``) the overlay is empty and nothing changes."""
    from ..pool.env_scope import scoped_environ

    overlay = {k: str(v) for k, v in (environ or {}).items() if os.environ.get(k) != str(v)}
    with scoped_environ(overlay):
        yield


def launch_rl_image(argv: Sequence[str], environ: Mapping[str, str] | None = None) -> str | None:
    """The ``--rl-image`` a ``yeto launch`` of ``argv`` runs with: the argv's own, else
    the image the launch fills in (``rl_backend.rl_image_choice``, the function the launch
    itself calls: the site image catalog for an ssh: trainer whose inventory names no
    image, else the engine's pinned default), resolved under the launch's environment
    (``environ`` over this process's, :func:`launch_environ`: ``YETO_IMAGE_REGISTRY``,
    ``YETO_SITE_IMAGES_PATH``, ``HOME`` decide). The argv's own ``--rl-image`` (None
    without one) when the argv is not an RL launch the parser takes; raises
    ``ValueError`` where the launch refuses to fill one in (an unknown backend, a broken
    catalog)."""
    given = argv_value(argv, "--rl-image")
    if given:
        return given
    from ...rl.up_consistency import trainer_args
    from ..rl_backend import rl_image_choice

    with launch_environ(environ):
        args = trainer_args(list(argv))
        if args is None:
            return given
        return rl_image_choice(args)[0]


def commands_for_launch_argv(argv: Sequence[str], environ: Mapping[str, str] | None = None) -> list[list[str]]:
    """The preflight commands of a ``yeto launch`` argv (``yeto run up``'s trainer, whose
    launch starts with ``environ``), with ``--image`` the image the launch runs
    (:func:`launch_rl_image`). A launch that refuses to fill in an image gets none: the
    preflight FAILs (no trainer image resolved), and the launch names why."""
    gpu = argv_value(argv, "--gpu")
    if not ssh_entries(gpu):
        return []
    try:
        image = launch_rl_image(argv, environ)
    except ValueError:
        image = argv_value(argv, "--rl-image")
    return preflight_commands(gpu, argv_value(argv, "--rl-node-inventory"), argv_value(argv, "--cluster-prefix"),
                              image)


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
    inventory's, else the image the launch fills in: :func:`launch_rl_image`).
    :func:`node_endpoints` keeps a node's own."""
    from types import SimpleNamespace

    from ..store_env import store_client_environ

    env = store_client_environ(SimpleNamespace(
        rl_object_store_endpoint=argv_value(argv, "--rl-object-store-endpoint"),
        rl_object_store_profile=argv_value(argv, "--rl-object-store-profile")), environ)
    found = [store_endpoint(argv_value(argv, "--rl-checkpoint-store"), env)]
    docker = [n for n in nodes if getattr(n, "runtime", "docker") == "docker"]
    if docker or not nodes:  # a rootfs node pulls no image
        image = argv_value(argv, "--rl-image") or next((n.image for n in docker if getattr(n, "image", None)), None)
        refused = False
        if not image:
            try:  # the image the launch fills in (the site image catalog's, else the pinned default)
                image = launch_rl_image(argv, environ)
            except ValueError:  # the launch refuses it and names why: no registry to check
                refused = True
        if not image and not refused:  # not an argv the launch parser takes
            from ...rl import default_rl_image

            with launch_environ(environ):
                image = default_rl_image(recipe=argv_value(argv, "--rl-model-recipe"),
                                         tuning=argv_value(argv, "--tuning"))
        if not refused:
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


def gate_nodes(args: Any, entry: str, inventory: str | None) -> list[Any]:
    """The nodes the gate checks for ``--gpu`` entry ``entry``: the inventory's (as
    ``yeto preflight`` resolves them); with an inventory ``lease_store`` the nodes the
    island's leases pick (``ssh_trainer_island.lease_trainer_nodes``: the launch then
    starts on the very nodes checked). Without ``lease_store`` exactly as before."""
    from ..node_inventory import lease_store_of

    nodes = getattr(args, "rl_node_inventory_nodes", None)
    if lease_store_of(nodes) is None or Path(entry[4:]).expanduser().is_file():
        return resolve_nodes(entry, inventory)
    from ...gpu_spec import parse_gpu_spec
    from ..sky_ops import learner_cluster_names
    from ..node_inventory import InventoryError
    from ..ssh_trainer_island import inventory_trainer_nodes, trainer_leases

    specs = parse_gpu_spec(args.gpu)
    entries = [e.strip() for e in str(args.gpu).split(",") if e.strip()]
    m = entries.index(entry) if len(entries) == len(specs) else next(
        i for i, spec in enumerate(specs) if spec.cloud == "ssh")
    try:
        return list(inventory_trainer_nodes(args, specs[m], leases=trainer_leases(args),
                                            name=learner_cluster_names(args.cluster_prefix, specs)[m]))
    except SshIslandError as exc:
        raise InventoryError(str(exc)) from None


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
        nodes = gate_nodes(args, entry, inventory)  # InventoryError is a ValueError: the launch is refused
        try:  # the image the trainer runs: --rl-image, an overlay-unsafe pin as its overlay-safe repack
            image = trainer_image(rl_image, nodes)
        except SshIslandError as exc:
            raise ValueError(str(exc)) from None
        out_dir = (runs.run_dir(prefix) / "preflight"
                   / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(clock())))
        _say(out, f"[yeto] cluster preflight on {len(nodes)} ssh node(s) of {entry} (node checks and TCP ports; "
                  f"{SKIP_FLAG} skips it) -> {out_dir}")
        report = ClusterPreflight(nodes, image=image, ops=ops, req=requirements_from_env(), steps=GATE_STEPS,
                                  out_dir=out_dir, log=lambda m: _say(out, m), clock=clock,
                                  inventory=getattr(args, "rl_node_inventory_nodes", None)).run()
        for line in summary_lines(report):
            _say(out, line)
        reports.append({"gpu": entry, "ok": report.ok, "counts": report.counts, "report": str(out_dir)})
        if not report.ok:
            fails = [f"{c.scope} {c.name}" for c in report.checks if c.status == FAIL]
            raise ValueError(f"cluster preflight of {entry} failed ({', '.join(fails[:6])}"
                             f"{', ...' if len(fails) > 6 else ''}); fix the nodes (report "
                             f"{Path(out_dir) / 'preflight.txt'}) or pass {SKIP_FLAG}")
    return {"reports": reports}

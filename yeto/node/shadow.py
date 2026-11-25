"""Shadow comparison of a route's own sampler start with the node runtime's.

National Compute (the customer's MI355X vLLM pool) and Daytona stay on their own
sampler starts until the node runtime renders the same launch there. This module
puts both sides in one shape, :class:`Launch` (device flags, mounts, env files,
environment, image, interpreter, sampler module, spec path), and lists every
difference (:func:`compare`). A route may switch only when the list is empty
(:func:`switchable`); until then the list is the work left.

- the route's side: a ``docker run`` command (:func:`docker_launch`, the NC route)
  or a start script (:func:`script_launch`, the Daytona process route);
- the node runtime's side: :func:`node_plan` on the route's node profile
  (:func:`nc_profile`, :func:`daytona_profile`), the plan ``python -m yeto.node run
  --dry-run`` prints, as a launch (:func:`plan_launch`).

Lifecycle flags are not compared (``-d`` / ``--rm`` / ``--name`` / ``--restart``):
the node runtime keeps its sampler in the foreground as its child, the routes
start it detached. Nothing here touches a node or the store.
"""

from __future__ import annotations

import dataclasses
import re
import shlex
from collections.abc import Mapping, Sequence
from typing import Any

from yeto.launch.node_inventory import NodePaths, NodeProfile

from .launch import GENERATION_ENV, HEARTBEAT_ENV, PROFILE_ENV, SUPERVISED_ENV, TICKET_ENV
from .records import Assignment, CodeRef

# the route-specific tuning a route's node profile carries (``env``), not derived by the runtime
NC_TUNING_ENV = ("VLLM_USE_BREAKABLE_CUDAGRAPH", "YETO_RL_STORE_DOWNLOAD_PROCESSES", "YETO_RL_STORE_FILE_CONCURRENCY",
                 "YETO_RL_RELEASE_SPREAD_S")
DAYTONA_TUNING_ENV = (*NC_TUNING_ENV, "XDG_CACHE_HOME", "VLLM_CACHE_ROOT", "TRITON_CACHE_DIR", "AITER_JIT_DIR",
                      "YETO_CACHE_DIR")
# only the node runtime sets these (its records, the family for the sampler's telemetry)
NODE_ONLY_ENV = frozenset({TICKET_ENV, GENERATION_ENV, PROFILE_ENV, SUPERVISED_ENV, HEARTBEAT_ENV,
                           "YETO_DEVICE_FAMILY"})
_LIFECYCLE = {"-d", "--rm", "--restart=no", "--detach"}
_WITH_VALUE = {"--name", "--restart"}
_EXEC = re.compile(r"exec\s+(\S+)\s+-m\s+(\S+)\s+--spec\s+(\S+?)['\"]*(?:\s|$)")


@dataclasses.dataclass(frozen=True)
class Launch:
    devices: tuple[str, ...] = ()  # every docker flag that is not a mount, env, lifecycle or the image
    mounts: Mapping[str, str] = dataclasses.field(default_factory=dict)  # container path -> host[:ro]
    env_files: tuple[str, ...] = ()
    env: Mapping[str, str] = dataclasses.field(default_factory=dict)
    image: str | None = None
    python: str | None = None
    module: str | None = None
    spec: str | None = None

    def record(self) -> dict[str, Any]:
        return {**dataclasses.asdict(self), "mounts": dict(self.mounts), "env": dict(self.env)}


def _sampler(command: str) -> tuple[str | None, str | None, str | None]:
    """``(python, module, spec)`` of the ``exec <python> -m <module> --spec <path>`` in ``command``."""
    found = _EXEC.search(command)
    return (found.group(1).strip("'\""), found.group(2), found.group(3).strip("'\"")) if found else (None, None, None)


def docker_launch(command: str | Sequence[str]) -> Launch:
    """The launch of a ``docker run`` (an argv, or a shell line whose ``docker run`` part is parsed)."""
    if isinstance(command, str):
        text = command[command.index("docker run") - len("sudo ") if "sudo docker run" in command
                       else command.index("docker run"):]
        argv = shlex.split(text)
    else:
        argv = list(command)
    i = argv.index("run") + 1
    devices: list[str] = []
    mounts: dict[str, str] = {}
    env_files: list[str] = []
    env: dict[str, str] = {}
    image = inner = None
    while i < len(argv):
        a = argv[i]
        if a in _LIFECYCLE:
            i += 1
        elif a in _WITH_VALUE:
            i += 2
        elif a == "-v":
            host, _, inside = argv[i + 1].partition(":")
            inside, _, mode = inside.partition(":")
            mounts[inside] = host + (f":{mode}" if mode else "")
            i += 2
        elif a == "--env-file":
            env_files.append(argv[i + 1])
            i += 2
        elif a == "-e":
            key, _, value = argv[i + 1].partition("=")
            env[key] = value
            i += 2
        elif a == "--entrypoint":
            image, inner = argv[i + 2], " ".join(argv[i + 3:])
            break
        elif a.startswith("-") and i + 1 < len(argv) and not argv[i + 1].startswith("-") and "=" not in a:
            devices += [a, argv[i + 1]]
            i += 2
        else:
            devices.append(a)
            i += 1
    python, module, spec = _sampler(inner or "")
    return Launch(tuple(devices), mounts, tuple(env_files), env, image, python, module, spec)


def script_launch(script: str) -> Launch:
    """The launch of a start script (``. <env file>`` lines, ``export K=V`` lines, the
    ``exec <python> -m <module> --spec <path>`` line). Exports with shell expansions
    are kept as written."""
    env_files: list[str] = []
    env: dict[str, str] = {}
    for line in script.splitlines():
        line = line.strip()
        if line.startswith(". "):
            env_files.append(shlex.split(line[2:])[0])
        elif line.startswith("export "):
            for part in shlex.split(line[len("export "):]):
                key, _, value = part.partition("=")
                env[key] = value
    python, module, spec = _sampler(script)
    return Launch(env_files=tuple(env_files), env=env, python=python, module=module, spec=spec)


def plan_launch(plan: Any) -> Launch:
    """The node runtime's plan (:class:`yeto.node.runtime.Plan`, or a map with its ``argv``
    and ``env``) as a launch."""
    argv = list(plan["argv"] if isinstance(plan, Mapping) else plan.argv)
    env = dict(plan.get("env") or {}) if isinstance(plan, Mapping) else dict(plan.env)
    if "run" in argv[:4]:  # a container: its environment is in its argv
        return docker_launch(argv)
    python, module, spec = argv[0], argv[2], argv[argv.index("--spec") + 1]
    return Launch(env=env, python=python, module=module, spec=spec)


def compare(old: Launch, new: Launch, *, node_only: frozenset[str] = NODE_ONLY_ENV) -> list[str]:
    """Every difference of ``new`` (the node runtime) from ``old`` (the route), one line each."""
    out: list[str] = []
    for name in ("devices", "env_files", "image", "python", "module", "spec"):
        a, b = getattr(old, name), getattr(new, name)
        if a != b:
            out.append(f"{name}: {a!r} -> {b!r}")
    for inside in sorted(set(old.mounts) | set(new.mounts)):
        a, b = old.mounts.get(inside), new.mounts.get(inside)
        if a != b:
            out.append(f"mount {inside}: {a or '(none)'} -> {b or '(none)'}")
    for key in sorted(set(old.env) | set(new.env)):
        if key in node_only and key not in old.env:
            continue
        a, b = old.env.get(key), new.env.get(key)
        if a != b:
            out.append(f"env {key}: {a if a is not None else '(unset)'} -> {b if b is not None else '(unset)'}")
    return out


def switchable(differences: Sequence[str]) -> bool:
    """A route may start its samplers through the node runtime only on an identical launch."""
    return not differences


def nc_profile(lease: Any, *, image: str | None = None) -> NodeProfile:
    """A National Compute node as a node profile: the route's node inventory entry
    (MI355X, its ROCm docker flags, its image) with the NVMe directories."""
    from yeto.launch.nc import ssh_sampler  # noqa: PLC0415

    node = ssh_sampler.node_of(lease, **({"image": image} if image else {}))
    root = ssh_sampler.NVME
    tuning = {k: v for k, v in ssh_sampler.sampler_env("-").items() if k in NC_TUNING_ENV}
    return NodeProfile.from_node(node, container="docker", model_source="store", env=tuning,
                                 paths=NodePaths(work=f"{root}/work", models=f"{root}/models",
                                                 cache=f"{root}/cache/vllm"))


def daytona_profile(name: str, *, gpus: int = 8) -> NodeProfile:
    """A Daytona sandbox as a node profile: MI355X, a process in the sandbox's image."""
    from yeto.launch import daytona_sampler as ds  # noqa: PLC0415

    tuning = {k: v for k, v in ds.process_env("-", "-", gpus=gpus).items() if k in DAYTONA_TUNING_ENV}
    return NodeProfile.for_card(name, "MI355X", gpus, hbm_gib=288.0, container="process", env=tuning,
                                paths=NodePaths(work=ds.WORK, models=ds.MODELS, cache=ds.CACHE))


def node_plan(profile: NodeProfile, *, spec: Mapping[str, Any], shard: str, run_prefix: str, code_sha: str,
              store_uri: str | None = None, env_file: str | None = None, image: str | None = None,
              python: str | None = None, source: str | None = None, environ: Mapping[str, str] | None = None,
              env: Mapping[str, str] | None = None, cards: tuple[int, ...] | None = None) -> Any:
    """The plan the node runtime would start ``shard`` with on ``profile`` (its dry run;
    no store, nothing written)."""
    from .runtime import NodeRuntime  # noqa: PLC0415

    assignment = Assignment(ticket=f"shadow-{shard}", shard=shard, spec=dict(spec),
                            code=CodeRef(sha=code_sha, key=f"node-code/{code_sha}.tar", sha256=""), token_sha256="",
                            run=run_prefix.strip("/"), source=source, image=image, python=python,
                            store_uri=store_uri, env=dict(env or {}),
                            cards=tuple(range(profile.cards_per_node)) if cards is None else cards)
    runtime = NodeRuntime(None, assignment, profile, store_uri=store_uri, environ=dict(environ or {}),
                          skip_preflight=True, log=lambda line: None)
    runtime.env_file = env_file
    from . import code as node_code  # noqa: PLC0415
    from .profile import node_dirs  # noqa: PLC0415

    return runtime.plan(node_code.code_dir(node_dirs(profile)["work"], code_sha))

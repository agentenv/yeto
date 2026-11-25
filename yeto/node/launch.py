"""How the node runtime starts the sampler: its environment and its command.

The sampler is always a child of the node runtime, in the foreground: a process
(``python -m yeto.rl.engine.engine_sampler --spec``) or a container run with
``--rm`` (the profile's device flags, the node's directories mounted at the
container paths, ``--network host --ipc host``). The runtime waits for it,
forwards SIGTERM (the sampler's orderly leave, ``YETO_SPOT_RECLAIM_MARKER``) and
writes its exit record.
"""

from __future__ import annotations

import json
import shlex
from collections.abc import Mapping, Sequence

from yeto.launch.node_inventory import NodeProfile

from .paths import sampler_dirs
from .profile import CONTAINER_MODELS, container_mounts
from .records import Assignment
from .store import reclaim_marker_env, sampler_route_env

SAMPLER_MODULE = "yeto.rl.engine.engine_sampler"
# the node records a sampler writes (engine_sampler_entry, yeto.node.report)
TICKET_ENV = "YETO_NODE_TICKET"
GENERATION_ENV = "YETO_NODE_GENERATION"
PROFILE_ENV = "YETO_NODE_PROFILE"
SUPERVISED_ENV = "YETO_NODE_SUPERVISED"  # the node runtime writes the exit record
HEARTBEAT_ENV = "YETO_NODE_HEARTBEAT_S"


def container_name(shard: str) -> str:
    return f"yeto-sampler-{shard}"


def sampler_paths(profile: NodeProfile, shard: str) -> dict[str, str]:
    """The sampler's files as the sampler sees them (container or process paths)."""
    work = sampler_dirs(profile)["work"]
    return {"spec": f"{work}/sampler/{shard}.json", "reclaim": f"{work}/sampler/{shard}.reclaim",
            "stage": f"{work}/stage", "log": f"{work}/logs/{shard}.log"}


def store_env(store_uri: str | None, *, stage: str, environ: Mapping[str, str]) -> dict[str, str]:
    """The object store settings of an ``s3://`` / ``r2://`` run store (credentials stay
    in the env file / the runtime's environment); nothing for a store directory."""
    if not store_uri or "://" not in store_uri:
        return {}
    from yeto.cloud import s3_store  # noqa: PLC0415

    out = {"YETO_RL_STORE_IMPL": s3_store.STORE_IMPL, s3_store.URI_ENV: s3_store.s3_uri(store_uri),
           s3_store.STAGE_ENV: stage, s3_store.REGION_ENV: environ.get(s3_store.REGION_ENV, "auto"),
           s3_store.CREDENTIALS_ENV: environ.get(s3_store.CREDENTIALS_ENV, s3_store.CREDENTIALS_AWS_ENV)}
    endpoint = environ.get(s3_store.ENDPOINT_ENV) or environ.get(s3_store.AWS_ENDPOINT_ENV)
    if endpoint:
        out[s3_store.ENDPOINT_ENV] = endpoint
    return out


def sampler_env(assignment: Assignment, profile: NodeProfile, *, yeto_dir: str, store_uri: str | None,
                environ: Mapping[str, str], heartbeat_s: float | None = None,
                host_paths: Sequence[str] | None = None) -> dict[str, str]:
    """The sampler's non-secret environment: the yeto tree, the reclaim marker (SIGTERM
    is its orderly leave), the release download's spread key, the family's visible
    cards and ``YETO_DEVICE_FAMILY``, the store settings, the node record identity,
    then the profile's and the assignment's own variables (the assignment's win)."""
    p = sampler_paths(profile, assignment.shard)
    devices = profile.devices(assignment.cards, host_paths=host_paths)
    sampler_dir = p["reclaim"].rsplit("/", 1)[0]
    env = {"PYTHONUNBUFFERED": "1", "PYTHONPATH": yeto_dir, **reclaim_marker_env(assignment.shard, sampler_dir),
           **devices.env, **profile.family_env()}
    if assignment.run:
        env.update(sampler_route_env(assignment.run, assignment.shard, sampler_dir))
    env.update(store_env(store_uri, stage=p["stage"], environ=environ))
    summary = {**profile.summary(), "name": profile.name, "source": assignment.source}
    env.update({TICKET_ENV: assignment.ticket, GENERATION_ENV: str(assignment.generation),
                PROFILE_ENV: json.dumps({k: v for k, v in summary.items() if v is not None}, sort_keys=True),
                SUPERVISED_ENV: "1"})
    if heartbeat_s is not None:
        env[HEARTBEAT_ENV] = f"{float(heartbeat_s):g}"
    env.update(profile.env)
    env.update({str(k): str(v) for k, v in dict(assignment.env).items()})
    return env


def process_argv(python: str, spec_path: str) -> list[str]:
    return [python, "-m", SAMPLER_MODULE, "--spec", spec_path]


def engine_prefix(profile: NodeProfile, sudo: bool) -> list[str]:
    return [*(["sudo", "-n"] if sudo else []), profile.container]


def container_argv(profile: NodeProfile, *, name: str, image: str, env: Mapping[str, str], python: str,
                   spec_path: str, env_files: Sequence[str] = (), cards: Sequence[int] | None = None,
                   host_paths: Sequence[str] | None = None, sudo: bool = False) -> list[str]:
    """``<engine> run --rm`` of the sampler in the foreground (the runtime is its parent)."""
    argv = [*engine_prefix(profile, sudo), "run", "--rm", "--name", name, "--network", "host", "--ipc", "host",
            *profile.devices(cards, host_paths=host_paths).args]
    for host, inside in sorted(container_mounts(profile).items()):
        argv += ["-v", f"{host}:{inside}{':ro' if inside == CONTAINER_MODELS else ''}"]
    for path in env_files:
        argv += ["--env-file", path]
    for key, value in sorted(env.items()):
        argv += ["-e", f"{key}={value}"]
    return [*argv, "--entrypoint", "bash", image, "-c",
            f"exec {shlex.quote(python)} -m {SAMPLER_MODULE} --spec {shlex.quote(spec_path)}"]


def stop_argv(profile: NodeProfile, name: str, *, sudo: bool = False) -> list[str]:
    """SIGTERM to the container's PID 1 (the sampler)."""
    return [*engine_prefix(profile, sudo), "kill", "--signal", "TERM", name]

"""Model weights onto the SSH trainer nodes before the first start.

With ``--model-store local:///PATH`` and ``--model-store-source URI`` the
preflight of an SSH trainer island (``SshTrainerIslands.preflight``) runs
:mod:`yeto.launch.weights_sync` on every node at once: the nodes copy
``--model@--model-revision`` from the source (``s3://`` / ``r2://`` object
store, or a directory such as an NFS mount) into the node-local directory, in
the Hugging Face hub cache layout, with resumable chunked downloads and sha256
checks, and refuse to start without the disk space. Their progress lines reach
the launcher log. The island's setup and run scripts then find the completed
snapshot there (``storage.model_store_env``: ``HF_HUB_CACHE`` and
``YETO_MODEL_STORE_HIT``), the model fetches resolve it with
``local_files_only`` and nothing reads the Hub.

``PATH`` is the directory as the trainer sees it: on a docker node it must lie
under one of the node's inventory ``mounts`` (the host directory is found
through the mount), on a rootfs node it is the host path.

Pipeline stages (``--model-store-shards``): ``stage`` gives each node only the
safetensors files of the pipeline stages its GPUs run, plus the files outside
the decoder layers and all small files (``weights_sync.select_shards``);
``all`` copies every file; ``auto`` (the default) is ``stage`` for a
trainer-only island (``--rl-sample-role trainer``) with ``--pipeline-parallel``
> 1, else ``all``. The stage of a GPU follows Miles' placement: the trainer's
ranks are the island's GPUs ordered by node IP (``miles.ray.placement_group.
sort_key``, the Ray node IP being the node's rendezvous address), then GPU id;
Megatron's rank order ``tp-cp-ep-dp-pp`` makes the pipeline stage the outermost
dimension, so stage ``s`` holds ranks ``[s*W/PP, (s+1)*W/PP)``. The uneven
split comes from ``--decoder-first/last-pipeline-num-layers`` in
``YETO_MILES_TRAIN_EXTRA_ARGV`` (the run's environment), else yeto's default.
mbridge opens only the files of the tensors a stage loads, so the full
``model.safetensors.index.json`` stays as it is.

Source credentials (an object store): ``YETO_WEIGHTS_S3_ACCESS_KEY_ID`` /
``_SECRET_ACCESS_KEY`` / ``_SESSION_TOKEN`` / ``_ENDPOINT_URL`` / ``_REGION`` of
the launcher's environment, else the run's ``--rl-checkpoint-store``
credentials (``YETO_RL_STORE_S3_*``, the same bucket as the weights on R2).
The launcher's own cloud credentials (``AWS_*``) never go to a node.
"""

from __future__ import annotations

import json
import os
import shlex
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import weights_sync

TOOL = Path(weights_sync.__file__)
SHARD_MODES = ("auto", "all", "stage")
CREDENTIAL_NAMES = ("ACCESS_KEY_ID", "SECRET_ACCESS_KEY", "SESSION_TOKEN", "ENDPOINT_URL", "REGION")
STORE_ENV_PREFIX = "YETO_RL_STORE_S3_"
TRAIN_EXTRA_ENV = "YETO_MILES_TRAIN_EXTRA_ARGV"
HEREDOC = "__YETO_WEIGHTS_SYNC_PY__"
SYNC_TIMEOUT_S = 6 * 3600.0
KEEP_LINES = 30  # of a failed node's output, in the error


class WeightsSyncError(RuntimeError):
    """The weights could not be placed on the trainer nodes."""


@dataclass(frozen=True)
class WeightsPlan:
    """What every node copies (module doc). ``store``: the node-local store root
    as the trainer sees it; ``env``: the source credentials for the nodes."""

    model: str
    revision: str
    store: str
    source: str
    stage_shards: bool = False
    pipeline_parallel: int = 1
    pp_first: int | None = None
    pp_last: int | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    workers: int = weights_sync.DEFAULT_WORKERS
    procs: bool = True
    chunk_mb: int = weights_sync.DEFAULT_CHUNK_MB
    headroom: float = weights_sync.DEFAULT_HEADROOM
    timeout_s: float = SYNC_TIMEOUT_S


# ---------------------------------------------------------------- stages of each node
def miles_sort_key(address: str) -> list[int]:
    """Miles' placement order of a Ray node IP (``placement_group.sort_key``): the
    dotted IPv4 numbers, else the characters (no DNS lookup: a rendezvous address
    is an IP)."""
    try:
        return list(map(int, address.split(".")))
    except ValueError:
        return [ord(c) for c in address]


def node_stages(addresses: Sequence[str], cards_per_node: int, pipeline_parallel: int) -> list[tuple[int, ...]]:
    """The pipeline stages each node (in ``addresses`` order) runs, when the
    trainer has every GPU of the island (module doc)."""
    pp, cards = int(pipeline_parallel), int(cards_per_node)
    world = len(addresses) * cards
    if pp < 1 or world % pp:
        raise WeightsSyncError(f"{world} GPUs do not split into {pp} pipeline stages")
    per_stage = world // pp
    order = sorted(range(len(addresses)), key=lambda i: (miles_sort_key(addresses[i]), i))
    out: list[tuple[int, ...]] = [()] * len(addresses)
    for position, index in enumerate(order):
        ranks = range(position * cards, (position + 1) * cards)
        out[index] = tuple(sorted({r // per_stage for r in ranks}))
    return out


def pipeline_split(raw: str | None) -> tuple[int | None, int | None]:
    """``--decoder-first/last-pipeline-num-layers`` of a train extra argv (the last
    value of each; space- or comma-separated as ``config.train_extra_argv``)."""
    tokens = shlex.split((raw or "").replace(",", " "))
    found: dict[str, int] = {}
    for i, token in enumerate(tokens):
        name, eq, value = token.partition("=")
        if name in ("--decoder-first-pipeline-num-layers", "--decoder-last-pipeline-num-layers"):
            text = value if eq else (tokens[i + 1] if i + 1 < len(tokens) else "")
            try:
                found[name] = int(text)
            except ValueError:
                raise WeightsSyncError(f"{name} needs an integer layer count, got {text!r}") from None
    return found.get("--decoder-first-pipeline-num-layers"), found.get("--decoder-last-pipeline-num-layers")


# ---------------------------------------------------------------- node paths and script
def host_path(node: Any, path: str) -> str:
    """The host directory of the trainer's ``path`` on ``node``: through the
    deepest inventory mount that holds it on a docker node, ``path`` itself on a
    rootfs node."""
    if getattr(node, "runtime", "docker") != "docker":
        return path
    best = None
    for host, container in (getattr(node, "mounts", None) or {}).items():
        inner = str(container).rstrip("/") or "/"
        holds = path == inner or path.startswith(inner.rstrip("/") + "/")
        if holds and (best is None or len(inner) > len(best[1])):
            best = (str(host), inner)
    if best is None:
        raise WeightsSyncError(
            f"node {node.name}: the model store {path} is not under any inventory mount of the node "
            f"({sorted((getattr(node, 'mounts', None) or {}).values())}); mount its NVMe directory there, "
            f"e.g. mounts: {{/mnt/nvme/yeto-models: {path}}}")
    host, inner = best
    rest = path[len(inner):].strip("/")
    return os.path.join(host, rest) if rest else host


def sync_argv(plan: WeightsPlan, root: str, stages: Sequence[int] | None, node_name: str) -> list[str]:
    argv = ["sync", "--source", plan.source, "--root", root, "--model", plan.model, "--revision", plan.revision,
            "--node", node_name, "--workers", str(plan.workers), "--chunk-mb", str(plan.chunk_mb),
            "--headroom", f"{plan.headroom:g}"]
    if plan.procs:
        argv.append("--procs")
    if stages is not None:
        argv += ["--stages", ",".join(str(s) for s in stages), "--pp", str(plan.pipeline_parallel)]
        if plan.pp_first is not None:
            argv += ["--pp-first", str(plan.pp_first)]
        if plan.pp_last is not None:
            argv += ["--pp-last", str(plan.pp_last)]
    return argv


def node_script(plan: WeightsPlan, node: Any, stages: Sequence[int] | None, *, image: str | None = None,
                tool_source: str | None = None) -> str:
    """The bash script one node runs over SSH (stdin, so the credentials it
    exports are in no argv): write the sync tool to a temporary file and run it
    with the host's ``python3``; a docker node without one runs it in the
    trainer image (pulled by the preflight) with the store directory mounted
    (also when the host's python3 is older than 3.8)."""
    source = TOOL.read_text() if tool_source is None else tool_source
    if HEREDOC in source:
        raise WeightsSyncError(f"the sync tool contains the heredoc delimiter {HEREDOC}")
    root = host_path(node, plan.store)
    argv = shlex.join(sync_argv(plan, root, stages, node.name))
    exports = "".join(f"export {k}={shlex.quote(str(v))}\n" for k, v in sorted(plan.env.items()))
    names = " ".join(f"-e {k}" for k in sorted(plan.env))
    if getattr(node, "runtime", "docker") == "docker" and image:
        fallback = (f"  docker run --rm --network host --user \"$(id -u):$(id -g)\" "
                    f"-v {shlex.quote(root)}:{shlex.quote(root)} "
                    f'-v "$tool":/yeto-weights-sync.py:ro {names} --entrypoint python3 {shlex.quote(image)} '
                    f"-I /yeto-weights-sync.py {argv}\n")
    else:
        fallback = "  echo '[yeto-weights] ERROR: no python3 >= 3.8 on this node' >&2; exit 127\n"
    return (
        "set -eu\n"
        + exports
        + f"mkdir -p {shlex.quote(root)}\n"
        + 'tool=$(mktemp "${TMPDIR:-/tmp}/yeto-weights-sync.XXXXXX")\n'
        + "trap 'rm -f \"$tool\"' EXIT\n"
        + f"cat > \"$tool\" <<'{HEREDOC}'\n{source.rstrip()}\n{HEREDOC}\n"
        # the tool needs python >= 3.8 (an older host python3 takes the fallback)
        + "if python3 -c 'import sys; sys.exit(sys.version_info < (3, 8))' >/dev/null 2>&1; then\n"
        + f'  python3 -I "$tool" {argv}\n'
        + "else\n"
        + fallback
        + "fi\n"
    )


# ---------------------------------------------------------------- run on every node
def sync_weights(plan: WeightsPlan, nodes: Sequence[Any], addresses: Sequence[str], ops: Any, *,
                 images: Sequence[str | None] | None = None, out: Callable[[str], None] | None = None,
                 clock: Callable[[], float] = time.monotonic) -> list[dict[str, Any]]:
    """Run the sync on every node at once (``ops.start``: the SSH process with its
    output stream); raises :class:`WeightsSyncError` naming every node that
    failed. Returns each node's summary."""
    say = out or (lambda line: print(line, flush=True))
    stages = (node_stages(addresses, nodes[0].cards_per_node, plan.pipeline_parallel)
              if plan.stage_shards else [None] * len(nodes))
    images = list(images) if images is not None else [None] * len(nodes)
    say(f"[launcher] weights: {plan.model}@{plan.revision[:12]} from {plan.source} into {plan.store} on "
        f"{len(nodes)} node(s)" + (f", stage files of PP {plan.pipeline_parallel}"
                                   + (f" (first {plan.pp_first} / last {plan.pp_last} layers)"
                                      if plan.pp_first is not None or plan.pp_last is not None else "")
                                   if plan.stage_shards else ", every file"))
    procs, tails, summaries = [], [[] for _ in nodes], [None] * len(nodes)
    threads = []
    for i, node in enumerate(nodes):
        if stages[i] is not None:
            say(f"[launcher] weights: {node.name} ({addresses[i]}) runs stage(s) {','.join(map(str, stages[i]))}")
        procs.append(ops.start(node, node_script(plan, node, stages[i], image=images[i])))

    def pump(i: int) -> None:
        for line in procs[i].stdout:
            line = line.rstrip("\n")
            if line.startswith(weights_sync.SUMMARY_TAG + " "):
                try:
                    summaries[i] = json.loads(line[len(weights_sync.SUMMARY_TAG) + 1:])
                except ValueError:
                    pass
                continue
            tails[i] = (tails[i] + [line])[-KEEP_LINES:]
            say(f"[weights {nodes[i].name}] {line}")

    for i in range(len(nodes)):
        thread = threading.Thread(target=pump, args=(i,), daemon=True, name=f"weights-{nodes[i].name}")
        thread.start()
        threads.append(thread)
    deadline = clock() + plan.timeout_s
    failed = []
    for i, proc in enumerate(procs):
        try:
            rc = proc.wait(timeout=max(1.0, deadline - clock()))
        except Exception:  # noqa: BLE001 - subprocess.TimeoutExpired, or a dead stream
            for other in procs:
                if other.poll() is None:
                    other.terminate()
            raise WeightsSyncError(f"weights sync did not finish within {plan.timeout_s:.0f} s on "
                                   f"{nodes[i].name}") from None
        threads[i].join(timeout=30)
        if rc != 0 or summaries[i] is None:
            failed.append(f"{nodes[i].name} (rc {rc}): " + " | ".join(tails[i][-5:]))
    if failed:
        raise WeightsSyncError("weights sync failed on " + "; ".join(failed))
    for i, node in enumerate(nodes):
        s = summaries[i]
        say(f"[launcher] weights on {node.name}: {s.get('bytes_total', 0) / 1e9:.1f} GB in place "
            f"({s.get('downloaded', 0)} downloaded, {s.get('bytes_fetched', 0) / 1e9:.1f} GB at "
            f"{s.get('fetch_gbps', 0):.2f} GB/s; {s.get('skipped', 0)} verified earlier) in {s.get('seconds', 0):.0f} s")
    return summaries


# ---------------------------------------------------------------- from the launch arguments
def _extra_argv(args: Any, env: Mapping[str, str]) -> str | None:
    if env.get(TRAIN_EXTRA_ENV):
        return env[TRAIN_EXTRA_ENV]
    for item in getattr(args, "modal_env", None) or ():
        name, eq, value = str(item).partition("=")
        if eq and name == TRAIN_EXTRA_ENV:
            return value
    return os.environ.get(TRAIN_EXTRA_ENV)


def source_env(source: str, env: Mapping[str, str], environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The credentials and endpoint the nodes read an object-store source with
    (module doc); {} for a directory source."""
    if not source.startswith(("s3://", "r2://")):
        return {}
    launcher = os.environ if environ is None else environ
    own = {name: launcher[weights_sync.ENV_PREFIX + name] for name in CREDENTIAL_NAMES
           if launcher.get(weights_sync.ENV_PREFIX + name)}
    if own.get("ACCESS_KEY_ID") and own.get("SECRET_ACCESS_KEY"):
        picked = own
    else:
        picked = {name: env[STORE_ENV_PREFIX + name] for name in CREDENTIAL_NAMES
                  if env.get(STORE_ENV_PREFIX + name)}
        picked.update({k: v for k, v in own.items() if k in ("ENDPOINT_URL", "REGION")})
    if not (picked.get("ACCESS_KEY_ID") and picked.get("SECRET_ACCESS_KEY")):
        raise WeightsSyncError(
            f"--model-store-source {source}: no credentials for the nodes; set {weights_sync.ENV_PREFIX}"
            f"ACCESS_KEY_ID and {weights_sync.ENV_PREFIX}SECRET_ACCESS_KEY (and _ENDPOINT_URL for R2) on the "
            "launcher, or use an s3:// --rl-checkpoint-store whose key reads the weights bucket")
    return {weights_sync.ENV_PREFIX + k: v for k, v in picked.items()}


def weights_plan_from_args(args: Any, env: Mapping[str, str], nodes: Sequence[Any],
                           environ: Mapping[str, str] | None = None) -> WeightsPlan | None:
    """The plan of the held-back ``ssh:`` trainer island (None without
    ``--model-store-source``). Raises :class:`WeightsSyncError` before any SSH."""
    from .storage import model_store_local_path

    source = getattr(args, "model_store_source", None)
    if not source:
        return None
    store = model_store_local_path(getattr(args, "model_store", None))
    if store is None:
        raise WeightsSyncError("--model-store-source needs --model-store local:///PATH (the node directory "
                               "the weights go to)")
    from yeto.models import resolve
    from yeto.provenance import is_local_reference

    model = str(resolve(args.model))
    if is_local_reference(model):
        raise WeightsSyncError(f"--model-store-source fills a hub cache: --model must be a Hub id "
                               f"(org/name), got the path {model}")
    revision = str(getattr(args, "model_revision", None) or "")
    if not weights_sync.HEX40.fullmatch(revision):
        raise WeightsSyncError("--model-store-source needs --model-revision as a 40-hex commit "
                               f"(the snapshot's name), got {revision or 'none'!r}")
    mode = getattr(args, "model_store_shards", None) or "auto"
    if mode not in SHARD_MODES:
        raise WeightsSyncError(f"--model-store-shards {mode!r}: expected one of {', '.join(SHARD_MODES)}")
    pp = int(getattr(args, "pipeline_parallel", 1) or 1)
    if mode == "auto":
        from yeto.rl.engine import sample_roles

        stage = pp > 1 and sample_roles.sample_role(args) == sample_roles.TRAINER
    else:
        stage = mode == "stage"
    first = last = None
    if stage:
        if pp < 2:
            raise WeightsSyncError("--model-store-shards stage needs --pipeline-parallel > 1")
        gpus = getattr(args, "actor_num_gpus_per_node", None)
        if gpus and int(gpus) != int(nodes[0].cards_per_node):
            raise WeightsSyncError(f"--model-store-shards stage needs the trainer on every GPU of a node "
                                   f"(actor GPUs per node {gpus}, the nodes have {nodes[0].cards_per_node})")
        first, last = pipeline_split(_extra_argv(args, env))
    for node in nodes:
        host_path(node, store)  # raises for a docker node without a mount there
    tuning = os.environ if environ is None else environ
    return WeightsPlan(model=model, revision=revision, store=store, source=str(source), stage_shards=stage,
                       pipeline_parallel=pp, pp_first=first, pp_last=last,
                       env=source_env(str(source), env, environ),
                       workers=int(tuning.get("YETO_WEIGHTS_WORKERS") or weights_sync.DEFAULT_WORKERS),
                       chunk_mb=int(tuning.get("YETO_WEIGHTS_CHUNK_MB") or weights_sync.DEFAULT_CHUNK_MB),
                       timeout_s=float(tuning.get("YETO_WEIGHTS_TIMEOUT_S") or SYNC_TIMEOUT_S))


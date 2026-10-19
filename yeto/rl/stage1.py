"""Stage 1 of the S25 goal as one plan: GLM5.3-Flash bf16 full-parameter RL on
PPU nodes (trainer + SGLang sampler on PPU, head on this server, OSS store).

``yeto-stage1 plan PLAN.yaml`` expands the plan file and its node inventory
(``--rl-node-inventory`` format, :mod:`.node_inventory`) into every command,
environment and path the run uses, checks that they agree with each other, and
prints the result. It never opens an SSH connection.

``yeto-stage1 run PLAN.yaml --confirm-machines-free`` runs the same steps in
order over SSH. It refuses while the plan has a check error or an open gap
(an entry point, image or path that does not exist yet), so it cannot start a
half-wired run on machines other people use.

What the plan expands (all derived from one place, so the names agree):

* nodes and roles: trainer nodes, sampler nodes, card counts, rendezvous
  addresses;
* trainer, first backend Miles/Megatron: the parallel layout (TP x PP x EP,
  the pipeline layer split), the model arguments of the yeto glm5_next recipe
  (:mod:`yeto.rl.adapters.miles.models.glm5_next`), the parallel and optimizer
  arguments, the yeto Miles island argv, and the island as an SSH trainer
  island (:class:`~yeto.launch.ssh_trainer_island.SshTrainerSpec`: Ray on
  every node in a docker container, the island entry on node 0); second
  backend verl/FSDP2: the trainer overrides (written, not the first backend);
* sampler: the engine sampler island's spec file and command
  (``python -m yeto.rl.engine.engine_sampler``), which starts the SGLang TP8
  server of the ``sglang-server`` adapter (:mod:`yeto.rl.adapters.sampler_engines`);
* re-quantizer: the :mod:`yeto.rl.quant.release` argv (N8.5 MXFP4 RTN), with
  the release directory named after the scheme;
* store: the run directory on OSS, the OSS staging paths of the weights, and
  how the head reaches the store (``store_access``: through a relay node,
  :mod:`yeto.cloud.oss_relay`, or directly);
* steps: the ordered per-node scripts of the run (preflight, staging, sampler,
  trainer).

Check errors are contradictions inside the plan (card counts, scheme names,
revisions, sampling values). Gaps are things the plan needs that the code base
or the machines do not provide yet; they are expected until the stage-1
wiring is finished and are listed so the runbook can close them one by one.

Nothing here was run against a PPU node.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import importlib.util
import ipaddress
import json
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from yeto.launch.node_inventory import InventoryError, Node, load_inventory
from yeto.rl.adapters.miles.models import glm5_next
from yeto.rl.adapters.sampler_engines.from_spec import FACTORY as ENGINE_FACTORY
from yeto.rl.engine.telemetry import DEVICE_FAMILY_ENV
from yeto.rl.quant.schemes import release_dir_for

PLAN_SCHEMA = "yeto.stage1/v1"
BACKENDS = ("miles", "verl")
NEUTRAL_PENALTIES = {"repetition_penalty": 1.0, "frequency_penalty": 0.0, "presence_penalty": 0.0}
RAY_PORT = 6379
HOST_USABLE_FRACTION = 0.85  # as yeto.rl.fullparam.memory.TrainPlan.host_usable_fraction
HEAD_ACCESS = ("relay", "direct")  # how the head reaches the store (store_access.head)
RAY_JOIN_TIMEOUT_S = 600


class Stage1Error(ValueError):
    """The plan file cannot be read or expanded."""


# --------------------------------------------------------------------- plan file

def _section(doc: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = doc.get(key)
    if not isinstance(value, dict):
        raise Stage1Error(f"plan: section {key!r} must be a mapping, got {type(value).__name__}")
    return value


def _get(section: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in section:
        raise Stage1Error(f"plan: {where}.{key} is required")
    return section[key]


@dataclass(frozen=True)
class Stage1Plan:
    """The plan file, read and with its inventory loaded."""

    path: Path
    doc: Mapping[str, Any]
    nodes: tuple[Node, ...]

    @property
    def run_id(self) -> str:
        return str(_get(self.doc, "run_id", "plan"))

    def section(self, key: str) -> dict[str, Any]:
        return _section(self.doc, key)

    def nodes_of(self, role: str) -> tuple[Node, ...]:
        by_name = {n.name: n for n in self.nodes}
        names = _section(self.doc, "roles").get(role) or []
        missing = [n for n in names if n not in by_name]
        if missing:
            raise Stage1Error(f"plan: roles.{role} names {missing}, which the inventory does not list")
        return tuple(by_name[n] for n in names)


def load_plan(path: str | Path) -> Stage1Plan:
    import yaml  # noqa: PLC0415 - optional at import time, like node_inventory

    file = Path(path)
    try:
        doc = yaml.safe_load(file.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise Stage1Error(f"cannot read the plan {file}: {exc}") from None
    if not isinstance(doc, dict):
        raise Stage1Error(f"{file}: the plan must be a mapping")
    inventory = file.parent / str(_get(doc, "inventory", "plan"))
    try:
        nodes = tuple(load_inventory(inventory))
    except InventoryError as exc:
        raise Stage1Error(str(exc)) from None
    return Stage1Plan(path=file, doc=doc, nodes=nodes)


# ----------------------------------------------------------------- expansion

@dataclass
class Step:
    """One step of the run: a script for one node (``node`` None = the head)."""

    name: str
    node: str | None
    script: str
    purpose: str
    service: bool = False  # runs for the whole run (started, not waited for)

    def record(self) -> dict[str, Any]:
        return {"name": self.name, "node": self.node, "purpose": self.purpose, "script": self.script,
                "service": self.service}


@dataclass
class Expanded:
    plan: Stage1Plan
    trainer: dict[str, Any] = field(default_factory=dict)
    sampler: dict[str, Any] = field(default_factory=dict)
    requant: dict[str, Any] = field(default_factory=dict)
    store: dict[str, Any] = field(default_factory=dict)
    trainer_spec: Any = None  # the SshTrainerSpec the trainer steps come from
    steps: list[Step] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)

    def record(self) -> dict[str, Any]:
        return {
            "schema": PLAN_SCHEMA,
            "run_id": self.plan.run_id,
            "plan": str(self.plan.path),
            "nodes": [n.record() for n in self.plan.nodes],
            "roles": {r: [n.name for n in self.plan.nodes_of(r)] for r in ("trainer", "sampler")},
            "store": self.store,
            "trainer": self.trainer,
            "sampler": self.sampler,
            "requant": self.requant,
            "steps": [s.record() for s in self.steps],
            "errors": list(self.errors),
            "gaps": list(self.gaps),
        }


def run_store(plan: Stage1Plan) -> str:
    """The run directory on the store: ``<store>/<run_id>``."""
    return f"{str(_get(plan.doc, 'store', 'plan')).rstrip('/')}/{plan.run_id}"


def megatron_layout(miles: Mapping[str, Any], trainer_cards: int, cards_per_node: int) -> dict[str, int]:
    """Data-parallel sizes of a TP x PP x EP layout (Megatron rules); raises on a
    layout Megatron would refuse."""
    tp, pp, cp, ep, etp = (int(miles[k]) for k in ("tp", "pp", "cp", "ep", "etp"))
    if trainer_cards % (tp * pp * cp):
        raise Stage1Error(f"trainer cards {trainer_cards} are not a multiple of TP*PP*CP = {tp * pp * cp}")
    dp = trainer_cards // (tp * pp * cp)
    if (dp * tp * cp) % (ep * etp):
        raise Stage1Error(f"EP*ETP = {ep * etp} must divide DP*TP*CP = {dp * tp * cp}")
    if ep * etp > cards_per_node or tp > cards_per_node:
        raise Stage1Error(f"EP*ETP = {ep * etp} and TP = {tp} must stay inside one node "
                          f"({cards_per_node} cards; A-TRN: EP and TP do not cross RoCE)")
    return {"tp": tp, "pp": pp, "cp": cp, "ep": ep, "etp": etp, "dp": dp,
            "expert_dp": dp * tp * cp // (ep * etp)}


def megatron_static_gib(model: Mapping[str, Any], layout: Mapping[str, int]) -> dict[str, float]:
    """Per-card static memory of the layout, estimate (A-TRN §2.1 method):
    bf16 weights + fp32 main grads = 6 B/param on the card; Adam + fp32 master
    = 12 B/param, sharded over the data-parallel group, on the host when
    offloaded. Activations are not included."""
    experts = float(model["expert_params_b"]) * 1e9
    dense = float(model["params_b"]) * 1e9 - experts
    per_card = experts / (layout["pp"] * layout["ep"] * layout["etp"]) + dense / (layout["pp"] * layout["tp"])
    optimizer_card = (experts / (layout["pp"] * layout["ep"] * layout["etp"] * layout["expert_dp"])
                      + dense / (layout["pp"] * layout["tp"] * layout["dp"])) * 12
    gib = 2.0 ** 30
    return {"params_per_card_b": round(per_card / 1e9, 2), "weights_grads_gib": round(per_card * 6 / gib, 1),
            "optimizer_gib_per_card": round(optimizer_card / gib, 1)}


def megatron_args(model: Mapping[str, Any], miles: Mapping[str, Any], layout: Mapping[str, int],
                  sampling: Mapping[str, Any]) -> list[str]:
    """The parallel, recompute and optimizer arguments of the run, on top of the
    model arguments of the glm5_next recipe (``glm5_next.model_argv``)."""
    split = [int(x) for x in miles["pp_layers"]]
    args = ["--tensor-model-parallel-size", str(layout["tp"]),
            "--pipeline-model-parallel-size", str(layout["pp"]),
            "--context-parallel-size", str(layout["cp"]),
            "--expert-model-parallel-size", str(layout["ep"]),
            "--expert-tensor-parallel-size", str(layout["etp"])]
    if miles.get("sequence_parallel"):
        args.append("--sequence-parallel")
    if layout["pp"] > 1:
        args += ["--decoder-first-pipeline-num-layers", str(split[0]),
                 "--decoder-last-pipeline-num-layers", str(split[-1])]
    args += ["--recompute-granularity", str(miles["recompute"]), "--recompute-method", "uniform",
             "--recompute-num-layers", "1", "--micro-batch-size", "1",
             "--max-tokens-per-gpu", str(int(miles["max_tokens_per_gpu"])),
             "--optimizer", "adam", "--lr", repr(float(miles["lr"])), "--lr-decay-style", "constant",
             "--use-distributed-optimizer", "--bf16",
             "--rollout-max-response-len", str(int(sampling["max_tokens"]))]
    fraction = float(miles.get("optimizer_offload_fraction") or 0.0)
    if fraction > 0:
        args += ["--optimizer-cpu-offload", "--optimizer-offload-fraction", repr(fraction),
                 "--overlap-cpu-optimizer-d2h-h2d"]
    return args


def verl_overrides(verl: Mapping[str, Any], trainer_nodes: Sequence[Node], task: Mapping[str, Any],
                   sampling: Mapping[str, Any], model_dir: str) -> list[str]:
    """verl trainer overrides of the FSDP2 backend (rollout runs on the yeto
    sampler island, not inside verl)."""
    actor = "actor_rollout_ref.actor"
    return [f"actor_rollout_ref.model.path={model_dir}",
            f"{actor}.strategy={verl['strategy']}",
            f"{actor}.fsdp_config.fsdp_size={int(verl['fsdp_size'])}",
            f"{actor}.fsdp_config.param_offload={bool(verl['param_offload'])}",
            f"{actor}.fsdp_config.optimizer_offload={bool(verl['optimizer_offload'])}",
            f"{actor}.ulysses_sequence_parallel_size={int(verl['ulysses'])}",
            f"{actor}.optim.lr={float(verl['lr'])}",
            f"actor_rollout_ref.rollout.n={int(task['n_samples_per_prompt'])}",
            f"data.train_batch_size={int(task['rollout_batch_size'])}",
            f"data.max_prompt_length={int(sampling['max_prompt_len'])}",
            f"data.max_response_length={int(sampling['max_tokens'])}",
            f"trainer.nnodes={len(trainer_nodes)}",
            f"trainer.n_gpus_per_node={trainer_nodes[0].cards_per_node}",
            "trainer.device=cuda"]  # PPU runs the CUDA compatibility layer (rl-ssh-node-pool D2)


def _sampler_engine(plan: Stage1Plan, node: Node):
    from yeto.rl.adapters.sampler_engines.server import ServerConfig  # noqa: PLC0415
    from yeto.rl.adapters.sampler_engines.sglang_server import SGLangOptions, SGLangServerEngine  # noqa: PLC0415

    s, q = plan.section("sampler"), plan.section("quant")
    extra = ["--moe-runner-backend", str(s["moe_runner_backend"])]
    if s.get("hierarchical_cache"):
        extra.append("--enable-hierarchical-cache")
    config = ServerConfig(
        base_dir=Path(plan.section("model")["sampler_initial"]["local_dir"]),
        card_model=node.card, device_family=node.device_family,
        engine_version=str(s["engine_version"]), precision=str(s["precision"]),
        quant_scheme=str(q["scheme"]), topology=str(s["topology"]),
        max_model_len=int(s["max_model_len"]), host="0.0.0.0", port=int(s["port"]),
        extra_args=tuple(extra), expected_format=str(q["format"]),
        in_place_reload=bool(s.get("in_place_reload")))
    options = SGLangOptions(tp=int(s["tp"]), nextn=bool(s.get("nextn")),
                            renames_weight_packed=bool(s.get("renames_weight_packed")))
    return SGLangServerEngine(config, options)


def _docker(plan: Stage1Plan, node: Node, role: str, image: str | None, command: Sequence[str],
            env: Mapping[str, str]) -> str:
    images = plan.section("images")
    argv = ["docker", "run", "--rm", "--name", f"yeto-{plan.run_id}-{role}-{node.name}",
            "--network", "host", "--ipc", "host", *[str(a) for a in images.get("docker_device_args") or []]]
    for host_path, container_path in sorted(node.mounts.items()):
        argv += ["-v", f"{host_path}:{container_path}"]
    for key, value in sorted(env.items()):
        argv += ["-e", f"{key}={value}"]
    return shlex.join([*argv, image or "<trainer-image-unset>", *command])


def _module_gap(module: str, what: str, attr: str = "main") -> str | None:
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        spec = None
    if spec is None:
        return f"{what}: module {module} does not exist in this tree"
    try:
        loaded = importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 - any import failure is a gap, reported as such
        return f"{what}: module {module} does not import here ({type(exc).__name__}: {exc})"
    if not callable(getattr(loaded, attr, None)):
        return f"{what}: module {module} has no {attr}() entry point"
    return None


def _provenance_args(reward: str) -> list[str]:
    """``--reward-sha256`` / ``--source-sha256`` of this tree, as the launcher
    computes them (the code pack the nodes run must be this tree)."""
    from yeto.provenance import python_spec_sha256, source_tree_sha256  # noqa: PLC0415

    repo = Path(__file__).resolve().parents[2]
    try:
        reward_sha = python_spec_sha256(reward, base_dir=repo)
    except (OSError, ValueError) as exc:
        raise Stage1Error(f"plan: task.reward {reward!r} does not resolve in this tree: {exc}") from None
    return ["--reward-sha256", reward_sha, "--source-sha256", source_tree_sha256(repo / "yeto")]


class _ParseError(Exception):
    pass


def _check_island_argv(argv: Sequence[str]) -> str | None:
    """Parse the Miles island argv with the island's own parser."""
    from yeto.rl.adapters.miles.island_entry import build_parser  # noqa: PLC0415

    parser = build_parser()

    def fail(message: str) -> None:
        raise _ParseError(message)

    parser.error = fail  # type: ignore[method-assign] - no usage dump, no exit
    try:
        _, unknown = parser.parse_known_args(list(argv[3:]))
    except _ParseError as exc:
        return f"Miles island argv does not parse: {exc}"
    return f"Miles island argv: flags unknown to island_entry: {unknown}" if unknown else None


def expand(plan: Stage1Plan, *, hf_config: str | Path | None = None) -> Expanded:
    """Every command, environment and path of the plan, plus the check results."""
    out = Expanded(plan=plan)
    model, quant, sampling = plan.section("model"), plan.section("quant"), plan.section("sampling")
    task, trainer_cfg, sampler_cfg = plan.section("task"), plan.section("trainer"), plan.section("sampler")
    images, rendezvous = plan.section("images"), plan.section("rendezvous")
    trainers, samplers = plan.nodes_of("trainer"), plan.nodes_of("sampler")
    store = run_store(plan)
    staging = {k: model[k]["staged"] for k in ("bf16_source", "sampler_initial")}
    out.store = {"run": store, "release_dir": f"{store}/{quant['release_dir']}",
                 "bf16_release_dir": f"{store}/policy-releases", "staging": staging}

    # Trainer -------------------------------------------------------------
    backend = str(trainer_cfg["backend"])
    if backend not in BACKENDS:
        raise Stage1Error(f"plan: trainer.backend {backend!r} is not one of {BACKENDS}")
    cards = sum(n.cards_per_node for n in trainers)
    per_node = trainers[0].cards_per_node if trainers else 0
    miles = _section(trainer_cfg, "miles")
    layout = megatron_layout(miles, cards, per_node)
    common_env = {DEVICE_FAMILY_ENV: trainers[0].device_family, "YETO_RUN_ID": plan.run_id,
                  "YETO_RL_STORE": store, "HF_HUB_OFFLINE": "1"}
    trainer_env = {**common_env, "TORCHINDUCTOR_COMPILE_THREADS": "1"}
    model_dir = str(model["bf16_source"]["local_dir"])
    state_dir = state_dir_of(plan)
    impl = str(miles["transformer_impl"])
    island_argv = ["python3", "-m", "yeto.rl.adapters.miles.island_entry",
                   "--model", model_dir, "--model-revision", str(model["bf16_source"]["revision"]),
                   "--data", str(task["data"]), "--data-revision", str(task["data_revision"]),
                   "--reward-function", str(task["reward"]),
                   "--samples-per-group", str(int(task["n_samples_per_prompt"])),
                   "--rollout-max-response-len", str(int(sampling["max_tokens"])),
                   "--seq-len", str(int(sampling["max_prompt_len"]) + int(sampling["max_tokens"])),
                   "--inner-lr", repr(float(miles["lr"])), "--seed", str(int(plan.doc.get("seed", 0))),
                   "--parameter-mode", "full", "--rl-sample-role", "trainer",
                   "--rl-resume-store", store,
                   "--rl-model-recipe", glm5_next.MODEL_NAME, "--rl-transformer-impl", impl,
                   "--tensor-parallel", str(layout["tp"]), "--pipeline-parallel", str(layout["pp"]),
                   "--expert-parallel", str(layout["ep"]),
                   "--miles-root", str(miles["workdir"]),
                   "--trust-remote-code",
                   "--learner-id", "0", "--num-learners", "1", "--rl-single-island-no-sync",
                   "--global-rounds", str(int(plan.doc["rounds"])),
                   "--groups-per-round", str(int(task["rollout_batch_size"])),
                   "--over-sampling-batch-size", str(int(task["rollout_batch_size"])),
                   "--optimizer-steps", "1",
                   "--actor-num-nodes", str(len(trainers)), "--actor-num-gpus-per-node", str(per_node),
                   "--rollout-num-gpus", "0",
                   "--rl-activation-recompute", "--rl-max-tokens-per-gpu", str(int(miles["max_tokens_per_gpu"])),
                   "--completed-groups-path", f"{state_dir}/island-checkpoint.pt",
                   "--event-tape", f"{state_dir}/events.jsonl", "--audit-dir", f"{state_dir}/audit",
                   *_provenance_args(str(task["reward"]))]
    try:
        model_args = glm5_next.model_argv(num_layers=int(model["num_layers"]), transformer_impl=impl)
    except ValueError as error:
        raise Stage1Error(f"plan: trainer.miles.transformer_impl: {error}") from None
    out.trainer = {
        "backend": backend, "nodes": [n.name for n in trainers], "cards": cards,
        "image": images.get("trainer"), "env": trainer_env,
        "miles": {"commit": miles["commit"], "megatron": miles["megatron"], "mbridge": miles["mbridge"],
                  "layout": layout, "static_memory_estimate": megatron_static_gib(model, layout),
                  "model_args": model_args,
                  "megatron_args": megatron_args(model, miles, layout, sampling),
                  "drop_model_args": list(glm5_next.TE_ONLY_ARGV) if impl == glm5_next.LOCAL else [],
                  "island_argv": island_argv},
        "verl": {"commit": _section(trainer_cfg, "verl").get("commit"),
                 "overrides": verl_overrides(_section(trainer_cfg, "verl"), trainers, task, sampling, model_dir)},
    }
    if hf_config is not None:
        out.trainer["verl"]["memory_estimate"] = _verl_memory(trainers, plan, sampling, hf_config)

    # Sampler -------------------------------------------------------------
    sampler_runs = []
    for node in samplers:
        engine = _sampler_engine(plan, node)
        env = {**common_env, DEVICE_FAMILY_ENV: node.device_family, **engine.env()}
        sampler_runs.append({"node": node.name, "engine_argv": engine.argv(engine.config.base_dir), "env": env,
                             "spec_path": f"{state_dir}/sampler-{node.name}.json",
                             "spec": sampler_spec(plan, node, engine),
                             "argv": ["python3", "-m", "yeto.rl.engine.engine_sampler",
                                      "--spec", f"{state_dir}/sampler-{node.name}.json"],
                             "url": f"http://{rendezvous.get(node.name)}:{engine.config.port}"})
    out.sampler = {"engine": sampler_cfg["engine"], "engine_version": sampler_cfg["engine_version"],
                   "image": images.get("sampler"), "topology": sampler_cfg["topology"],
                   "precision": sampler_cfg["precision"], "quant_scheme": quant["scheme"],
                   "initial": {k: model["sampler_initial"][k] for k in ("repo", "revision", "local_dir")},
                   "sampling": dict(sampling), "engines": sampler_runs}

    # Re-quantizer (node 0 of the trainer, a side process of the island) ---
    from yeto.launch.ssh_trainer_island import requant_argv  # noqa: PLC0415

    # The release directory follows the template's scheme (quant.schemes); the
    # plan's release_dir is checked against it, not passed to the re-quantizer.
    argv = requant_argv(store, str(model["sampler_initial"]["local_dir"]), python="python3")
    out.requant = {"node": trainers[0].name if trainers else None, "scheme": quant["scheme"],
                   "format": quant["format"], "argv": argv, "loop_argv": [*argv, "--loop"]}

    out.store["head_access"] = head_store_access(plan)
    out.trainer_spec = trainer_island_spec(plan, out, trainers) if backend == "miles" and trainers else None
    out.steps = _steps(plan, out, trainers, samplers)
    out.errors = check(plan, out)
    out.gaps = gaps(plan, out)
    return out


def state_dir_of(plan: Stage1Plan) -> str:
    """The run's state directory on every node (``/data`` is on every node)."""
    return f"/data/yeto/{plan.run_id}"


def sampler_spec(plan: Stage1Plan, node: Node, engine: Any) -> dict[str, Any]:
    """The engine sampler island's spec file (``engine_sampler_entry``)."""
    sampling, task, quant = plan.section("sampling"), plan.section("task"), plan.section("quant")
    config, options = engine.config, engine.options
    server = {f.name: getattr(config, f.name) for f in dataclasses.fields(config)}
    server["base_dir"] = str(config.base_dir)
    server["extra_args"] = list(config.extra_args)
    server["extra_env"] = dict(config.extra_env)
    state_dir = state_dir_of(plan)
    return {
        "sampler_id": f"{plan.run_id}-{node.name}",
        "engine": {"factory": ENGINE_FACTORY, "adapter": f"{plan.section('sampler')['engine']}-server",
                   "server": server,
                   "options": {k: list(v) if isinstance(v, tuple) else v
                               for k, v in dataclasses.asdict(options).items()}},
        "sampling": {"max_tokens": int(sampling["max_tokens"]), "temperature": float(sampling["temperature"]),
                     "top_p": float(sampling["top_p"]), "top_k": int(sampling["top_k"]),
                     "n": int(task["n_samples_per_prompt"]), "seed": int(plan.doc.get("seed", 0)),
                     **{k: float(sampling.get(k, v)) for k, v in NEUTRAL_PENALTIES.items()}},
        "num_rollouts": int(plan.doc["rounds"]),
        # decision T-GAPS 10-19: one round ahead of the trainer at most (stage 1 runs on-policy)
        "max_backlog_groups": int(task["rollout_batch_size"]),
        "max_prompt_tokens": int(sampling["max_prompt_len"]),
        "base_version": 0, "base_dir": str(config.base_dir),
        "store": run_store(plan), "release_dir": str(quant["release_dir"]),
        "prompts": {"path": str(_get(task, "prompts_jsonl", "task")), "input_key": str(_get(task, "input_key", "task")),
                    "label_key": str(_get(task, "label_key", "task")),
                    "groups_per_round": int(task["rollout_batch_size"]),
                    "format": str(task.get("prompt_format", "chat")), "trust_remote_code": True},
        "reward": str(task["reward"]), "reward_kind": "miles",
        "checks": {"expected_expert_matrices": int(quant["expected_expert_matrices"])},
        "event_tape": f"{state_dir}/sampler-{node.name}/events.jsonl",
    }


def head_store_access(plan: Stage1Plan) -> dict[str, Any]:
    """How the head on this server reaches the run store (``store_access``):
    ``relay`` = ``ossutil`` on a relay node over SSH (:mod:`yeto.cloud.oss_relay`;
    main-agent ruling 10-19), ``direct`` = the OSS endpoint from this server."""
    from yeto.cloud.oss_relay import RELAY_ENV, Relay, relay_shell  # noqa: PLC0415

    access = plan.doc.get("store_access") or {"head": "direct"}
    mode = str(access.get("head", "direct"))
    if mode not in HEAD_ACCESS:
        raise Stage1Error(f"plan: store_access.head {mode!r} is not one of {HEAD_ACCESS}")
    if mode == "direct":
        return {"mode": mode, "env": {}}
    name = str(access.get("relay") or "")
    node = next((n for n in plan.nodes if n.name == name), None)
    if node is None:
        raise Stage1Error(f"plan: store_access.relay {name!r} is not an inventory node")
    relay = Relay(node.host, str(access.get("ossutil") or "ossutil"))
    return {"mode": mode, "relay": node.name, "relay_host": node.host, "env": {RELAY_ENV: node.host},
            "check": relay_shell(relay, f"{relay.ossutil} ls -s {shlex.quote(run_store(plan) + '/')} | tail -n 3")}


def trainer_island_spec(plan: Stage1Plan, out: Expanded, trainers: Sequence[Node]) -> Any:
    """The Miles trainer island as an SSH trainer island (``SshTrainerSpec``):
    Ray on every trainer node, the island entry on node 0 after every node
    joined, the re-quantizer as node 0's side process; one docker container per
    node. Unset image or device flags show as placeholders (gaps N0.2/N0.2b)."""
    from yeto.launch.ssh_trainer_island import SshTrainerSpec  # noqa: PLC0415

    images, rendezvous = plan.section("images"), plan.section("rendezvous")
    nodes = tuple(n if n.address else dataclasses.replace(n, address=str(rendezvous.get(n.name)))
                  for n in trainers)
    island = shlex.join(out.trainer["miles"]["island_argv"])
    join = (f"python3 -c {shlex.quote(_RAY_JOIN)} \"$SKYPILOT_NUM_NODES\" {RAY_JOIN_TIMEOUT_S}")
    script = (
        "set -eu\n"
        "head=$(echo \"$SKYPILOT_NODE_IPS\" | sed -n 1p)\n"
        "me=$(echo \"$SKYPILOT_NODE_IPS\" | sed -n \"$((SKYPILOT_NODE_RANK + 1))p\")\n"
        "if [ \"$SKYPILOT_NODE_RANK\" = 0 ]; then\n"
        f"  ray start --head --port={RAY_PORT} --node-ip-address=\"$me\" --num-gpus=\"$SKYPILOT_NUM_GPUS_PER_NODE\"\n"
        f"  {join}\n"
        f"  exec {island}\n"
        "fi\n"
        f"exec ray start --address=\"$head:{RAY_PORT}\" --node-ip-address=\"$me\" "
        "--num-gpus=\"$SKYPILOT_NUM_GPUS_PER_NODE\" --block\n")
    try:
        return SshTrainerSpec(
            nodes=nodes, run_id=plan.run_id, workdir=str(_section(plan.section("trainer"), "miles")["workdir"]),
            run_script=script, env=out.trainer["env"], requant=tuple(out.requant["argv"]),
            image=images.get("trainer") or "<trainer-image-unset>",
            docker_args=tuple(images.get("docker_device_args") or ("<docker-device-args-unset>",)))
    except Exception as error:  # SshIslandError: reported as a check error, not a crash
        raise Stage1Error(f"trainer island: {error}") from None


# Rank 0 waits until every trainer node joined Ray (argv: node count, timeout seconds).
_RAY_JOIN = ("import sys, time, ray\n"
             "ray.init(address=\"auto\")\n"
             "want, deadline = int(sys.argv[1]), time.time() + float(sys.argv[2])\n"
             "alive = lambda: sum(n[\"Alive\"] for n in ray.nodes())\n"
             "while alive() < want:\n"
             "    if time.time() > deadline:\n"
             "        sys.exit(f\"only {alive()} of {want} nodes joined Ray\")\n"
             "    time.sleep(5)\n")


def _verl_memory(trainers: Sequence[Node], plan: Stage1Plan, sampling: Mapping[str, Any],
                 hf_config: str | Path) -> dict[str, Any]:
    from yeto.rl.fullparam.memory import estimate, plan_for_nodes, shape_from_hf_config  # noqa: PLC0415

    verl = _section(plan.section("trainer"), "verl")
    shape = shape_from_hf_config(json.loads(Path(hf_config).read_text()))
    # yeto's FSDP2 config has no optimizer-only host offload (N7.1); the optimizer
    # offload of the plan is estimated as the full CPU offload policy.
    offload = "full" if verl["param_offload"] or verl["optimizer_offload"] else "none"
    train_plan = plan_for_nodes(list(trainers), seq_len=int(sampling["max_prompt_len"]) + int(sampling["max_tokens"]),
                                fsdp_size=int(verl["fsdp_size"]), ulysses=int(verl["ulysses"]), offload=offload)
    return estimate(shape, train_plan).summary()


def _steps(plan: Stage1Plan, out: Expanded, trainers: Sequence[Node], samplers: Sequence[Node]) -> list[Step]:
    from yeto.hw.card_tools import preflight_commands  # noqa: PLC0415
    from yeto.launch.ssh_trainer_island import container_name, remote_paths, start_script  # noqa: PLC0415

    steps: list[Step] = []
    for node in plan.nodes:
        tools = "\n".join(shlex.join(c) for c in preflight_commands(node.device_family))
        steps.append(Step(f"preflight-{node.name}", node.name,
                          f"set -eu\n{tools}\nip -4 -o addr show {node.interface or 'eth0'}\n"
                          "df -h /data\nfree -g\n",
                          "cards match the inventory; rendezvous address and /data present"))
    access = out.store["head_access"]
    if access["mode"] == "relay":
        steps.append(Step("head-store-relay", None, f"set -eu\n{access['check']}\n",
                          f"the head reads the run store through {access['relay']} (ossutil over SSH)"))
    # The re-quantizer on trainer node 0 reads the MXFP4 export as its template.
    for key, nodes in (("bf16_source", trainers), ("sampler_initial", (*samplers, *trainers[:1]))):
        src = plan.section("model")[key]
        for node in nodes:
            steps.append(Step(f"stage-{key}-{node.name}", node.name,
                              f"set -eu\nmkdir -p {shlex.quote(src['local_dir'])}\n"
                              f"ossutil cp -r -u {shlex.quote(src['staged'])} {shlex.quote(src['local_dir'])}/\n"
                              f"test -f {shlex.quote(src['local_dir'])}/model.safetensors.index.json\n",
                              f"{key} weights on the node's /data (revision {src['revision']})"))
    for run in out.sampler["engines"]:
        node = next(n for n in samplers if n.name == run["node"])
        spec_json = json.dumps(run["spec"], indent=1, sort_keys=True)
        steps.append(Step(f"sampler-spec-{node.name}", node.name,
                          f"set -eu\nmkdir -p {shlex.quote(str(Path(run['spec_path']).parent))}\n"
                          f"cat > {shlex.quote(run['spec_path'])} <<'YETO_SAMPLER_SPEC'\n{spec_json}\n"
                          "YETO_SAMPLER_SPEC\n",
                          "the engine sampler island's spec file"))
        steps.append(Step(f"sampler-{node.name}", node.name,
                          _docker(plan, node, "sampler", out.sampler["image"], run["argv"], run["env"]),
                          "engine sampler island: starts SGLang TP8 on the round-0 MXFP4 weights, runs the "
                          "post-load checks, writes groups to the store, follows the MXFP4 releases",
                          service=True))
    spec = out.trainer_spec
    if spec is not None:
        paths = remote_paths(plan.run_id, "trainer", 0)
        for rank, node in enumerate(spec.nodes):
            steps.append(Step(f"trainer-{node.name}", node.name,
                              start_script(spec, rank, {}, paths,
                                           container=container_name(plan.run_id, "trainer", 0, rank)),
                              ("Ray head, then the yeto Miles island entry (sample role trainer) and the "
                               "MXFP4 re-quantizer side process" if rank == 0 else "Ray worker of the trainer island"),
                              service=True))
    return steps


# ------------------------------------------------------------------ checks

def check(plan: Stage1Plan, out: Expanded) -> list[str]:
    """Contradictions inside the expanded plan. Empty list = consistent."""
    errors: list[str] = []
    model, quant, sampling = plan.section("model"), plan.section("quant"), plan.section("sampling")
    sampler_cfg, trainer_cfg = plan.section("sampler"), plan.section("trainer")
    trainers, samplers = plan.nodes_of("trainer"), plan.nodes_of("sampler")
    miles = _section(trainer_cfg, "miles")

    # Nodes and roles.
    t_names, s_names = {n.name for n in trainers}, {n.name for n in samplers}
    if t_names & s_names:
        errors.append(f"roles: nodes {sorted(t_names & s_names)} are both trainer and sampler")
    if unused := sorted({n.name for n in plan.nodes} - t_names - s_names):
        errors.append(f"roles: inventory nodes {unused} have no role")
    for node in (*trainers, *samplers):
        role = "trainer" if node.name in t_names else "sampler"
        if node.roles and role not in node.roles:
            errors.append(f"{node.name}: the plan gives role {role}, the inventory says {list(node.roles)}")
        ip = plan.section("rendezvous").get(node.name)
        try:
            ipaddress.ip_address(str(ip))
        except ValueError:
            errors.append(f"{node.name}: rendezvous address {ip!r} is not an IP address")
        if node.address is not None and node.address != str(ip):
            errors.append(f"{node.name}: rendezvous address {ip} differs from the inventory address {node.address}")
    if len({(n.device_family, n.card, n.cards_per_node) for n in trainers}) > 1:
        errors.append("trainer nodes differ in device family, card or card count")

    # Trainer layout.
    layout = out.trainer["miles"]["layout"]
    split = [int(x) for x in miles["pp_layers"]]
    if len(split) != layout["pp"]:
        errors.append(f"miles.pp_layers has {len(split)} stages, PP = {layout['pp']}")
    if sum(split) != int(model["num_layers"]):
        errors.append(f"miles.pp_layers sum to {sum(split)}, the model has {model['num_layers']} layers")
    if layout["pp"] > 2 and len(set(split[1:-1])) > 1:
        errors.append(f"miles.pp_layers middle stages differ {split[1:-1]}; Megatron splits them evenly")
    if layout["tp"] * layout["pp"] * layout["dp"] * layout["cp"] != out.trainer["cards"]:
        errors.append("TP*PP*DP*CP does not equal the trainer card count")
    if miles["transformer_impl"] not in ("transformer_engine", "local"):
        errors.append(f"miles.transformer_impl {miles['transformer_impl']!r} is not transformer_engine or local")
    flat = " ".join(out.trainer["miles"]["megatron_args"])
    if "fp8" in flat or "fp4" in flat:
        errors.append("trainer arguments enable FP8/FP4; training is bf16 only (user 10-19, report #1)")
    verl = _section(trainer_cfg, "verl")
    if int(verl["fsdp_size"]) not in (0, out.trainer["cards"]) and out.trainer["cards"] % int(verl["fsdp_size"]):
        errors.append(f"verl.fsdp_size {verl['fsdp_size']} does not divide {out.trainer['cards']} cards")

    # Sampler.
    for node in samplers:
        if int(sampler_cfg["tp"]) != node.cards_per_node:
            errors.append(f"{node.name}: sampler TP {sampler_cfg['tp']} != {node.cards_per_node} cards "
                          "(stage 1: one TP engine per node)")
    if f"tp{int(sampler_cfg['tp'])}" != sampler_cfg["topology"]:
        errors.append(f"sampler topology {sampler_cfg['topology']} does not match TP {sampler_cfg['tp']}")
    image = str(plan.section("images").get("sampler") or "")
    if not image.endswith(f":v{sampler_cfg['engine_version']}"):
        errors.append(f"sampler image {image!r} does not carry engine version v{sampler_cfg['engine_version']}")
    if int(sampler_cfg["max_model_len"]) < int(sampling["max_prompt_len"]) + int(sampling["max_tokens"]):
        errors.append("sampler max_model_len is shorter than max_prompt_len + max_tokens")

    # Quantization scheme, names and revisions.
    errors += _check_scheme(quant, sampler_cfg)
    initial = model["sampler_initial"]
    if str(initial["revision"]) not in str(initial["local_dir"]) or str(initial["revision"]) not in str(initial["staged"]):
        errors.append(f"sampler_initial revision {initial['revision']} is not in its local_dir/staged names")
    bf16 = model["bf16_source"]
    short = str(bf16["revision"])[:7]
    if short not in str(bf16["local_dir"]) or short not in str(bf16["staged"]) or str(bf16["revision"]) not in str(bf16["origin"]):
        errors.append(f"bf16_source revision {bf16['revision']} is not in its origin/local_dir/staged names")
    requant = out.requant["argv"]
    if requant[requant.index("--store") + 1] != out.store["run"]:
        errors.append("re-quantizer --store is not the run store")

    # Store.
    store = out.store["run"]
    if not store.startswith("oss://"):
        errors.append(f"store {store!r} is not an oss:// URI (stage 1: Alibaba OSS)")
    if not store.endswith("/" + plan.run_id):
        errors.append("store run directory does not end with the run id")
    bucket = store.split("/")[2] if store.startswith("oss://") else None
    for key, uri in out.store["staging"].items():
        if bucket and not str(uri).startswith(f"oss://{bucket}/"):
            errors.append(f"staging path of {key} is not in the run's bucket {bucket}")

    # Sampling: one value for every source, neutral penalties.
    for key, neutral in NEUTRAL_PENALTIES.items():
        if float(sampling.get(key, neutral)) != neutral:
            errors.append(f"sampling.{key} = {sampling[key]}; rollouts use the neutral value {neutral} (A-INF §5)")
    island = out.trainer["miles"]["island_argv"]
    if island[island.index("--rollout-max-response-len") + 1] != str(int(sampling["max_tokens"])):
        errors.append("trainer --rollout-max-response-len differs from sampling.max_tokens")
    margs = out.trainer["miles"]["megatron_args"]
    if margs[margs.index("--rollout-max-response-len") + 1] != str(int(sampling["max_tokens"])):
        errors.append("Megatron --rollout-max-response-len differs from sampling.max_tokens")
    verl_max = [o for o in out.trainer["verl"]["overrides"] if o.startswith("data.max_response_length=")]
    if verl_max != [f"data.max_response_length={int(sampling['max_tokens'])}"]:
        errors.append("verl data.max_response_length differs from sampling.max_tokens")

    # Static memory estimate must leave room for activations (A-TRN §2.3).
    mem = out.trainer["miles"]["static_memory_estimate"]
    hbm = trainers[0].hbm_gib if trainers else 0
    on_card = mem["weights_grads_gib"] + (0 if float(miles.get("optimizer_offload_fraction") or 0) >= 1
                                          else mem["optimizer_gib_per_card"])
    if on_card > 0.75 * hbm:
        errors.append(f"static memory estimate {on_card:.1f} GiB/card leaves under 25 % of {hbm:g} GiB "
                      "for activations")
    fraction = float(miles.get("optimizer_offload_fraction") or 0)
    if trainers and fraction > 0:
        host = mem["optimizer_gib_per_card"] * fraction * trainers[0].cards_per_node
        if host > HOST_USABLE_FRACTION * trainers[0].host_memory_gib:
            errors.append(f"offloaded optimizer estimate {host:.0f} GiB/node exceeds "
                          f"{HOST_USABLE_FRACTION:.0%} of {trainers[0].host_memory_gib:g} GiB host memory")
    return errors


def _check_scheme(quant: Mapping[str, Any], sampler_cfg: Mapping[str, Any]) -> list[str]:
    from yeto.rl.quant.requant import SCHEMES  # noqa: PLC0415

    errors = []
    scheme = SCHEMES.get(str(quant["format"]))
    if scheme is None:
        return [f"quant.format {quant['format']!r} is not a re-quantizer scheme (known: {sorted(SCHEMES)})"]
    if scheme.name != quant["scheme"]:
        errors.append(f"quant.format {quant['format']} is scheme {scheme.name}, the plan says {quant['scheme']}")
    if quant["release_dir"] != release_dir_for(scheme.name):
        errors.append(f"quant.release_dir {quant['release_dir']!r} is not {release_dir_for(scheme.name)!r} "
                      "(the re-quantizer writes to its scheme's directory)")
    if not scheme.name.startswith(str(sampler_cfg["precision"])):
        errors.append(f"sampler precision {sampler_cfg['precision']} does not match scheme {scheme.name}")
    if scheme.data_suffix.endswith("weight_packed") and not sampler_cfg.get("renames_weight_packed"):
        errors.append(f"scheme {scheme.name} stores *.weight_packed; the sglang-server adapter refuses it "
                      "unless sampler.renames_weight_packed (A-INF item 1)")
    return errors


def _miles_recipes() -> list[str]:
    from yeto.rl.adapters.miles.island_entry import build_parser  # noqa: PLC0415

    for action in build_parser()._actions:  # noqa: SLF001 - argparse has no public lookup
        if "--rl-model-recipe" in action.option_strings:
            return sorted(action.choices or ())
    return []


def gaps(plan: Stage1Plan, out: Expanded) -> list[str]:
    """What the run needs that does not exist yet (each one blocks ``run``)."""
    found: list[str] = []
    if not plan.section("images").get("trainer"):
        found.append("images.trainer is unset: the PPU trainer image is task N0.2")
    if not plan.section("images").get("docker_device_args"):
        found.append("images.docker_device_args is empty: the PPU device flags of docker run are N0.2b")
    for module, what in (("yeto.rl.engine.engine_sampler", "sampler island entry (N2 wiring)"),):
        if gap := _module_gap(module, what):
            found.append(gap)
    if out.trainer["backend"] == "miles":
        if gap := _check_island_argv(out.trainer["miles"]["island_argv"]):
            found.append(gap)
        recipes = _miles_recipes()
        if glm5_next.MODEL_NAME not in recipes:
            found.append(f"Miles island: no {glm5_next.MODEL_NAME} model recipe "
                         f"(--rl-model-recipe choices: {recipes})")
        if gap := _full_parameter_layout_gap(plan, out):
            found.append(gap)
    else:
        if not out.trainer["verl"]["commit"]:
            found.append("trainer.verl.commit is unset (verl pin)")
        found.append("verl backend: no SSH launch path for the verl trainer island and no cut save path (N7.6 verl)")
    if out.store["run"].startswith("oss://") and out.store["head_access"]["mode"] == "direct":
        found.append("head on this server cannot reach the intranet-only OSS bucket (PublicEndpointForbidden): "
                     "set store_access.head: relay (through ppu1) or give a public endpoint/CNAME (runbook N3.5)")
    return found


def _full_parameter_layout_gap(plan: Stage1Plan, out: Expanded) -> str | None:
    """The Miles island's own full-parameter layout rule (``run_config``)."""
    from yeto.rl.engine.run_config import full_parameter_layout_problem  # noqa: PLC0415

    argv = out.trainer["miles"]["island_argv"]
    problem = full_parameter_layout_problem(
        data_parallel=out.trainer["miles"]["layout"]["dp"],
        actor_num_nodes=int(argv[argv.index("--actor-num-nodes") + 1]),
        rollout_gpus=int(argv[argv.index("--rollout-num-gpus") + 1]),
        rollout_gpus_per_engine=1)  # the island's default; it gives no engine
    if problem is None:
        return None
    return (f"Miles island: the full-parameter mode refuses this layout ({problem}; run_config Milestone 1): "
            f"{len(plan.nodes_of('trainer'))} trainer nodes, DP {out.trainer['miles']['layout']['dp']}, "
            "samples from the store instead of colocated engines")


# --------------------------------------------------------------------- CLI

def run_steps(out: Expanded, *, ssh: Sequence[str] = ("ssh", "-o", "BatchMode=yes"),
              runner=subprocess.run, starter=None) -> int:
    """Run the steps in order; stop at the first failure. A service step is
    started (``starter``, default ``subprocess.Popen``) and not waited for; after
    the last step the run waits for the first trainer node's service, whose
    exit is the run's exit (the other services are then left to the operator:
    runbook S11 stops them)."""
    starter = starter or (lambda argv, script: subprocess.Popen(  # noqa: S603
        argv, stdin=subprocess.PIPE, text=True))
    hosts = {n.name: n.host for n in out.plan.nodes}
    services: list[tuple[Step, Any]] = []
    for step in out.steps:
        print(f"[yeto-stage1] step {step.name} on {step.node or 'head'}: {step.purpose}", flush=True)
        argv = ["bash", "-c", step.script] if step.node is None else [*ssh, hosts[step.node], "bash", "-s"]
        if step.service:
            proc = starter(argv, step.script)
            if step.node is not None and getattr(proc, "stdin", None) is not None:
                proc.stdin.write(step.script)
                proc.stdin.close()
            services.append((step, proc))
            continue
        rc = runner(argv, input=None if step.node is None else step.script, text=True).returncode
        if rc != 0:
            print(f"[yeto-stage1] step {step.name} failed (exit {rc}); later steps not run", file=sys.stderr)
            return rc
    trainer = next((p for step, p in services if step.name.startswith("trainer-")), None)
    if trainer is None:
        return 0
    rc = trainer.wait()
    print(f"[yeto-stage1] trainer island ended (exit {rc})", file=sys.stderr if rc else sys.stdout, flush=True)
    return rc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="yeto-stage1", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name, text in (("plan", "expand and check the plan; no SSH"),
                       ("run", "run the steps over SSH (refused while errors or gaps remain)")):
        p = sub.add_parser(name, help=text)
        p.add_argument("plan")
        p.add_argument("--hf-config", help="GLM config.json for the verl/FSDP2 memory estimate")
        p.add_argument("--backend", choices=BACKENDS, help="override trainer.backend")
    sub.choices["plan"].add_argument("--json", action="store_true", help="print the full expansion as JSON")
    sub.choices["run"].add_argument("--confirm-machines-free", action="store_true",
                                    help="the machine owner said the nodes are free for this run")
    args = parser.parse_args(argv)
    try:
        plan = load_plan(args.plan)
        if args.backend:
            doc = dict(plan.doc)
            doc["trainer"] = {**plan.section("trainer"), "backend": args.backend}
            plan = Stage1Plan(path=plan.path, doc=doc, nodes=plan.nodes)
        out = expand(plan, hf_config=args.hf_config)
    except Stage1Error as exc:
        print(f"[yeto-stage1] plan rejected: {exc}", file=sys.stderr)
        return 2
    if args.command == "plan":
        if args.json:
            print(json.dumps(out.record(), indent=2, sort_keys=False))
        else:
            for step in out.steps:
                print(f"# --- {step.name} ({step.node or 'head'}): {step.purpose}\n{step.script.rstrip()}\n")
        for e in out.errors:
            print(f"[yeto-stage1] ERROR {e}", file=sys.stderr)
        for g in out.gaps:
            print(f"[yeto-stage1] GAP {g}", file=sys.stderr)
        return 1 if out.errors else 0
    if out.errors or out.gaps:
        print(f"[yeto-stage1] run refused: {len(out.errors)} error(s), {len(out.gaps)} gap(s); "
              "see `yeto-stage1 plan`", file=sys.stderr)
        return 2
    if not args.confirm_machines_free:
        print("[yeto-stage1] run refused: pass --confirm-machines-free after the machine owner "
              "said the nodes are free", file=sys.stderr)
        return 2
    return run_steps(out)


if __name__ == "__main__":
    raise SystemExit(main())

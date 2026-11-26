"""Step 5 of ``yeto image adopt``: the role's smoke run of the image on one node.

**trainer** (``--role trainer``): the model, cut down until one node holds it, trains
one step, saves a checkpoint, reads it back and trains on:

1. ``prune`` (on the node, in the image): ``scripts/prune_layers.py`` keeps the first
   ``N`` decoder layers (all their experts, the embeddings and the head; the MFU 16L
   cut of 2026-11-18) and lays the result out as a Hugging Face hub snapshot under the
   node's ``scratch`` (revision = the sha1 of the cut ``config.json``). ``N`` is the
   largest of at most :data:`MAX_LAYERS` whose trainer state fits the node's cards in
   the model registry's sizing (:func:`yeto.rl.model_registry.recommend.plan_trainer`,
   with the backend and recipe ``yeto adapt`` recommends for the full model);
2. ``train`` (on the launcher): ``yeto launch`` of an ``ssh:`` island of that one node
   with the image, the registry's flags for the cut model (recipe, tuning, TP / PP /
   EP) and a train-only replay of synthetic rollouts (Miles' debug-rollout replay,
   ``YETO_TEST_SYNTH_ROLLOUT``: no sampler); ``--rl-stop-after-rounds 1`` ends the
   round with a cut (the checkpoint) in ``--smoke-store``;
3. ``resume`` (on the launcher): the same launch with ``--rl-stop-after-rounds 2``: it
   resumes from that cut (the checkpoint read) and trains the second round.

**sampler** (``--role sampler``): the pool's sampler spec on one node, started as the
node runtime starts a sampler (:func:`yeto.node.launch.container_argv`) running
:mod:`yeto.node.smoke`: start the engine, one generation, one reload (the reload of a
trainer release), one more generation.

:func:`trainer_plan` / :func:`sampler_plan` build the steps (what ``--dry-run``
prints); :func:`run_plan` runs them (an ``SshOps`` for the node steps, ``subprocess``
for the launcher steps; tests pass fakes). The pinned image, the image's interpreter
and the cut's revision are known only while adopting: the steps name them
:data:`REF`, :data:`PYTHON` and :data:`REVISION` until then.
"""

from __future__ import annotations

import base64
import copy
import dataclasses
import json
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .node_inventory import _KNOWN as INVENTORY_KEYS
from .node_inventory import Node, card_arch_of, profile_of

REF, PYTHON, REVISION = "@REF@", "@PYTHON@", "@REVISION@"
MAX_LAYERS = 16  # the MFU 16L cut
HUB_IN_CONTAINER = "/data/image-adopt-hub"  # where the trainer sees the cut (--model-store local://)
SMOKE_MODEL_ORG = "yeto-local"
REVISION_MARK = "YETO_SMOKE_REVISION"
TRAINER_BACKENDS = ("miles",)  # backends with a train-only replay (Miles' debug-rollout replay)
TRAIN_TIMEOUT_S = 3 * 3600.0
PRUNE_TIMEOUT_S = 2 * 3600.0
SAMPLER_TIMEOUT_S = 2 * 3600.0
REPO_ROOT = Path(__file__).resolve().parents[2]
PRUNE_SCRIPT = REPO_ROOT / "scripts" / "prune_layers.py"
SYNTH_ENV = "YETO_TEST_SYNTH_ROLLOUT"  # yeto.rl.adapters.miles.synth_rollout.SYNTH_ENV
REPLAY_ENV = "YETO_TEST_MILES_LOAD_DEBUG_ROLLOUT_DATA"  # yeto.rl.adapters.miles.config.TEST_DEBUG_ROLLOUT_ENV
REPLAY_PATH = "/tmp/yeto-image-adopt-rollout/{rollout_id}.pt"


@dataclass
class StepResult:
    """One adoption step's outcome (``yeto.launch.image_adopt``; the smoke returns one)."""

    name: str
    ok: bool
    detail: str
    seconds: float
    data: dict[str, Any] = field(default_factory=dict)


class SmokeError(ValueError):
    """No smoke can be planned for this adoption (the reason says what is missing)."""


@dataclass(frozen=True)
class SmokeStep:
    """One step: ``where`` ``node`` (a bash script run over ssh on the smoke node) or
    ``launcher`` (``argv`` run here). ``expect``: a pattern the output must contain."""

    name: str
    where: str
    script: str = ""
    argv: tuple[str, ...] = ()
    timeout_s: float = 3600.0
    expect: str | None = None
    show: str = ""  # what --dry-run prints (default: the argv, or the script)

    def render(self, values: Mapping[str, str]) -> SmokeStep:
        def fill(text: str) -> str:
            for key, value in values.items():
                text = text.replace(key, value)
            return text
        return dataclasses.replace(self, script=fill(self.script), argv=tuple(fill(a) for a in self.argv),
                                   show=fill(self.show))

    def text(self) -> str:
        return self.show or (shlex.join(self.argv) if self.argv else self.script)


@dataclass(frozen=True)
class SmokePlan:
    role: str
    node: Node
    steps: tuple[SmokeStep, ...]
    files: Mapping[str, str] = field(default_factory=dict)  # launcher files the steps read
    notes: tuple[str, ...] = ()
    data: Mapping[str, Any] = field(default_factory=dict)

    def lines(self) -> list[str]:
        out = [f"smoke ({self.role}) on node {self.node.name}:"]
        out += [f"  note: {n}" for n in self.notes]
        for path in self.files:
            out.append(f"  writes {path}")
        for i, step in enumerate(self.steps, 1):
            out.append(f"  {i}. {step.name} [{step.where}]: {step.text()}")
        return out


# --- trainer ---------------------------------------------------------------------------


def _slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(model).rstrip("/").rsplit("/", 1)[-1]).strip("-") or "model"


def _hardware(node: Node):
    from ..rl.model_registry.schema import HardwareFacts  # noqa: PLC0415
    from .engine_images import canonical_device_family  # noqa: PLC0415

    return HardwareFacts(device_family=canonical_device_family(node.device_family), arch=card_arch_of(node.card) or "?",
                         card=node.card, hbm_gib=int(node.hbm_gib), cards_per_node=int(node.cards_per_node),
                         max_nodes=1, interconnect="nvlink+ib")


def cut_config(config: Mapping[str, Any], layers: int) -> dict[str, Any]:
    """``config`` with ``num_hidden_layers`` = ``layers`` (the text config of a VL model)."""
    out = copy.deepcopy(dict(config))
    target = out.get("text_config") if isinstance(out.get("text_config"), dict) else out
    target["num_hidden_layers"] = int(layers)
    return out


def smoke_layout(config: Mapping[str, Any], node: Node, *, engine: str | None,
                 layers: int | None = None) -> dict[str, Any]:
    """The registry's recipe for the full model (``yeto adapt``'s; ``engine`` None: the
    backend it recommends) and the layout of the cut that fits ``node``: ``{"engine",
    "recommendation", "recipe", "layers", "plan", "flags", "tuning"}``. Raises
    :class:`SmokeError` when the registry has no recipe or no cut fits one node."""
    from ..rl.model_registry.match import facts_from_config  # noqa: PLC0415
    from ..rl.model_registry.recommend import plan_trainer, recommend  # noqa: PLC0415
    from ..rl.model_registry.render import BACKEND, cli_flags  # noqa: PLC0415
    from ..rl.model_registry.schema import Constraints  # noqa: PLC0415

    hw = _hardware(node)
    try:
        full = facts_from_config(dict(config))
    except (KeyError, TypeError, ValueError) as error:
        raise SmokeError(f"the model's config.json is not one the registry reads ({type(error).__name__}: "
                         f"{error})") from None
    rec = recommend(full, hw, Constraints(trainer_backend=engine), card_explicit=True)
    recipe = rec.recipe
    if recipe is None or recipe.rl_model_recipe is None or recipe.status == "unsupported":
        why = "; ".join(f"{name}: {reason}" for name, reason in rec.rejected) or "no recipe"
        raise SmokeError(f"the model registry has no {engine or 'trainer'} recipe for this model on {node.card} "
                         f"({why})")
    engine = engine or BACKEND.get(recipe.trainer.split("-", 1)[0], recipe.trainer)
    tuning = rec.trainer_parallel.get("tuning") or recipe.tunings[0]
    candidates = [int(layers)] if layers else list(range(min(MAX_LAYERS, full.layers), 0, -1))
    for n in candidates:
        if n < 1 or n > full.layers:
            raise SmokeError(f"--smoke-layers {n}: the model has {full.layers} layers")
        facts = facts_from_config(cut_config(config, n))
        plan = plan_trainer(facts, hw, recipe, tuning)
        if plan.get("fits"):  # max_nodes 1: a fitting world is one node
            cut = copy.copy(rec)
            cut.facts, cut.supported = facts, True
            cut.trainer_parallel = {**plan, "tuning": tuning}
            flags = cli_flags(cut, f"{SMOKE_MODEL_ORG}/{_slug('m')}")
            unresolved = [f for f in flags if f.startswith("<")]
            return {"engine": engine, "recommendation": rec, "recipe": recipe, "layers": n,
                    "plan": cut.trainer_parallel, "flags": flags, "unresolved": unresolved, "tuning": tuning}
    raise SmokeError(f"no cut of at most {candidates[0]} layers fits one {node.cards_per_node} x {node.card} node "
                     f"in the registry's sizing (tuning {tuning})")


def inventory_entry(node: Node, **over: Any) -> dict[str, Any]:
    """``node`` as an inventory entry (the keys a node of the file may have)."""
    defaults = {f.name: f.default for f in dataclasses.fields(Node)
                if f.default is not dataclasses.MISSING}
    out: dict[str, Any] = {}
    for key in INVENTORY_KEYS:
        value = getattr(node, key, None)
        if key in defaults and value == defaults[key]:
            continue
        if value in (None, (), {}):
            continue
        out[key] = list(value) if isinstance(value, tuple) else dict(value) if isinstance(value, Mapping) else value
    out.update(over)
    return {k: v for k, v in out.items() if v is not None}


def _replace_flag(flags: list[str], extra: Sequence[str]) -> list[str]:
    """``flags`` without the ``--flag value`` pairs ``extra`` gives again (the operator's
    value wins: a registry placeholder such as ``<expert_selection.json>``)."""
    given = {a.split("=", 1)[0] for a in extra if a.startswith("--")}
    out, i = [], 0
    while i < len(flags):
        if flags[i] in given:
            i += 2 if i + 1 < len(flags) and not flags[i + 1].startswith("--") else 1
            continue
        out.append(flags[i])
        i += 1
    return out


def trainer_plan(node: Node, *, model: str, config: Mapping[str, Any], engine: str | None, store: str | None,
                 run_id: str, workdir: Path, model_revision: str | None = None, model_path: str | None = None,
                 layers: int | None = None, extra_args: Sequence[str] = (),
                 python: str | None = None) -> SmokePlan:
    """The trainer smoke's steps (module doc). Raises :class:`SmokeError` for what is
    missing: a backend without a train-only replay, a node without ``scratch``, no
    store for the checkpoint, a registry placeholder no ``--smoke-arg`` fills."""
    if node.runtime != "docker":
        raise SmokeError(f"node {node.name} runs {node.runtime}: the trainer smoke runs the image (runtime docker)")
    if not node.scratch:
        raise SmokeError(f"node {node.name} has no scratch: the cut model is written under the node's scratch "
                         "(inventory scratch: <local NVMe path>)")
    if not store:
        raise SmokeError("the trainer smoke writes its checkpoint to a run store: pass --smoke-store s3://BUCKET/PREFIX")
    layout = smoke_layout(config, node, engine=engine, layers=layers)
    engine = layout["engine"]
    if engine not in TRAINER_BACKENDS:
        raise SmokeError(f"no trainer smoke for backend {engine} yet (smoke recipes: {list(TRAINER_BACKENDS)}): "
                         "its train-only replay is not wired")
    n = layout["layers"]
    slug = f"{_slug(model)}-{n}layer"
    repo = f"{SMOKE_MODEL_ORG}/{slug}"
    host_hub = f"{node.scratch.rstrip('/')}/image-adopt/hub"
    flags = _replace_flag([repo if f == f"{SMOKE_MODEL_ORG}/{_slug('m')}" else f for f in layout["flags"]],
                          extra_args)
    unresolved = [f for f in flags if f.startswith("<")]
    if unresolved:
        raise SmokeError(f"the registry's flags for this model need {unresolved}: pass the flag with --smoke-arg "
                         "(e.g. --smoke-arg=--rl-expert-selection=/path/selection.json)")
    # 1. prune: the cut as a hub snapshot on the node (models--<org>--<name>/snapshots/<rev>)
    source = model_path or f"hf://{model}"
    cache = f"models--{SMOKE_MODEL_ORG}--{slug}"
    script_b64 = base64.b64encode(PRUNE_SCRIPT.read_bytes()).decode()
    mounts = ["-v", f"{host_hub}:{HUB_IN_CONTAINER}"]
    if model_path:
        mounts += ["-v", f"{model_path}:{model_path}:ro"]
    inner = (f"set -euo pipefail; tmp={HUB_IN_CONTAINER}/.cut-{slug}; rm -rf \"$tmp\"; "
             f"{PYTHON} /tmp/yeto-prune_layers.py {shlex.quote(source)} \"$tmp\" --layers {n}"
             + (f" --revision {shlex.quote(model_revision)}" if model_revision and not model_path else "")
             + f"; rev=$(sha1sum \"$tmp/config.json\" | cut -c1-40); "
             f"mkdir -p {HUB_IN_CONTAINER}/{cache}/snapshots {HUB_IN_CONTAINER}/{cache}/refs; "
             f"rm -rf {HUB_IN_CONTAINER}/{cache}/snapshots/$rev; mv \"$tmp\" {HUB_IN_CONTAINER}/{cache}/snapshots/$rev; "
             f"printf %s \"$rev\" > {HUB_IN_CONTAINER}/{cache}/refs/main; echo {REVISION_MARK} $rev")
    prune = (f"set -euo pipefail\nmkdir -p {shlex.quote(host_hub)}\n"
             f"base64 -d > {shlex.quote(host_hub)}/.prune_layers.py <<'YETO_PRUNE_EOF'\n{script_b64}\nYETO_PRUNE_EOF\n"
             + shlex.join(["docker", "run", "--rm", "--network", "host", *mounts,
                           "-v", f"{host_hub}/.prune_layers.py:/tmp/yeto-prune_layers.py:ro",
                           "-e", "HF_TOKEN", "-e", "HF_ENDPOINT", "--entrypoint", "bash", REF, "-c", inner]) + "\n")
    prune_show = (f"docker run --rm {' '.join(mounts)} {REF} {PYTHON} scripts/prune_layers.py {source} "
                  f"<hub>/{cache}/snapshots/<sha1 of config.json> --layers {n}")
    # 2./3. train and resume: yeto launch of a one-node ssh: island
    inventory = workdir / "inventory.yaml"
    entry = inventory_entry(node, roles=["trainer"], image=None,
                            mounts={**dict(node.mounts), host_hub: HUB_IN_CONTAINER})
    facts = layout["recommendation"].facts
    synth = {"lengths": [2048], "samples": 2, "prompt": 1024, "topk": 2, "layers": n,
             "dense_layers": min(int(facts.first_k_dense or 0), n), "router_topk": max(1, int(facts.topk or 1)),
             "experts": max(1, int(facts.num_experts or 1)), "vocab": int(facts.vocab)}
    if synth["experts"] % synth["router_topk"]:
        synth["router_topk"] = 1
    base = [python or sys.executable, "-m", "yeto.cli", "launch", *flags, "--model-revision", REVISION,
            "--gpu", f"ssh:1x{node.cards_per_node}x{node.card}", "--rl-node-inventory", str(inventory),
            "--rl-image", f"docker:{REF}", "--model-store", f"local://{HUB_IN_CONTAINER}",
            "--rl-checkpoint-store", f"{store.rstrip('/')}/{run_id}", "--cluster-prefix", run_id,
            "--island-env", f"{SYNTH_ENV}={json.dumps(synth, separators=(',', ':'))}",
            "--island-env", f"{REPLAY_ENV}={REPLAY_PATH}", *extra_args]
    steps = (
        SmokeStep("prune", "node", script=prune, timeout_s=PRUNE_TIMEOUT_S, expect=REVISION_MARK, show=prune_show),
        SmokeStep("train", "launcher", argv=(*base, "--rl-stop-after-rounds", "1"), timeout_s=TRAIN_TIMEOUT_S),
        SmokeStep("resume", "launcher", argv=(*base, "--rl-stop-after-rounds", "2"), timeout_s=TRAIN_TIMEOUT_S,
                  expect=r"rl_resume|resum(?:e|ed|ing) from"),
    )
    import yaml  # noqa: PLC0415

    plan = layout["plan"]
    notes = (f"{layout['recipe'].name} ({layout['recipe'].rl_model_recipe}, tuning {layout['tuning']}): "
             f"the first {n} of {facts.layers} layers on one node, TP{plan['tensor_parallel']} "
             f"PP{plan['pipeline_parallel']} EP{plan['expert_parallel']}, {plan['state_gib_per_card']} GiB state per card",
             "train-only: synthetic rollouts replayed by the trainer (no sampler); the cut of round 1 is the "
             "checkpoint, round 2 resumes from it")
    return SmokePlan("trainer", node, steps, files={str(inventory): yaml.safe_dump({"nodes": [entry]},
                                                                                     sort_keys=False)},
                     notes=notes, data={"layers": n, "layout": {k: plan[k] for k in (
                         "tensor_parallel", "pipeline_parallel", "expert_parallel", "data_parallel")},
                         "recipe": layout["recipe"].rl_model_recipe, "tuning": layout["tuning"], "run": run_id,
                         "engine": engine})


# --- sampler ---------------------------------------------------------------------------


def sampler_plan(node: Node, *, spec: Mapping[str, Any], run_id: str, release: str | None = None,
                 code: tuple[str, bytes] | None = None) -> SmokePlan:
    """The sampler smoke's steps: put the yeto tree and the spec on the node, then the
    node runtime's container start of :mod:`yeto.node.smoke` on the image."""
    from yeto.node import launch as node_launch  # noqa: PLC0415
    from yeto.node.code import tree_code_tar  # noqa: PLC0415
    from yeto.node.profile import CONTAINER_WORK, node_dirs  # noqa: PLC0415

    profile = profile_of(node)
    if profile.container not in ("docker", "podman"):
        raise SmokeError(f"node {node.name} runs samplers as a {profile.container}: the image smoke needs a "
                         "container runtime (docker / podman)")
    if not isinstance(spec.get("engine"), Mapping):
        raise SmokeError("the sampler spec has no engine section")
    sha, tar = code or tree_code_tar(REPO_ROOT)
    work = node_dirs(profile)["work"]
    base = f"{work}/image-adopt/{run_id}"
    tree_host, tree_in = f"{work}/image-adopt/yeto/{sha}", f"{CONTAINER_WORK}/image-adopt/yeto/{sha}"
    spec_host = f"{base}/spec.json"
    spec_in = f"{CONTAINER_WORK}/image-adopt/{run_id}/spec.json"
    put = (f"set -euo pipefail\nmkdir -p {shlex.quote(base)}\n"
           f"if [ ! -d {shlex.quote(tree_host)}/yeto ]; then rm -rf {shlex.quote(tree_host)}.tmp; "
           f"mkdir -p {shlex.quote(tree_host)}.tmp; base64 -d <<'YETO_TREE_EOF' | tar x -C {shlex.quote(tree_host)}.tmp\n"
           f"{base64.b64encode(tar).decode()}\nYETO_TREE_EOF\nmv {shlex.quote(tree_host)}.tmp {shlex.quote(tree_host)}; fi\n"
           f"cat > {shlex.quote(spec_host)} <<'YETO_SPEC_EOF'\n{json.dumps(dict(spec), indent=1)}\nYETO_SPEC_EOF\n")
    env = {"PYTHONPATH": tree_in, "PYTHONUNBUFFERED": "1", **profile.family_env()}
    argv = node_launch.container_argv(profile, name=f"yeto-adopt-{run_id}", image=REF, env=env, python=PYTHON,
                                      spec_path=spec_in, module="yeto.node.smoke",
                                      module_args=["--spec", spec_in, *(["--release", release] if release else [])])
    steps = (SmokeStep("upload", "node", script=put, timeout_s=600.0,
                       show=f"put the yeto tree {sha} at {tree_host} and the spec at {spec_host}"),
             SmokeStep("engine", "node", script=shlex.join(argv), timeout_s=SAMPLER_TIMEOUT_S))
    engine = spec["engine"]
    notes = (f"engine {engine.get('adapter') or engine.get('factory')}: start, generate, reload "
             f"({'release ' + release if release else 'the base weights'} as version 1), generate",)
    return SmokePlan("sampler", node, steps, notes=notes, data={"run": run_id, "code": sha})


# --- run ------------------------------------------------------------------------------


def run_plan(plan: SmokePlan, *, ops: Any, ref: str, python: str | None,
             run: Callable[..., Any] = subprocess.run, clock: Callable[[], float] = time.time,
             log: Callable[[str], None] = lambda m: print(m, file=sys.stderr, flush=True)):
    """Every step of ``plan`` in order; the smoke's ``StepResult`` (``ok`` False names the
    step and the output's tail)."""
    from yeto.node.smoke import parse_result  # noqa: PLC0415

    values = {REF: ref, PYTHON: python or "python3"}
    for path, text in plan.files.items():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text)
    t_start = clock()
    times: dict[str, float] = {}
    data: dict[str, Any] = {**plan.data, "steps": times}

    def failed(step: SmokeStep, why: str) -> Any:
        return StepResult("smoke", False, f"{step.name}: {why}", clock() - t_start, data)

    for step in plan.steps:
        step = step.render(values)
        log(f"[image adopt] smoke {step.name} [{step.where}]")
        t0 = clock()
        try:
            if step.where == "node":
                done = ops.run(plan.node, step.script, timeout=step.timeout_s)
            else:
                done = run(list(step.argv), capture_output=True, text=True, timeout=step.timeout_s, check=False)
        except Exception as error:  # noqa: BLE001 - a timeout or a dead ssh is the step's failure
            times[step.name] = round(clock() - t0, 1)
            return failed(step, f"{type(error).__name__}: {error}"[:600])
        times[step.name] = round(clock() - t0, 1)
        out = (getattr(done, "stdout", "") or "") + "\n" + (getattr(done, "stderr", "") or "")
        result = parse_result(out) if plan.role == "sampler" and step.name == "engine" else None
        if result is not None and not result.get("ok"):
            data["engine_steps"] = result.get("seconds")
            return failed(step, f"{result.get('step')} failed: {result.get('error')}")
        if done.returncode != 0:
            return failed(step, f"rc {done.returncode}: {out.strip()[-600:]}")
        if step.expect and not re.search(step.expect, out):
            return failed(step, f"the output has no {step.expect!r}: {out.strip()[-400:]}")
        if step.name == "engine" and plan.role == "sampler":
            if result is None:
                return failed(step, f"no smoke result in the output: {out.strip()[-400:]}")
            data["engine_steps"] = result.get("seconds")
            if result.get("reload_mode"):
                data["reload_mode"] = result["reload_mode"]
        match = re.search(rf"{REVISION_MARK} ([0-9a-f]{{40}})", out)
        if match:
            values[REVISION] = match.group(1)
            data["revision"] = match.group(1)
    what = "one step trained, checkpoint saved and resumed" if plan.role == "trainer" else \
        "engine started, generated, reloaded and generated again"
    return StepResult("smoke", True, f"{what} on {plan.node.name}", clock() - t_start, data)

"""Step 5 of ``yeto image adopt``: the role's smoke run of the image on one node.

**trainer** (``--role trainer``): the model, cut down until one node holds it, trains
one step, saves a checkpoint, reads it back and trains on:

1. ``prune`` (on the node, in the image): ``scripts/prune_layers.py`` keeps the first
   ``N`` decoder layers (all their experts, the embeddings and the head; the MFU 16L
   cut of 2026-11-18; MTP layers are dropped) and lays the result out as a Hugging
   Face hub snapshot under the node's ``scratch`` (revision = the sha1 of the cut
   ``config.json``); the shards it downloads go to a cache next to it, removed after
   the cut, and the scratch must have room for both (:func:`cut_gib`). ``N`` is the
   largest of at most :data:`MAX_LAYERS` whose trainer state fits the node's cards in
   the model registry's sizing (:func:`yeto.rl.model_registry.recommend.plan_trainer`,
   with the backend, recipe and tuning ``yeto adapt`` recommends for the full model;
   ``--smoke-arg=--tuning=...`` overrides the tuning);
2. ``train`` (on the launcher): ``yeto launch --controller local`` of one ``ssh:``
   island of that node, no syncer (``--rl-single-island-no-sync``), with the image,
   the registry's flags for the cut model (recipe, tuning, TP / PP / EP), a small
   local prompt file and the built-in numeric reward (the launch's own checks need
   them), ``--trust-remote-code`` as :func:`remote_code` decides, and a train-only
   replay of synthetic rollouts (Miles' debug-rollout replay,
   ``YETO_TEST_SYNTH_ROLLOUT``: no sampler); ``--rl-stop-after-rounds 1`` ends the
   round with a cut (the checkpoint) in ``--smoke-store``. The step passes when the
   trainer's ``rl_round_cut`` event names a cut whose manifest is in the store;
3. ``resume`` (on the launcher): the same launch with ``--rl-stop-after-rounds 2``: it
   passes when its ``rl_resume`` / ``rl_round_cut_restored`` event names that cut.

Whatever the outcome (a failure, a timeout), the smoke's containers are removed and
its run stopped, and its artifacts (the cut on the node, the checkpoint in the store,
the launcher's files) deleted unless ``--keep-smoke-artifacts``.

**sampler** (``--role sampler``): the pool's sampler spec on one node, started as the
node runtime starts a sampler (:func:`yeto.node.launch.container_argv`) running
:mod:`yeto.node.smoke`: start the engine, one generation, one reload (the reload of a
trainer release: the release ``--release`` names, else the engine's base weights;
the reload exercises the engine's hot-reload endpoint, not the format of a trainer
release), one more generation. The reload must be in place: one that falls back to a restart
fails the smoke (its ``reload_mode`` named).

:func:`trainer_plan` / :func:`sampler_plan` build the steps (what ``--dry-run``
prints); :func:`run_plan` runs them (an ``SshOps`` for the node steps, ``subprocess``
for the launcher steps; tests pass fakes). The pinned image, the image's interpreter
and the cut's revision are known only while adopting: the steps name them
:data:`REF`, :data:`PYTHON` and :data:`REVISION` until then, and a node script is
built only once they are known (:meth:`SmokeStep.render`: the values are argv
elements of ``shlex.join``, or ``shlex.quote``-d where they sit inside a shell string).
"""

from __future__ import annotations

import argparse
import base64
import copy
import dataclasses
import functools
import json
import os
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
PYTHON_SH = "@PYTHON_SH@"  # the interpreter inside a shell string (rendered shlex.quote-d)
PYTHON_PATH = re.compile(r"/[\w./+-]+")  # an image interpreter the smoke runs
# passed to the node's prune from the launcher's environment: exported by the script (ssh stdin), never in argv
PASS_ENV = ("HF_TOKEN", "HF_ENDPOINT")
TRUST = "--trust-remote-code"
NO_TRUST = "--no-trust-remote-code"  # a --smoke-arg the smoke reads (yeto launch has no such flag)
REWARD = "yeto.rl.numeric_answer_reward:score"  # a built-in reward (the replay never scores)
PROMPTS = [{"prompt": "What is 2 + 2? Put the answer in \\boxed{}.", "label": "4"},
           {"prompt": "What is 3 + 5? Put the answer in \\boxed{}.", "label": "8"}]
# launch flags the smoke sets unless --smoke-arg gives them: a rate (one step: it barely matters); no model
# preflight from the launcher (the cut exists only on the node: the prune step made and checked it)
LAUNCH_DEFAULTS = (("--inner-lr", "1e-6"), ("--preflight-model", "off"))
ATTENTION = "--rl-training-attention-backend"  # the trainer's attention backend (yeto image adopt --attention-backend)
ATTENTION_BACKENDS = ("auto", "flash", "fused", "unfused")  # the launch flag's choices
BYTES_PER_PARAM = {"bf16": 2.0, "fp16": 2.0, "fp32": 4.0, "fp8": 1.0, "int8": 1.0, "mxfp4": 0.53, "int4": 0.53,
                   "nvfp4": 0.56}
MAX_LAYERS = 16  # the MFU 16L cut
HUB_IN_CONTAINER = "/data/image-adopt-hub"  # where the trainer sees the cut (--model-store local://)
SMOKE_MODEL_ORG = "yeto-local"
REVISION_MARK = "YETO_SMOKE_REVISION"
SPACE_MARK = "YETO_SMOKE_SPACE"
TRAINER_BACKENDS = ("miles",)  # backends with a train-only replay (Miles' debug-rollout replay)
# A global rule, not a backend's: yeto.launch.rl_args refuses every RL launch (any --rl-backend) without
# --trust-remote-code, whatever the model, so the smoke's launch always passes it
TRUST_REQUIRED = ("yeto launch refuses every RL launch without it (the pinned trainer's loaders always allow "
                  "remote model code, so the launch states it)")
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


def fill(text: str, values: Mapping[str, str]) -> str:
    """``text`` with the placeholders of ``values``; :data:`PYTHON_SH` takes the
    interpreter ``shlex.quote``-d (it sits inside a shell string)."""
    for key, value in values.items():
        text = text.replace(key, value)
    if PYTHON in values:
        text = text.replace(PYTHON_SH, shlex.quote(values[PYTHON]))
    return text


@dataclass(frozen=True)
class SmokeStep:
    """One step: ``where`` ``node`` (a bash script run over ssh on the smoke node:
    ``script``, or ``prefix`` + ``shlex.join(node_argv)`` + ``suffix`` built at
    :meth:`render`) or ``launcher`` (``argv`` run here). ``expect``: a pattern the output must contain; ``check``: the run's
    check of the output (``round_cut`` / ``resume``, :func:`run_plan`); ``env``: the
    launcher variables a node script exports first (their values never printed)."""

    name: str
    where: str
    script: str = ""
    argv: tuple[str, ...] = ()
    timeout_s: float = 3600.0
    expect: str | None = None
    show: str = ""  # what --dry-run prints (default: the argv, or the script)
    node_argv: tuple[str, ...] = ()
    prefix: str = ""
    suffix: str = ""
    check: str | None = None
    env: tuple[str, ...] = ()

    def render(self, values: Mapping[str, str], environ: Mapping[str, str] | None = None) -> SmokeStep:
        """The step with the placeholders filled. A node script built from ``node_argv``
        joins the filled elements (``shlex.join``: a value is one shell word), and
        exports the step's ``env`` variables the launcher has (``environ``)."""
        script = fill(self.script, values)
        if self.node_argv:
            script = fill(self.prefix, values) + shlex.join(fill(a, values) for a in self.node_argv) + \
                fill(self.suffix, values)
        environ = os.environ if environ is None else environ
        exports = "".join(f"export {name}={shlex.quote(environ[name])}\n" for name in self.env if environ.get(name))
        if exports and script:
            script = exports + script
        return dataclasses.replace(self, script=script, argv=tuple(fill(a, values) for a in self.argv),
                                   show=fill(self.show, values))

    def text(self) -> str:
        if self.show:
            return self.show
        if self.argv:
            return shlex.join(self.argv)
        if self.node_argv:
            return shlex.join(self.node_argv)
        return self.script


@dataclass(frozen=True)
class SmokePlan:
    role: str
    node: Node
    steps: tuple[SmokeStep, ...]
    files: Mapping[str, str] = field(default_factory=dict)  # launcher files the steps read
    notes: tuple[str, ...] = ()
    data: Mapping[str, Any] = field(default_factory=dict)
    cleanup: tuple[SmokeStep, ...] = ()  # run whatever the steps did (run_plan's finally)
    artifacts: tuple[tuple[str, str], ...] = ()  # (where, path): what the cleanup deletes

    def lines(self) -> list[str]:
        out = [f"smoke ({self.role}) on node {self.node.name}:"]
        out += [f"  note: {n}" for n in self.notes]
        for path in self.files:
            out.append(f"  writes {path}")
        for i, step in enumerate(self.steps, 1):
            out.append(f"  {i}. {step.name} [{step.where}]: {step.text()}")
        for step in self.cleanup:
            out.append(f"  finally: {step.name} [{step.where}]: {step.text()}")
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
                 layers: int | None = None, tuning: str | None = None) -> dict[str, Any]:
    """The registry's recipe for the full model (``yeto adapt``'s; ``engine`` None: the
    backend it recommends) and the layout of the cut that fits ``node``: ``{"engine",
    "recommendation", "recipe", "layers", "plan", "flags", "tuning"}`` (``tuning`` None:
    the registry's). Raises :class:`SmokeError` when the registry has no recipe or no
    cut fits one node."""
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
    if tuning is not None and tuning not in recipe.tunings:
        raise SmokeError(f"--tuning {tuning}: the recipe {recipe.name} has tunings {list(recipe.tunings)}")
    tuning = tuning or rec.trainer_parallel.get("tuning") or recipe.tunings[0]
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


def cut_gib(config: Mapping[str, Any], layers: int) -> float:
    """The cut's weights in GiB, the registry's parameter count of the cut times the
    checkpoint's bytes per parameter."""
    from ..rl.model_registry.match import facts_from_config  # noqa: PLC0415

    facts = facts_from_config(cut_config(config, layers))
    per = BYTES_PER_PARAM.get(str(facts.checkpoint_quant).lower(), 2.0)
    return round(float(facts.total_params_b) * 1e9 * per / 2**30, 1)


def _flag_value(args: Sequence[str], flag: str) -> str | None:
    """The value of ``flag`` in ``args`` (``--flag value`` or ``--flag=value``; the last wins)."""
    found = None
    for i, arg in enumerate(args):
        if arg == flag and i + 1 < len(args):
            found = args[i + 1]
        elif arg.startswith(flag + "="):
            found = arg.split("=", 1)[1]
    return found


@functools.lru_cache(maxsize=1)
def _launch_options() -> frozenset[str]:
    """The long options of ``yeto launch``."""
    from ..cli import build_parser  # noqa: PLC0415

    parser = build_parser()
    action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))  # noqa: SLF001
    return frozenset(o for a in action.choices["launch"]._actions for o in a.option_strings  # noqa: SLF001
                     if o.startswith("--"))


def _abbreviates(name: str, flag: str) -> bool:
    """``name`` is an abbreviation of ``flag`` as argparse took one: a proper prefix that
    no other ``yeto launch`` option starts with (a prefix that other options share is
    ambiguous, not an abbreviation)."""
    if not name.startswith("--") or len(name) <= 2 or name == flag or not flag.startswith(name):
        return False
    return not any(o.startswith(name) for o in _launch_options() if o != flag)


def smoke_attention(attention_backend: str | None, extra_args: Sequence[str]) -> str:
    """The trainer's attention backend the fit checks and the smoke trains with:
    ``--attention-backend``, else a ``--smoke-arg`` of :data:`ATTENTION`, else ``auto``
    (the recipe's). The two naming different backends, a value outside
    :data:`ATTENTION_BACKENDS` and an abbreviation of the flag are a :class:`SmokeError`."""
    for arg in extra_args:  # yeto launch takes long options by their full name only (no abbreviations)
        name = str(arg).split("=", 1)[0]
        if name != ATTENTION and _abbreviates(name, ATTENTION):
            raise SmokeError(f"--smoke-arg {name}: an abbreviation of {ATTENTION}; yeto launch takes long options "
                             f"by their full name only: write {ATTENTION}")
    asked, given = attention_backend or "auto", _flag_value(extra_args, ATTENTION)
    for where, value in (("--attention-backend", asked), (f"--smoke-arg {ATTENTION}", given)):
        if value is not None and value not in ATTENTION_BACKENDS:
            raise SmokeError(f"{where} {value!r} is not an attention backend (one of {list(ATTENTION_BACKENDS)})")
    if given is not None and asked != "auto" and given != asked:
        raise SmokeError(f"--attention-backend {asked} but --smoke-arg {ATTENTION} {given}: the fit would check "
                         "another backend than the smoke trains with; give one")
    return given or asked


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
                 layers: int | None = None, extra_args: Sequence[str] = (), python: str | None = None,
                 keep: bool = False, attention_backend: str = "auto") -> SmokePlan:
    """The trainer smoke's steps (module doc). Raises :class:`SmokeError` for what is
    missing: a backend without a train-only replay, a node without ``scratch``, no
    store for the checkpoint, a registry placeholder no ``--smoke-arg`` fills.
    ``keep``: the cleanup leaves the artifacts and prints where they are;
    ``attention_backend`` (not ``auto``): the launch trains with it (:data:`ATTENTION`)."""
    engine_name = profile_of(node).container
    if node.runtime != "docker" or engine_name != "docker":
        raise SmokeError(f"node {node.name} runs {node.runtime} (container {engine_name}): the trainer smoke runs "
                         "the image as an ssh: trainer island, which starts docker containers only (inventory "
                         "runtime: docker); podman and rootfs nodes take sampler images (--role sampler)")
    if not node.scratch:
        raise SmokeError(f"node {node.name} has no scratch: the cut model is written under the node's scratch "
                         "(inventory scratch: <local NVMe path>)")
    if not store:
        raise SmokeError("the trainer smoke writes its checkpoint to a run store: pass --smoke-store s3://BUCKET/PREFIX")
    tuning = _flag_value(extra_args, "--tuning")
    layout = smoke_layout(config, node, engine=engine, layers=layers, tuning=tuning)
    engine = layout["engine"]
    if engine not in TRAINER_BACKENDS:
        raise SmokeError(f"no trainer smoke for backend {engine} yet (smoke recipes: {list(TRAINER_BACKENDS)}): "
                         "its train-only replay is not wired")
    trust, trust_notes = remote_code(config, layout["recommendation"].family, extra_args)
    extra_args = [a for a in extra_args if a not in (TRUST, NO_TRUST)]
    attention = smoke_attention(attention_backend, extra_args)
    if attention != "auto" and _flag_value(extra_args, ATTENTION) is None:
        extra_args += [ATTENTION, attention]  # as a --smoke-arg: it replaces a registry flag of its own
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
    download = f"{HUB_IN_CONTAINER}/.dl-{slug}"
    size = cut_gib(config, n)
    need_gib = round(size * (1.1 if model_path else 2.2) + 1.0, 1)  # the cut, and the shards it reads (hf://)
    script_b64 = base64.b64encode(PRUNE_SCRIPT.read_bytes()).decode()
    mounts = ["-v", f"{host_hub}:{HUB_IN_CONTAINER}"]
    if model_path:
        mounts += ["-v", f"{model_path}:{model_path}:ro"]
    options = ""
    if not model_path:  # hf://: the revision asked, the shards in a cache removed after the cut
        options = (f" --revision {shlex.quote(model_revision)}" if model_revision else "") + f" --cache-dir {download}"
    inner = (f"set -euo pipefail; tmp={HUB_IN_CONTAINER}/.cut-{slug}; rm -rf \"$tmp\"; "
             f"{PYTHON_SH} /tmp/yeto-prune_layers.py {shlex.quote(source)} \"$tmp\" --layers {n}{options}"
             + f"; rev=$(sha1sum \"$tmp/config.json\" | cut -c1-40); "
             f"mkdir -p {HUB_IN_CONTAINER}/{cache}/snapshots {HUB_IN_CONTAINER}/{cache}/refs; "
             f"rm -rf {HUB_IN_CONTAINER}/{cache}/snapshots/$rev; mv \"$tmp\" {HUB_IN_CONTAINER}/{cache}/snapshots/$rev; "
             f"printf %s \"$rev\" > {HUB_IN_CONTAINER}/{cache}/refs/main; rm -rf {download}; echo {REVISION_MARK} $rev")
    hub_q = shlex.quote(host_hub)
    prefix = (f"set -euo pipefail\nmkdir -p {hub_q}\n"
              f"free_kb=$(df -Pk {hub_q} | awk 'NR==2 {{print $4}}')\n"
              f"if [ \"$free_kb\" -lt {int(need_gib * 1024 * 1024)} ]; then echo \"{SPACE_MARK} {host_hub} has "
              f"$((free_kb / 1048576)) GiB free; the cut ({size} GiB) and the shards it reads need {need_gib} GiB\"; "
              "exit 7; fi\n"
              f"base64 -d > {hub_q}/.prune_layers.py <<'YETO_PRUNE_EOF'\n{script_b64}\nYETO_PRUNE_EOF\n")
    node_argv = ("docker", "run", "--rm", "--network", "host", *mounts,
                 "-v", f"{host_hub}/.prune_layers.py:/tmp/yeto-prune_layers.py:ro",
                 *(a for name in PASS_ENV for a in ("-e", name)), "--entrypoint", "bash", REF, "-c", inner)
    prune_show = (f"docker run --rm {' '.join(mounts)} {REF} {PYTHON} scripts/prune_layers.py {source} "
                  f"<hub>/{cache}/snapshots/<sha1 of config.json> --layers {n}"
                  + ("" if model_path else f" --cache-dir {download}") + f" (needs {need_gib} GiB free)")
    # 2./3. train and resume: yeto launch of a one-node ssh: island
    inventory = workdir / "inventory.yaml"
    prompts = workdir / "prompts.jsonl"
    entry = inventory_entry(node, roles=["trainer"], image=None,
                            mounts={**dict(node.mounts), host_hub: HUB_IN_CONTAINER})
    facts = layout["recommendation"].facts
    synth = {"lengths": [2048], "samples": 2, "prompt": 1024, "topk": 2, "layers": n,
             "dense_layers": min(int(facts.first_k_dense or 0), n), "router_topk": max(1, int(facts.topk or 1)),
             "experts": max(1, int(facts.num_experts or 1)), "vocab": int(facts.vocab)}
    if synth["experts"] % synth["router_topk"]:
        synth["router_topk"] = 1
    run_store = f"{store.rstrip('/')}/{run_id}"
    yeto = [python or sys.executable, "-m", "yeto.cli"]
    base = [*yeto, "launch", *flags, "--model-revision", REVISION,
            "--gpu", f"ssh:1x{node.cards_per_node}x{node.card}", "--rl-node-inventory", str(inventory),
            "--rl-image", f"docker:{REF}", "--model-store", f"local://{HUB_IN_CONTAINER}",
            "--rl-checkpoint-store", run_store, "--cluster-prefix", run_id,
            "--controller", "local", "--rl-single-island-no-sync", *((TRUST,) if trust else ()),
            "--data", str(prompts), "--rl-allow-local-data", "--reward-function", REWARD,
            *(a for flag, value in LAUNCH_DEFAULTS if _flag_value(extra_args, flag) is None for a in (flag, value)),
            "--island-env", f"{SYNTH_ENV}={json.dumps(synth, separators=(',', ':'))}",
            "--island-env", f"{REPLAY_ENV}={REPLAY_PATH}", *extra_args]
    steps = (
        SmokeStep("prune", "node", node_argv=node_argv, prefix=prefix, suffix="\n", timeout_s=PRUNE_TIMEOUT_S,
                  expect=REVISION_MARK, show=prune_show, env=PASS_ENV),
        SmokeStep("train", "launcher", argv=(*base, "--rl-stop-after-rounds", "1"), timeout_s=TRAIN_TIMEOUT_S,
                  check="round_cut"),
        SmokeStep("resume", "launcher", argv=(*base, "--rl-stop-after-rounds", "2"), timeout_s=TRAIN_TIMEOUT_S,
                  check="resume"),
    )
    import yaml  # noqa: PLC0415

    plan = layout["plan"]
    picked = "--smoke-arg" if tuning else "the registry's"
    notes = [f"{layout['recipe'].name} ({layout['recipe'].rl_model_recipe}, tuning {layout['tuning']} ({picked}; "
             f"--smoke-arg=--tuning=... overrides it)): the first {n} of {facts.layers} layers on one node, "
             f"TP{plan['tensor_parallel']} PP{plan['pipeline_parallel']} EP{plan['expert_parallel']}, "
             f"{plan['state_gib_per_card']} GiB state per card",
             "train-only: synthetic rollouts replayed by the trainer (no sampler); the cut of round 1 is the "
             "checkpoint, round 2 resumes from it", *trust_notes]
    mtp = _mtp_layers(config)
    if mtp:
        notes.append(f"the cut drops the model's {mtp} MTP layer(s) (num_nextn_predict_layers): the smoke does not "
                     "train them")
    container = f"^yeto-{run_id}-"
    remove = (f"docker ps -aq --filter name={shlex.quote(container)} | while read -r c; do "
              "docker rm -f \"$c\" >/dev/null 2>&1 || true; done")
    cut_dirs = [f"{host_hub}/{cache}", f"{host_hub}/.dl-{slug}", f"{host_hub}/.cut-{slug}"]
    artifacts = (*(("node", d) for d in cut_dirs), ("store", run_store), ("launcher", str(workdir)))
    cleanup = [SmokeStep("stop", "launcher", argv=(*yeto, "down", run_id, "--missing-ok"), timeout_s=600.0),
               SmokeStep("containers", "node", script=remove, timeout_s=300.0)]
    if keep:
        cleanup.append(SmokeStep("keep", "node", script="du -sh " + " ".join(shlex.quote(d) for d in cut_dirs)
                                 + " 2>/dev/null || true", timeout_s=300.0))
    else:
        cleanup.append(SmokeStep("artifacts", "node", script="rm -rf " + " ".join(shlex.quote(d) for d in cut_dirs),
                                 timeout_s=600.0))
    return SmokePlan("trainer", node, steps,
                     files={str(inventory): yaml.safe_dump({"nodes": [entry]}, sort_keys=False),
                            str(prompts): "".join(json.dumps(p) + "\n" for p in PROMPTS)},
                     notes=tuple(notes), data={"layers": n, "layout": {k: plan[k] for k in (
                         "tensor_parallel", "pipeline_parallel", "expert_parallel", "data_parallel")},
                         "recipe": layout["recipe"].rl_model_recipe, "tuning": layout["tuning"], "run": run_id,
                         "engine": engine, "store": run_store, "workdir": str(workdir), "keep": bool(keep),
                         "cut_gib": size, "trust_remote_code": trust, "notes": list(notes)},
                     cleanup=tuple(cleanup), artifacts=artifacts)


def remote_code(config: Mapping[str, Any], family: str | None,
                extra_args: Sequence[str] = ()) -> tuple[bool, list[str]]:
    """``(on, notes)``: whether the smoke's launch passes ``--trust-remote-code``.

    The model: the registry has no such field; its equivalent is the model's
    ``config.json`` ``auto_map`` (the classes transformers loads from the repository,
    what :func:`yeto.rl.adapters.miles.remote_code_prewarm.auto_map` reads). A config
    (or its ``text_config``) with one needs remote code, a registry family without one
    does not, and a model the registry does not know (its generic fallback) is taken
    to need it: the smoke cannot tell. ``--smoke-arg=--trust-remote-code`` /
    ``--smoke-arg=--no-trust-remote-code`` decide instead (the last one given wins).

    The launch: every RL launch refuses to start without the flag whatever the model
    and backend (:data:`TRUST_REQUIRED`, a global rule of :mod:`yeto.launch.rl_args`),
    so the result is always on: the note says what the model alone needs, and
    ``--no-trust-remote-code`` is a :class:`SmokeError`."""
    from ..rl.model_registry.families import FAMILIES  # noqa: PLC0415

    given = [a for a in extra_args if a in (TRUST, NO_TRUST)]
    text = config.get("text_config") if isinstance(config.get("text_config"), Mapping) else {}
    if given:
        model = given[-1] == TRUST
        why = f"the model: {TRUST} {'on' if model else 'off'} (--smoke-arg={given[-1]})"
    elif config.get("auto_map") or text.get("auto_map"):
        model, why = True, f"the model: {TRUST} on (its config.json auto_map names remote code)"
    elif family not in {f.family for f in FAMILIES}:
        model, why = True, (f"the model: {TRUST} on by default (the model registry does not know this model, "
                            f"family {family or '?'}: the smoke cannot tell whether it needs remote code)")
    else:
        model, why = False, (f"the model: {TRUST} off (the registry's family {family} has no auto_map in "
                             "config.json)")
    if given and not model:
        raise SmokeError(f"--smoke-arg={NO_TRUST}: the launch refuses to start without {TRUST} ({TRUST_REQUIRED})")
    if not model:
        why += f"; the launch passes it anyway: {TRUST_REQUIRED}"
    return True, [why]


def _mtp_layers(config: Mapping[str, Any]) -> int:
    text = config.get("text_config") if isinstance(config.get("text_config"), Mapping) else config
    return int((text or {}).get("num_nextn_predict_layers") or 0)


# --- sampler ---------------------------------------------------------------------------


def sampler_plan(node: Node, *, spec: Mapping[str, Any], run_id: str, release: str | None = None,
                 code: tuple[str, bytes] | None = None, keep: bool = False, sudo: bool = False) -> SmokePlan:
    """The sampler smoke's steps: put the yeto tree and the spec on the node, then the
    node runtime's container start of :mod:`yeto.node.smoke` on the image. The reload
    loads ``release``, else the spec's ``engine.server.base_dir``: a spec with neither
    is refused here. ``keep``: the cleanup leaves the run's directory on the node;
    ``sudo``: the engine runs as ``sudo -n <engine>`` (the pool's ``container.sudo``)."""
    from yeto.node import launch as node_launch  # noqa: PLC0415
    from yeto.node.code import tree_code_tar  # noqa: PLC0415
    from yeto.node.profile import CONTAINER_WORK, node_dirs  # noqa: PLC0415
    from yeto.node.smoke import base_dir  # noqa: PLC0415

    profile = profile_of(node)
    if profile.container not in ("docker", "podman"):
        raise SmokeError(f"node {node.name} runs samplers as a {profile.container}: the image smoke needs a "
                         "container runtime (docker / podman)")
    if not isinstance(spec.get("engine"), Mapping):
        raise SmokeError("the sampler spec has no engine section")
    if not release and not base_dir(spec):
        raise SmokeError("the sampler smoke's reload loads a release: pass --release (a container path), or give "
                         "the spec's engine.server.base_dir")
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
    name = container_name(run_id)
    argv = node_launch.container_argv(profile, name=name, image=REF, env=env, python=PYTHON_SH,
                                      spec_path=spec_in, sudo=sudo, module="yeto.node.smoke",
                                      module_args=["--spec", spec_in, *(["--release", release] if release else [])])
    steps = (SmokeStep("upload", "node", script=put, timeout_s=600.0,
                       show=f"put the yeto tree {sha} at {tree_host} and the spec at {spec_host}"),
             SmokeStep("engine", "node", node_argv=tuple(argv), timeout_s=SAMPLER_TIMEOUT_S))
    engine = spec["engine"]
    notes = (f"engine {engine.get('adapter') or engine.get('factory')}: start, generate, reload "
             f"({'release ' + release if release else 'the base weights'} as version 1), generate; the reload "
             "exercises the engine's hot-reload endpoint, not the format of a trainer release",)
    ce = ("sudo -n " if sudo else "") + shlex.quote(profile.container)
    cleanup = [SmokeStep("container", "node", script=f"{ce} rm -f {shlex.quote(name)} >/dev/null 2>&1 || true",
                         timeout_s=300.0)]
    cleanup.append(SmokeStep("keep", "node", script=f"du -sh {shlex.quote(base)} 2>/dev/null || true", timeout_s=300.0)
                   if keep else SmokeStep("artifacts", "node", script=f"rm -rf {shlex.quote(base)}", timeout_s=300.0))
    return SmokePlan("sampler", node, steps, notes=notes, data={"run": run_id, "code": sha, "keep": bool(keep)},
                     cleanup=tuple(cleanup), artifacts=(("node", base),))


def container_name(run_id: str) -> str:
    """The sampler smoke's container (removed whatever the smoke did)."""
    return f"yeto-adopt-{run_id}"


# --- run ------------------------------------------------------------------------------


def events(text: str) -> list[dict[str, Any]]:
    """The trainer's event records in a launch's output (``YETO_RL_EVENT`` lines,
    :mod:`yeto.rl.event_echo`)."""
    from yeto.rl.event_echo import INVALID, parse_line  # noqa: PLC0415

    out = []
    for line in (text or "").splitlines():
        raw = parse_line(line)
        if raw is None or raw is INVALID:
            continue
        try:
            out.append(json.loads(raw))
        except ValueError:
            continue
    return out


def _store_keys(objects: Any) -> list[str]:
    return list(objects.list(""))


def check_round_cut(text: str, objects: Any) -> tuple[str | None, str]:
    """``(cut_id, why)`` of the train step: the newest ``rl_round_cut`` event that wrote
    a cut, and its manifest in the store (``round-cuts/<cut_id>/manifest.json``);
    cut_id None: ``why`` says what is missing."""
    cuts = [e for e in events(text) if e.get("event") == "rl_round_cut" and e.get("ok") and e.get("cut_id")]
    if not cuts:
        return None, "no rl_round_cut event with a cut_id in the launch's output"
    cut_id = str(cuts[-1]["cut_id"])
    if objects is None:
        return None, "the smoke store has no object API to read the cut's manifest from"
    want = f"round-cuts/{cut_id}/manifest.json"
    if not any(key == want or key.endswith("/" + want) for key in _store_keys(objects)):
        return None, f"the store has no {want} for the cut {cut_id} the trainer reported"
    return cut_id, ""


def check_resume(text: str, cut_id: str | None) -> str:
    """Why the resume step did not resume from ``cut_id`` (empty: it did): an
    ``rl_resume`` or ``rl_round_cut_restored`` event naming that cut."""
    found = [e for e in events(text) if e.get("event") in ("rl_resume", "rl_round_cut_restored")]
    if not found:
        return "no rl_resume / rl_round_cut_restored event in the launch's output"
    if not any(str(e.get("cut_id")) == str(cut_id) for e in found):
        return f"it resumed from {sorted({str(e.get('cut_id')) for e in found})}, not the cut {cut_id} of the train"
    return ""


def run_plan(plan: SmokePlan, *, ops: Any, ref: str, python: str | None,
             run: Callable[..., Any] = subprocess.run, clock: Callable[[], float] = time.time,
             log: Callable[[str], None] = lambda m: print(m, file=sys.stderr, flush=True),
             objects_for: Callable[[str], Any] | None = None, environ: Mapping[str, str] | None = None):
    """Every step of ``plan`` in order, then (whatever happened: a failure, a timeout)
    its cleanup; the smoke's ``StepResult`` (``ok`` False names the step and the
    output's tail). ``objects_for``: the store's object API of a store URI (default:
    the launcher's, :func:`yeto.launch.island_lifecycle.store_objects`)."""
    from yeto.node.smoke import parse_result  # noqa: PLC0415

    t_start = clock()
    times: dict[str, float] = {}
    data: dict[str, Any] = {**plan.data, "steps": times}
    python = python or "python3"
    if not PYTHON_PATH.fullmatch(python):
        return StepResult("smoke", False, f"the image's interpreter {python!r} is not a plain absolute path "
                                          "(/[\\w./+-]+): the smoke does not run it", 0.0, data)
    if objects_for is None:
        from .island_lifecycle import store_objects  # noqa: PLC0415

        objects_for = store_objects
    values = {REF: ref, PYTHON: python}
    for path, text in plan.files.items():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text)

    def failed(step: SmokeStep, why: str) -> Any:
        return StepResult("smoke", False, f"{step.name}: {why}", clock() - t_start, data)

    def go(step: SmokeStep) -> tuple[Any, str]:
        if step.where == "node":
            done = ops.run(plan.node, step.script, timeout=step.timeout_s)
        else:
            done = run(list(step.argv), capture_output=True, text=True, timeout=step.timeout_s, check=False)
        return done, (getattr(done, "stdout", "") or "") + "\n" + (getattr(done, "stderr", "") or "")

    try:
        for step in plan.steps:
            step = step.render(values, environ)
            log(f"[image adopt] smoke {step.name} [{step.where}]")
            t0 = clock()
            try:
                done, out = go(step)
            except Exception as error:  # noqa: BLE001 - a timeout or a dead ssh is the step's failure
                times[step.name] = round(clock() - t0, 1)
                return failed(step, f"{type(error).__name__}: {error}"[:600])
            times[step.name] = round(clock() - t0, 1)
            result = parse_result(out) if plan.role == "sampler" and step.name == "engine" else None
            if result is not None and not result.get("ok"):
                data["engine_steps"] = result.get("seconds")
                return failed(step, f"{result.get('step')} failed: {result.get('error')}")
            space = re.search(rf"{SPACE_MARK} (.*)", out)
            if space:
                return failed(step, space.group(1).strip())
            if done.returncode != 0:
                return failed(step, f"rc {done.returncode}: {out.strip()[-600:]}")
            if step.expect and not re.search(step.expect, out):
                return failed(step, f"the output has no {step.expect!r}: {out.strip()[-400:]}")
            if step.check == "round_cut":
                cut_id, why = check_round_cut(out, objects_for(str(plan.data.get("store"))))
                if cut_id is None:
                    return failed(step, f"{why}: {out.strip()[-300:]}")
                data["cut_id"] = cut_id
            elif step.check == "resume":
                why = check_resume(out, data.get("cut_id"))
                if why:
                    return failed(step, f"{why}: {out.strip()[-300:]}")
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
    finally:
        data["cleanup"] = clean(plan, ops=ops, run=run, log=log, objects_for=objects_for, go=go, values=values)


def clean(plan: SmokePlan, *, ops: Any, run: Callable[..., Any], log: Callable[[str], None],
          objects_for: Callable[[str], Any], go: Callable[[SmokeStep], tuple[Any, str]],
          values: Mapping[str, str]) -> dict[str, str]:
    """The plan's cleanup steps, then the launcher side of its artifacts: the smoke
    store's objects and the launcher's files (deleted, or with ``keep`` named with
    their size). A cleanup that fails is a warning: the smoke's answer stands."""
    import shutil  # noqa: PLC0415

    out: dict[str, str] = {}
    for step in plan.cleanup:
        step = step.render(values, {})
        try:
            done, text = go(step)
            out[step.name] = "ok" if done.returncode == 0 else f"rc {done.returncode}"
            if step.name == "keep" and text.strip():
                for line in text.strip().splitlines():
                    log(f"[image adopt] kept on {plan.node.name}: {line.strip()}")
        except Exception as error:  # noqa: BLE001 - never hides the smoke's answer
            out[step.name] = f"{type(error).__name__}: {error}"[:200]
            log(f"[image adopt] WARNING: smoke cleanup {step.name} failed: {out[step.name]}")
    keep = bool(plan.data.get("keep"))
    for where, path in plan.artifacts:
        try:
            if where == "store":
                objects = objects_for(path)
                keys = _store_keys(objects) if objects is not None else []
                if keep:
                    log(f"[image adopt] kept in the store: {path} ({len(keys)} object(s))")
                else:
                    for key in keys:
                        objects.delete(key)
                    out["store"] = f"{len(keys)} object(s) deleted"
            elif where == "launcher":
                target = Path(path)
                if keep:
                    size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) if target.exists() else 0
                    log(f"[image adopt] kept on the launcher: {target} ({size / 2**20:.1f} MiB)")
                else:
                    shutil.rmtree(target, ignore_errors=True)
                    out["launcher"] = "deleted"
        except Exception as error:  # noqa: BLE001
            out[where] = f"{type(error).__name__}: {error}"[:200]
            log(f"[image adopt] WARNING: smoke cleanup of {where} {path} failed: {out[where]}")
    return out

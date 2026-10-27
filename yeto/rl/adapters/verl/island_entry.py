"""verl island process (``python3 -m yeto.rl.adapters.verl.island_entry``).

The launcher runs this instead of the Miles island entry when
``--rl-backend verl`` (``launch_flags.ISLAND_ENTRY_MODULE``), with the same
command-line flags; only the flags verl needs are read, the Miles-only ones are
ignored.  Steps:

1. runtime check: verl commit and library versions against ``pins``; runtime
   manifest + pip freeze written next to the tape;
2. model snapshot at the pinned revision; dataset at the pinned revision,
   converted to verl's parquet schema (prompt = chat messages, ground truth);
3. algorithm spec -> verl options (only mapped fields; others refused);
4. ``VerlRunConfig`` checked (``config.validate``) -> hydra overrides;
5. ``verl_main`` with the island plan; its output is tee'd into
   ``~/yeto-output/verl-train-<id>.log``.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from yeto.rl.quant.tiers import TIER_TRAINER_FP8

from . import config as vconf
from . import megatron_engine
from . import run_config_rules
from .pins import VERL_COMMIT, expected_versions_for_engine, verl_root
from .reward_fn import function_for

OUTPUT = Path(os.path.expanduser("~/yeto-output"))
# Bring-up only: "1" turns version/commit pin mismatches into a recorded
# warning (manifest "problems_waived") instead of exit 2. Never set in production.
ALLOW_UNPINNED_ENV = "YETO_VERL_ALLOW_UNPINNED"
TEST_FILE = Path(os.path.expanduser("~/.yeto-verl-test.json"))  # TEST ONLY switches


def parse_args(argv):
    p = argparse.ArgumentParser(prog="yeto.rl.adapters.verl.island_entry")
    p.add_argument("--model", required=True)
    p.add_argument("--model-revision", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--data-revision", default=None)
    p.add_argument("--syncer", default=None)
    p.add_argument("--rl-single-island-no-sync", action="store_true")
    p.add_argument("--learner-id", type=int, required=True)
    p.add_argument("--reward-function", required=True)
    p.add_argument("--global-rounds", type=int, required=True)
    p.add_argument("--groups-per-round", type=int, required=True)
    p.add_argument("--samples-per-group", type=int, required=True)
    p.add_argument("--rollout-max-response-len", type=int, required=True)
    p.add_argument("--event-tape", required=True)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-targets", default="all-linear")
    p.add_argument("--inner-lr", type=float, required=True)
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--wan-streams", type=int, default=4)
    # S26 E3: bf16 downlink deltas (yeto/downlink.py; the syncer runs --bcast-delta-full-every)
    p.add_argument("--bcast-delta", action="store_true")
    p.add_argument("--apply-chat-template-kwargs", default=None)
    # S27 G-EQ2: the launcher passes this to every island; the verl island used to
    # ignore it, so a model whose HF support ships with the checkpoint could not load
    p.add_argument("--trust-remote-code", action="store_true")
    # S27 G-EQ2: FSDP2 host-memory offload (npu1: 64 GiB cards, 30B full parameters)
    p.add_argument("--rl-fsdp-offload", choices=("none", "param", "optimizer", "both"),
                   default="none")
    p.add_argument("--rl-algorithm-spec", default=None)
    p.add_argument("--rl-island-pool", action="store_true")
    p.add_argument("--rl-syncer-epoch", type=int, default=0)
    p.add_argument("--rl-sync-island-lease-s", type=float, default=None)
    p.add_argument("--rl-echo-events", action="store_true")
    p.add_argument("--rl-max-policy-age", type=int, default=0)
    # S27 G-EV (VERL-PARITY #10): the separate-mode island generates one batch ahead
    p.add_argument("--rl-rollout-ahead", type=int, default=0)
    # S25 C2: a trainer island over the sample buffer (sample_roles; the sampler is
    # the engine sampler island, yeto.rl.engine.engine_sampler). --rl-resume-store
    # names the buffer's store there, and the round-cut store otherwise (N7.6b)
    p.add_argument("--rl-sample-role", default=None)
    p.add_argument("--rl-resume-store", default=None)
    p.add_argument("--rl-sample-wait-s", type=float, default=None)
    # S27 G-EQ3: staged bf16 delta releases on a full-parameter buffer trainer
    # (sample_roles._check_release_delta; 0 turns them off)
    p.add_argument("--rl-release-delta-full-every", type=int, default=0)
    # S28 G-120C: a memory / MFU measurement run on a buffer trainer writes no policy
    # release (one 120B release is about 225 GB and a dozen minutes per round)
    p.add_argument("--rl-measure-no-release", action="store_true")
    # S28 G-120C: release only these policy versions (comma-separated), e.g. "2": a
    # multi-step run trains on the samples it has and releases once for the samplers
    p.add_argument("--rl-release-only-versions", default=None,
                   type=lambda text: [int(v) for v in text.split(",") if v.strip()])
    # S26 E9.4: the pair flags the launcher forwards per island (sample_roles.island_flags);
    # a sampler island of a verl launch runs the engine sampler (.sampler_entry)
    p.add_argument("--rl-sampler-node", default=None)
    p.add_argument("--rl-buffer-shards", default=None, type=lambda text: [s for s in text.split(",") if s])
    p.add_argument("--rl-sampler-data-slot", default=None)
    # S27 P4: a sampler island claims its rows from the task dispenser (the launcher
    # sets these; yeto.rl.engine.task_dispense.DispensedPrompts)
    p.add_argument("--rl-data-dispense", action="store_true", default=False)
    p.add_argument("--rl-dispense-addr", default=None)
    p.add_argument("--rl-dispense-prefetch-blocks", type=int, default=2)
    p.add_argument("--rl-dispense-mix", default=None)
    p.add_argument("--rl-dispense-by-label", action="store_true", default=False)
    p.add_argument("--rl-dispense-labels", default=None)
    # S27 G-FS (VERL-PARITY #33): the rollout runs Harbor trials; the rows select
    # yeto's agent loop (``agent_name: yeto_harbor``).
    p.add_argument("--rl-agentic-harbor", action="store_true", default=False)
    # S27 task assignment: this training island's row slot of --data (data_split)
    p.add_argument("--rl-data-slot", default=None)
    p.add_argument("--rl-sampler-max-backlog-rounds", type=int, default=None)
    p.add_argument("--rl-sampler-min-round-seconds", type=float, default=None)
    # S25 C3: a synced trainer island (syncer, learner m of N) reads lane m of the
    # buffer in this store; the launcher forwards both (yeto.rl.engine.merged_release)
    p.add_argument("--rl-sample-store", default=None)
    p.add_argument("--rl-sample-lanes", type=int, default=None)
    # S27 G-EV: a relay trainer (spot samplers through the head's sample relay,
    # .relay_trainer) and the layout board for the spot subscribers (island 0)
    p.add_argument("--rl-sample-relay", action="store_true")
    p.add_argument("--rl-sample-relay-prefetch", type=int, default=None)
    p.add_argument("--rl-spot-subscribers", type=int, default=0)
    # N7.6b round cuts (same flags as the Miles island)
    p.add_argument("--rl-resume-state-dir", default=None)
    p.add_argument("--rl-cut-every", type=int, default=None)
    p.add_argument("--rl-cut-keep", type=int, default=None)
    p.add_argument("--rl-stop-after-rounds", type=int, default=None)
    p.add_argument("--rl-resume-allow-config-change", action="store_true")
    # S27 G-ED: island cuts of a synced island (S27 R-2) and the timed cut interval
    p.add_argument("--rl-island-cut-store", default=None)
    p.add_argument("--rl-cut-auto-mtbf-s", type=float, default=None)
    p.add_argument("--rl-cut-max-interval-s", type=float, default=None)
    # S26 E6: rollout quantization tier (yeto.rl.quant.tiers; the launcher validated it)
    p.add_argument("--rl-rollout-quant-tier", type=int, choices=(0, 1, 2, 3), default=0)
    # S27 G-DC: the Megatron engine island (megatron_engine; EP/TP/PP > 1 select it);
    # --parameter-mode comes from add_island_flags below (choices lora/full, default lora)
    p.add_argument("--tensor-parallel", type=int, default=1)
    p.add_argument("--pipeline-parallel", type=int, default=1)
    p.add_argument("--expert-parallel", type=int, default=None)
    # S28 G-CPW: the launcher's neutral --context-parallel (rl_island_task forwards it
    # when > 1); parse_known_args used to drop it, so the island trained at CP=1
    p.add_argument("--context-parallel", type=int, default=1)
    p.add_argument("--actor-num-nodes", type=int, default=1)
    p.add_argument("--actor-num-gpus-per-node", type=int, default=None)
    p.add_argument("--rl-island-gpus-per-node", type=int, default=None)
    p.add_argument("--rl-placement", default=None)
    p.add_argument("--rollout-num-gpus", type=int, default=None)
    p.add_argument("--rollout-num-gpus-per-engine", type=int, default=1)
    p.add_argument("--sglang-mem-fraction-static", type=float, default=None)
    p.add_argument("--rl-max-tokens-per-gpu", type=int, default=None)
    # S27 G-FL: the Ascend recipe's plugin stack (yeto.rl.ascend_plugin_stack), Megatron only
    p.add_argument("--rl-ascend-plugin-stack", default=None)
    # S27 G-FL: Megatron's per-bucket non-finite gradient check is on unless this says otherwise
    p.add_argument("--rl-no-grad-nan-check", action="store_true")
    # S27 G-FN: host share of the Megatron optimizer state (the engine-neutral flag the
    # launcher already forwards; megatron_engine.optimizer_memory_overrides maps it)
    p.add_argument("--rl-optimizer-offload-fraction", type=float, default=0.0)
    # S28 G-RCP: precision-aware optimizer on the card, bf16 Adam moments, FP32 master
    p.add_argument("--rl-megatron-bf16-moments", action="store_true")
    # S28 G-120: uneven pipeline stages and full-layer recompute (megatron_engine.MegatronShape)
    p.add_argument("--rl-megatron-first-stage-layers", type=int, default=None)
    p.add_argument("--rl-megatron-last-stage-layers", type=int, default=None)
    p.add_argument("--rl-megatron-recompute-full-layers", type=int, default=None)
    # S28 G-120C: verl's fused log-prob kernels. Off unless asked for: on this model
    # they are 47.8% of the activation per token, and the Ascend blockers are in
    # infra-drafts/SESSION28/report/FCE-PLAN.md.
    p.add_argument("--rl-megatron-fused-logprob-kernels", action="store_true")
    # S27 G-FL: PP > 1 in a full-parameter session is refused unless this lifts the gate
    p.add_argument("--rl-allow-pipeline-parallel-sync", action="store_true")
    p.add_argument("--use-rollout-routing-replay", action="store_true")
    p.add_argument("--rl-lr-schedule", default=None)
    p.add_argument("--rl-export-hf-rounds", default=None)
    p.add_argument("--rl-export-hf-store", default=None)
    # S27 B10: a full-parameter island and its outer sync (Miles island spellings)
    from .full_parameter_stream import add_island_flags

    add_island_flags(p)
    # S27 G-EN: the Miles island's heldout eval flags (VERL-PARITY #34, .eval_port)
    from .eval_port import add_island_flags as add_eval_flags

    add_eval_flags(p)
    args, ignored = p.parse_known_args(argv)
    return args, ignored


def algorithm_options(spec_dict: dict | None) -> dict:
    """Mapped AlgorithmSpec fields -> verl options; anything unmapped is refused (D11)."""
    if not spec_dict:
        return {"correction": "none", "tis_upper": 2.0}
    from yeto.rl.engine.algorithm import AlgorithmSpec

    spec = AlgorithmSpec.from_dict(spec_dict)
    problems = []
    if spec.advantage_estimator != "grpo":
        problems.append(f"advantage estimator {spec.advantage_estimator!r}")
    loss = spec.loss
    if loss.variant != "policy_loss" or loss.aggregation != "default" or loss.reducer or loss.custom_loss \
            or loss.eps_clip_c is not None:
        problems.append(f"loss {loss}")
    if spec.kl.placement != "none" or spec.kl.coef:
        problems.append(f"kl {spec.kl}")
    if spec.entropy_coef:
        problems.append("entropy_coef")
    if spec.dynamic_sampling_filter or spec.plugins:
        problems.append("dynamic sampling filter / plugins")
    if getattr(spec, "critic", None) is not None and getattr(spec.execution, "needs_critic", False):
        problems.append("critic")
    correction = spec.correction
    method = (correction.method or "none") if correction is not None else "none"
    out = {"correction": method, "tis_upper": 2.0,
           # S27 G-DC: mapped by the Megatron island (megatron_engine.build_overrides)
           "eps_clip": loss.eps_clip, "eps_clip_high": loss.eps_clip_high}
    if method == "tis":
        out["tis_upper"] = float(correction.tis_clip if correction.tis_clip is not None else 2.0)
        out["tis_lower"] = float(correction.tis_clip_low or 0.0)
    if problems:
        raise vconf.VerlConfigError("verl 后端未映射的算法字段: " + ", ".join(problems))
    return out


def runtime_manifest(learner_id: int, device_family: str | None = None, engine: str = "fsdp2") -> dict:
    """Versions this island actually runs, with the per-family expected pins.

    ``device_family`` None reads the family from this node's accelerator, so an
    Ascend island asserts the NPU pins and runs ``npu-smi`` instead of
    ``nvidia-smi`` (tasks 3.6, 3.14). The NVIDIA manifest keeps its old shape
    and its ``nvidia_smi`` key.
    """
    import importlib.metadata as md

    from yeto.hw.catalog import device_family as detect_device_family
    from yeto.hw.catalog import runtime_versions

    family = device_family if device_family is not None else detect_device_family()
    # S27 G-DC: the Megatron engine island also asserts the Megatron stack (pins)
    expected = expected_versions_for_engine(engine, family)
    versions = {}
    for name in expected:
        try:
            versions[name] = md.version(name.replace("_", "-"))
        except md.PackageNotFoundError:
            versions[name] = None
    try:
        import torch

        versions["torch"] = torch.__version__
    except ImportError:
        pass
    if family == "ascend":
        try:
            import torch_npu

            versions["torch_npu"] = getattr(torch_npu, "__version__", versions.get("torch_npu"))
        except ImportError:
            pass
    sha_file = Path("/workspace/verl_sha.txt")
    commit = sha_file.read_text().strip() if sha_file.is_file() else None
    smi_cmd = ("npu-smi info" if family == "ascend" else
               "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader")
    smi = subprocess.run(smi_cmd, shell=True, capture_output=True, text=True).stdout.strip()
    smi_key = "npu_smi" if family == "ascend" else "nvidia_smi"
    manifest = {"schema": "yeto-verl-runtime-v1", "verl_commit": commit, "versions": versions,
                "expected_versions": expected, smi_key: smi, "learner_id": learner_id,
                "device_family": family, "python": sys.version.split()[0]}
    if family == "ascend":
        manifest["nvidia_smi"] = ""  # keeps the manifest shape stable for readers
        manifest.update({k: v for k, v in runtime_versions(family).items()
                         if k in ("npu_driver_version", "cann_version")})
    problems = []
    if commit != VERL_COMMIT:
        problems.append(f"verl commit {commit} != pin {VERL_COMMIT}")
    for name, want in expected.items():
        if versions.get(name) != want:
            problems.append(f"{name} {versions.get(name)} != {want}")
    manifest["problems"] = problems
    return manifest


def _threshold_card(manifest: dict) -> str:
    """Card name for the numeric threshold table key.

    NPU and GPU thresholds are separate rows: the NPU island has no
    ``nvidia-smi`` output, so the key comes from ``npu-smi`` instead. The CUDA
    behaviour (H100 shorthand, else the raw smi line) is unchanged.
    """
    if manifest.get("device_family") == "ascend":
        from yeto.gpu_spec import parse_npu_smi_names

        names = parse_npu_smi_names(manifest.get("npu_smi") or "")
        return names[0] if names else (manifest.get("npu_smi") or "unknown-npu")
    smi = manifest.get("nvidia_smi") or ""
    return "H100" if "H100" in smi else smi


def agentic_rows(frame: Any) -> bool:
    """True when the data file already holds verl agentic rows (S27 G-FS).

    ``task_lists.verl_rows`` writes the format verl's agent loop reads:
    ``agent_name`` selects the loop and ``extra_info.agent_task`` is the
    :class:`AgentTask` mapping.  Such rows carry no ``messages`` / ``label``, so
    the math conversion below would fail on them; they pass through unchanged.
    """
    columns = set(getattr(frame, "columns", ()))
    return "agent_name" in columns and "extra_info" in columns and "messages" not in columns


def prepare_data(data: str, revision: str | None, root: Path, *, val_rows: int = 64,
                 slot: Any = None, number_rows: bool = False) -> tuple[str, str]:
    """``data``'s train and test splits as verl parquet files. ``slot``
    (``data_split.DataSlot``, S27 task assignment): the train file keeps only the
    slot's rows, in data order, each with its row number in the unsplit data as
    ``extra_info["yeto_data_row"]``; the validation rows stay whole.
    ``number_rows`` (the task dispenser, S27 G-EU): every train row keeps its row
    number, and ``root/train-dispense.jsonl`` (``task_dispense.write_rows_file``)
    is written for the dispenser's row count and fingerprint.
    S27 G-FS: a file of agentic rows (:func:`agentic_rows`) is written through
    unchanged -- the rows are already in verl's schema."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    from yeto.rl.engine.data_split import DATA_ROW_KEY

    root.mkdir(parents=True, exist_ok=True)
    out = {}
    local = Path(os.path.expanduser(data))
    for split in ("train", "test"):
        if local.is_file():  # S27 G-DC: a local JSONL task file (one file: train; its head: test)
            # parsed with json, not pd.read_json: pandas would turn a label "34" into 34.0
            frame = pd.DataFrame([json.loads(line) for line in local.read_text().splitlines() if line.strip()])
        else:
            src = hf_hub_download(data, f"{split}.parquet", repo_type="dataset", revision=revision)
            frame = pd.read_parquet(src)
        if agentic_rows(frame):  # S27 G-FS: already verl's schema, keep every field
            rows = [{**row, "extra_info": {**(row["extra_info"] or {}), "split": split, "index": int(i)}}
                    for i, row in frame.iterrows()]
        else:
            rows = [{
                "data_source": "yeto/" + data,
                "prompt": [{"role": m["role"], "content": m["content"]} for m in row["messages"]],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": str(row["label"])},
                "extra_info": {"split": split, "index": int(i)},
            } for i, row in frame.iterrows()]
        if split == "test":
            rows = rows[:val_rows]
        elif number_rows:  # S27 G-EU: the dispenser hands out rows by number
            if slot is not None and not slot.whole:
                raise ValueError("a data slot and the task dispenser exclude each other")
            from .task_dispense import ROWS_FILE, write_rows_file

            write_rows_file(root / ROWS_FILE, (row for _, row in frame.iterrows()))
            rows = [{**row, "extra_info": {**row["extra_info"], DATA_ROW_KEY: number}}
                    for number, row in enumerate(rows)]
        elif slot is not None and not slot.whole:
            rows = [{**row, "extra_info": {**row["extra_info"], DATA_ROW_KEY: number}}
                    for number, row in slot.numbered(rows)]
        path = root / f"{split}.parquet"
        pd.DataFrame(rows).to_parquet(path)
        out[split] = str(path)
    return out["train"], out["test"]


# --------------------------------------------------------------------------- agentic (S27 G-FS)
#
# VERL-PARITY #33. The pieces already existed and are not rewritten here:
# ``harbor_agent_loop.HarborAgentLoop`` (the loop), ``rollout_engine`` (the records
# proxy over verl's own rollout servers) and ``task_lists.verl_rows`` (the row
# format). What was missing is this island's wiring: the loop's registry entry,
# the estimator, the row pass-through and the recorder environment.
AGENT_LOOP_YAML = "agent-loop.yaml"
# yeto's agent loop class, by name: importing ``harbor_agent_loop`` here would
# import verl (its registry decorator), which the island entry must not do before
# the runtime check has run. A test keeps the spelling equal to the class.
AGENT_LOOP_TARGET = "yeto.rl.adapters.verl.harbor_agent_loop.HarborAgentLoop"


def agentic_island(args: Any) -> bool:
    """``--rl-agentic-harbor``: the rollout runs Harbor trials (``harness_env.is_harbor_agent``)."""
    return bool(getattr(args, "rl_agentic_harbor", False))


def agentic_run_fields(args: Any, *, out_dir: str) -> dict:
    """``VerlRunConfig`` fields of an agentic island ({} when it is not one).

    Writes the agent-loop registry file verl reads
    (``rollout.agent.agent_loop_config_path``): one entry whose ``name`` is the
    ``agent_name`` the rows carry and whose ``_target_`` is yeto's loop class.
    """
    if not agentic_island(args):
        return {}
    from yeto.rl.harness.harbor.task_lists import VERL_AGENT_LOOP

    path = Path(out_dir) / AGENT_LOOP_YAML
    path.parent.mkdir(parents=True, exist_ok=True)
    # verl reads the file with OmegaConf and *iterates* it, so the top level is a
    # LIST of entries, each a mapping with ``name`` and ``_target_``
    # (agent_loop.py:554-556: ``for c in configs: registry[c.name] = c``).  A
    # top-level mapping iterates over its keys and fails with
    # "'str' object has no attribute 'name'" -- S27 G-FS run s27-fs-2 died there.
    # Written by hand: two scalar fields, and the island image guarantees no yaml dumper.
    path.write_text(f"- name: {VERL_AGENT_LOOP}\n  _target_: {AGENT_LOOP_TARGET}\n")
    return {"agentic": True, "agent_loop_config": str(path),
            "agent_loop_workers": int(os.environ.get("YETO_VERL_AGENT_LOOP_WORKERS", "1"))}


def _agent_task_model_name() -> str:
    """``AgentTask.model_name``'s default: the name the agent sends when a row sets none.

    ``task_lists.verl_rows`` deliberately keeps the model name out of the rows
    (it is a deployment fact), so every sample uses this default.  A row that does
    set ``model_name`` must be launched with a matching ``YETO_RECORDS_MODEL``.
    """
    from dataclasses import fields

    from yeto.rl.agentic import AgentTask

    field = next(f for f in fields(AgentTask) if f.name == "model_name")
    return str(field.default)


def agentic_records_env(run: Any, *, model_path: str, card: str) -> dict[str, str]:
    """Records-proxy environment of the agent-loop workers (``rollout_engine`` module doc).

    The agent talks OpenAI to the proxy; the proxy renders the chat template and
    asks verl's own rollout servers for tokens.  The sampling it signs is the
    island's (temperature / top_p / top_k and the response cap), so a reply the
    proxy accepts was sampled the way the training step assumes.
    ``YETO_RECORDS_PROXY_PORT`` and ``..._PUBLIC_BASE`` are **not** set here: the
    island's rank 0 gets them from its Modal tunnel
    (``modal_runner.harbor_relay_tunnel``), and a value set here would win over it.
    """
    from yeto.rl.harness.chat_templates import TEMPLATES
    from yeto.rl.harness.gateway import records_env

    # The template is a property of the checkpoint, so the launch names it
    # (--modal-env YETO_RECORDS_TEMPLATE=...); there is no safe default.
    template = os.environ.get(records_env.TEMPLATE_ENV, "").strip()
    if template not in TEMPLATES:
        raise SystemExit(f"agentic verl 岛要 {records_env.TEMPLATE_ENV}=<chat template>"
                         f"（已知：{sorted(TEMPLATES)}），收到 {template!r}")
    sampling = {"temperature": run.temperature, "top_p": run.top_p, "top_k": run.top_k,
                "max_tokens": run.max_response_len}
    return {
        records_env.TEMPLATE_ENV: template,
        "YETO_AGENTIC_RECORDER": "yeto.rl.harness.gateway.recorder:recorder_from_env",
        records_env.PROXY_ENV: "yeto.rl.adapters.verl.rollout_engine:records_proxy_from_env",
        records_env.TOKENIZER_ENV: str(model_path),
        # The model name the proxy accepts must equal the one the agent sends, and
        # the agent sends ``AgentTask.model_name`` (Harbor's AgentConfig gets
        # ``openai/<model_name>``, ``harbor.trial.build_trial_config``). The gateway
        # refuses any other name outright ("model is fixed by the gateway",
        # ``gateway/core.py:116-117``), so this is taken from the dataclass default
        # rather than guessed from the checkpoint path: a plausible-looking
        # checkpoint name would refuse every model call of the run.
        records_env.MODEL_ENV: os.environ.get(records_env.MODEL_ENV) or _agent_task_model_name(),
        records_env.SAMPLING_ENV: json.dumps(sampling, sort_keys=True),
        records_env.MAX_MODEL_LEN_ENV: str(run.max_model_len or run.max_prompt_len + run.max_response_len),
        records_env.CARD_ENV: card,
        # The proxy binds every interface: the Modal tunnel reaches it from outside
        # the container, and 127.0.0.1 would refuse that connection.
        records_env.PROXY_HOST_ENV: os.environ.get(records_env.PROXY_HOST_ENV, "0.0.0.0"),
    }


def island_data_slot(args: Any) -> Any:
    """The training island's ``--rl-data-slot`` (``data_split.DataSlot``), None when
    it reads every row; refused on a sampler island (it reads
    ``--rl-sampler-data-slot``)."""
    from yeto.rl.engine import sample_roles
    from yeto.rl.engine.data_split import DataSlot

    if not getattr(args, "rl_data_slot", None):
        return None
    if sample_roles.sample_role(args) != sample_roles.BOTH:
        raise SystemExit("--rl-data-slot belongs to a training island that generates and trains")
    return DataSlot.parse(args.rl_data_slot)


MIN_RAY_CPUS = 32  # logical scheduling slots verl needs on one GPU (TQ storage units, agent/reward workers, ...)


def _ray_resources(address: str) -> dict:
    code = ("import json, ray; ray.init(address=%r, logging_level='ERROR'); "
            "print('YETO_RAY_RES ' + json.dumps(ray.cluster_resources())); ray.shutdown()" % address)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180).stdout
    for line in out.splitlines():
        if line.startswith("YETO_RAY_RES "):
            return json.loads(line[len("YETO_RAY_RES "):])
    return {}


def island_gpus(ray_info: Mapping[str, Any], *, device_family: str | None = None) -> int:
    """Cards of this single-node island: the island-local Ray's accelerator
    resource (the launcher starts it with every card of the container), else 1.

    S26 E9 G3: the sync island used ``n_gpus=1`` always, so a 2-card container
    trained on one rank (e9-g3-a); every card now joins the FSDP2 group.

    S27 G-EQ2 (npu1): the resource is named ``GPU`` on CUDA and ``NPU`` on
    Ascend (``catalog.ray_accelerator_resource``), so an 8-card Ascend island
    used to read 1 here and train on one rank. Either key is read, so a node
    whose Ray reports the other name still gives its cards."""
    from yeto.hw.catalog import ray_accelerator_resource

    resources = ray_info.get("after") or ray_info.get("before") or {}
    resource = ray_accelerator_resource(device_family)
    count = int(resources.get(resource, 0))
    if not count:  # the other family's spelling, when this node's family was read wrong
        count = max(int(resources.get(name, 0)) for name in ("GPU", "NPU"))
    return max(1, count)


def ensure_ray_cpus(min_cpus: int = MIN_RAY_CPUS) -> dict:
    """verl needs more Ray CPU slots than a small container reports.

    S17 V2 a: the island-local Ray the launcher starts saw 8 CPUs on one Modal host
    and 24 on another; with 8, verl's transfer-queue storage units took every slot and
    the trainer workers never scheduled (GPU 0/1 used, no error).  Ray CPUs are only
    scheduling tokens here, so when the cluster reports fewer than ``min_cpus`` the
    island-local Ray (the launcher's, same temp dir, single node) is restarted with
    ``--num-cpus min_cpus``.  Only inside a Modal container (no SkyPilot runtime Ray
    there); elsewhere it is reported, not changed.
    """
    address = os.environ.get("RAY_ADDRESS")
    if not address:
        return {"skipped": "no RAY_ADDRESS"}
    before = _ray_resources(address)
    info = {"before": before}
    if before.get("CPU", 0) >= min_cpus:
        return info
    if not os.environ.get("MODAL_TASK_ID"):
        info["skipped"] = "not a Modal container; Ray left as is"
        return info
    host, _, port = address.rpartition(":")
    temp = os.path.expanduser("~/miles-ray")
    # Not ``pkill -f <temp dir>`` inside ``sh -c``: the pattern matches that shell itself (V2 b).
    # Inside a Modal container the launcher's island Ray is the only Ray on the node.
    info["stop_rc"] = subprocess.run(["ray", "stop", "--force"], timeout=180).returncode
    # S27 G-EQ2: the cards come back under this node's own resource name; on an
    # Ascend node that is --resources '{"NPU": n}', not --num-gpus (catalog).
    from yeto.hw.catalog import ray_accelerator_resource, ray_start_accelerator_flag

    resource = ray_accelerator_resource()
    gpus = int(before.get(resource, 0)) or max(int(before.get(name, 0)) for name in ("GPU", "NPU"))
    cmd = (f'ray start --head --node-ip-address={host} --port={port} --num-cpus={min_cpus} '
           f'{ray_start_accelerator_flag(gpus)} --include-dashboard=false --temp-dir={temp}')
    info["restart"] = cmd
    info["restart_rc"] = subprocess.run(cmd, shell=True).returncode
    info["after"] = _ray_resources(address)
    return info


def main(argv=None) -> int:
    args, ignored = parse_args(sys.argv[1:] if argv is None else argv)
    run_config_rules.refuse_agentic_argv(ignored)  # S25 N10.4: never ignore agentic flags
    buffer_section = sample_buffer_section(args)  # S25 C2: refused roles stop before any work
    spot_env = spot_rollout_env(args, buffer_section)  # S27 G-EV: relay trainer / layout board
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lid = args.learner_id
    t0 = time.time()
    engine = megatron_engine.engine_for(tensor_parallel=args.tensor_parallel,
                                        pipeline_parallel=args.pipeline_parallel,
                                        expert_parallel=args.expert_parallel or 1,
                                        context_parallel=args.context_parallel)
    manifest = runtime_manifest(lid, engine=engine)
    manifest["ignored_flags"] = ignored
    (OUTPUT / f"verl-runtime-{lid}.json").write_text(json.dumps(manifest, indent=1))
    subprocess.run(f"uv pip freeze --python {sys.executable} > {OUTPUT}/verl-pip-freeze-{lid}.txt 2>/dev/null"
                   f" || {sys.executable} -m pip freeze > {OUTPUT}/verl-pip-freeze-{lid}.txt 2>/dev/null",
                   shell=True)
    print(f"[yeto-verl] island {lid} runtime {json.dumps(manifest)}", flush=True)
    if manifest["problems"] and os.environ.get(ALLOW_UNPINNED_ENV) == "1":
        # Bring-up only (new card, vendor image): run anyway, keep the evidence.
        manifest["problems_waived"] = manifest.pop("problems")
        manifest["problems"] = []
        (OUTPUT / f"verl-runtime-{lid}.json").write_text(json.dumps(manifest, indent=1))
        print(f"[yeto-verl] runtime check WAIVED by {ALLOW_UNPINNED_ENV}=1: "
              f"{manifest['problems_waived']}", file=sys.stderr, flush=True)
    if manifest["problems"]:
        print(f"[yeto-verl] runtime check failed: {manifest['problems']}", file=sys.stderr, flush=True)
        return 2
    from yeto.rl.engine import sample_roles

    if sample_roles.sample_role(args) == sample_roles.SAMPLER:  # S26 E9.4: no verl trainer here
        from .sampler_entry import SamplerIslandError, run_sampler_island

        try:
            return run_sampler_island(args, manifest=manifest, card=_threshold_card(manifest), output=OUTPUT)
        except SamplerIslandError as error:  # the island flags are incomplete
            raise SystemExit(str(error)) from None

    ray_info = ensure_ray_cpus()
    (OUTPUT / f"verl-ray-{lid}.json").write_text(json.dumps(ray_info, indent=1))
    print(f"[yeto-verl] island {lid} ray {json.dumps(ray_info)}", flush=True)
    if ray_info.get("after") is not None and ray_info["after"].get("CPU", 0) < MIN_RAY_CPUS:
        print("[yeto-verl] Ray restart did not give enough CPU slots", file=sys.stderr, flush=True)
        return 3

    from .task_dispense import DispenseIslandError, dispense_section, island_dispense
    from .train_options import debug_plan, refuse_unmapped_argv, verify_plan

    try:  # S27 G-EU (VERL-PARITY #31, #25, #26): refused paths stop before any model or data work
        dispense = island_dispense(args, engine=engine, buffer_trainer=buffer_section is not None)
        full_mode = args.parameter_mode == vconf.FULL_PARAMETER_MODE or engine == megatron_engine.ENGINE_MEGATRON
        refuse_unmapped_argv(ignored, full=full_mode)
        verify = {**verify_plan(args, full=full_mode), **debug_plan(args)}
    except ValueError as error:
        raise SystemExit(str(error)) from None
    # S27 G-EQ2 (s27-verl-pool-profile): a directory on this node is used as is
    # (npu1's n2-n4 have no route to the hub)
    model_path = island_model_path(args.model, args.model_revision)
    work = Path(os.path.expanduser("~/yeto-verl"))
    train_file, val_file = prepare_data(args.data, args.data_revision, work / "data",
                                        slot=island_data_slot(args), number_rows=dispense)
    try:
        dispense_plan = (dispense_section(args, work / "data" / "train-dispense.jsonl",
                                          block=args.groups_per_round) if dispense else None)
    except DispenseIslandError as error:
        raise SystemExit(str(error)) from None
    spec_dict = json.loads(Path(os.path.expanduser(args.rl_algorithm_spec)).read_text()) \
        if args.rl_algorithm_spec else None
    algo = algorithm_options(spec_dict)
    chat = json.loads(args.apply_chat_template_kwargs) if args.apply_chat_template_kwargs else {}
    # 512 is enough for a single-turn data prompt (one question). On the agentic
    # path the row's prompt is only a label: what verl receives as the prompt is
    # the joined agent conversation, which is tens of thousands of tokens, so the
    # cap comes from the sequence budget as it does on the Megatron path
    # (:func:`megatron_island_plan`). S27 G-FS run s27-fs-3 hit this: with the 512
    # cap every one of 8 trajectories was ABORTED with prompt_exceeds_prompt_length
    # (the agent's own n_input_tokens were 7 585 to 64 960), the batch then had no
    # trainable token, and verl refused the training step.
    max_prompt = (max(512, int(args.seq_len) - int(args.rollout_max_response_len))
                  if agentic_island(args) else 512)
    ahead, limit = rollout_ahead_of(args)  # limit: the fully_async path's (0 with rollout ahead)
    full = args.parameter_mode == vconf.FULL_PARAMETER_MODE
    pools = None
    if engine == megatron_engine.ENGINE_MEGATRON:  # S27 G-DC: verl Megatron island
        megatron = megatron_island_plan(
            args, model_path=model_path, train_file=train_file, val_file=val_file, out_dir=str(work / "out"),
            algo=algo, chat=chat, ray_info=ray_info, reward_path=str(Path(__file__).with_name("reward_fn.py")),
            reward_name=function_for(args.reward_function),
            buffer_trainer=buffer_section is not None)  # S27 G-FN (B1-B3)
        run, overrides = megatron["run"], megatron["overrides"]
    else:
        megatron = None
        ascend_plugin_stack_section(args, engine=engine)  # S27 G-FL: refuses the flag here
        if full and args.rl_rollout_quant_tier:
            raise SystemExit("verl 全参数岛不支持 --rl-rollout-quant-tier")
        pools = fsdp_separate_pools(args, ray_info)  # S27 G-EK: None = colocated
        if pools is not None and buffer_section is not None:  # S27 G-EW: no rollout on a buffer trainer
            raise SystemExit("verl 全参数缓冲训练岛（--rl-sample-role trainer）不接 --rl-placement fixed-partition："
                             "它没有 rollout 卡，采样岛生成")
        if full and limit and buffer_section is None and pools is None:  # S27 G-EV (VERL-PARITY #11)
            raise SystemExit("verl 全参数 fully_async 岛（--rl-max-policy-age > 0）需要 --rl-placement fixed-partition "
                             "--rl-rollout-gpus N：rollouter 在自己的卡上生成")
        run, overrides = fsdp_island_run(args, model_path=model_path, train_file=train_file,
                                         val_file=val_file, out_dir=str(work / "out"), algo=algo, chat=chat,
                                         ray_info=ray_info, max_prompt=max_prompt, limit=limit, pools=pools)
    if args.rl_single_island_no_sync:
        sync = "none"
    elif args.rl_island_pool:
        sync = "pool"
    else:
        sync = "strict"
    if sync != "none" and not args.syncer:
        raise SystemExit("--syncer is required unless --rl-single-island-no-sync")
    from .round_cuts import resume_config

    try:
        resume = resume_config(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    full_plan: dict = {}
    if full and (megatron is None or sync != "none"
                 or buffer_section is not None):  # S27 B10: model identity, and the outer
        # full-parameter sync when synced (S27 G-FG: a synced Megatron island too).
        # S27 G-FN (B4): a Megatron buffer trainer runs on the fully_async ports, and
        # their identity is this section's too (``fully_async_full.full_parameter_hashes``
        # reads ``plan["model_config_sha256"]``); only the synchronous Megatron island
        # takes its identity from ``megatron_ports.config_hash`` instead.
        from yeto.rl.engine.algorithm import AlgorithmSpec

        from .full_parameter_stream import plan_section

        try:
            full_plan = plan_section(
                args, run.to_dict(), model_path=model_path, sync=sync,
                algorithm_spec_sha256=(AlgorithmSpec.from_dict(spec_dict) if spec_dict
                                       else AlgorithmSpec()).sha256())
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    asserted_keys = vconf.ASSERTED_KEYS
    store_note = None
    if megatron is not None:
        asserted_keys = megatron["asserted_keys"]  # S27 G-FN: without a rollout pool, fewer keys
        resume, store_note = megatron_resume(args, resume)
    elif pools is not None and not limit:  # S27 G-EK: the FSDP2 island in verl's separate mode
        from .separate import ASSERTED_KEYS as SEPARATE_ASSERTED_KEYS

        asserted_keys = (*vconf.ASSERTED_KEYS, *SEPARATE_ASSERTED_KEYS)
    if full and megatron is None:  # S27 G-EC: the Miles training options verl can express (.train_options);
        # the Megatron island maps its own (megatron_engine.build_overrides)
        from .train_options import apply_training_options

        try:
            overrides, asserted_keys = apply_training_options(
                args, overrides, asserted_keys, max_prompt_len=max_prompt,
                max_response_len=args.rollout_max_response_len)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    try:  # S27 G-EN: heldout eval through verl's _validate (.eval_port); the sync trainer only
        from . import eval_port

        if megatron is not None or limit or buffer_section is not None:
            eval_port.refuse_unwired(args, "Megatron island" if megatron is not None
                                     else "fully_async" if limit else "buffer trainer")
            eval_plan = None
        else:
            eval_plan = eval_port.eval_section(
                args, work=work / "data", max_prompt_len=max_prompt,
                max_response_len=args.rollout_max_response_len, temperature=run.temperature,
                top_p=run.top_p)
        overrides, asserted_keys = eval_port.apply_eval_overrides(eval_plan, overrides, asserted_keys)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    quant_overrides, quant_summary = rollout_quant_overrides(
        args.rl_rollout_quant_tier, Path(model_path), colocated=buffer_section is None)
    overrides += quant_overrides
    if limit:  # agentic-rollout-utilization 6.4b: fully_async path
        before = asserted_keys
        # pools (G-EV) and trainer_cards (G-EZ2) never both apply: a buffer trainer with
        # fixed-partition pools is refused above (G-EW), so pools is None whenever
        # buffer_section is not None.
        overrides, asserted_keys = fully_async_plan(
            limit, overrides, spec_dict, args, sync, pools=pools,
            trainer_cards=run.n_gpus if buffer_section is not None else None)
        if full:  # S27 G-EV: a full-parameter island keeps its training options' asserted keys
            asserted_keys = tuple(dict.fromkeys((*asserted_keys, *before)))
        pools = None  # S27 G-EV: the fully_async path places its own pools; not verl's separate mode
        append_correction_record(args, spec_dict)
    test = json.loads(TEST_FILE.read_text()) if TEST_FILE.is_file() else {}
    exit_after = (test.get("exit_after_version") or {}).get(str(lid))
    env_exit = os.environ.get("YETO_VERL_TEST_EXIT_AFTER", "")  # TEST ONLY "ISLAND:VERSION[,...]"
    for item in filter(None, env_exit.split(",")):
        island, _, version = item.partition(":")
        if int(island) == lid:
            exit_after = int(version)
    readback = work / "readback"
    plan = {
        "learner_id": lid, "sync": sync, "syncer": args.syncer, "global_rounds": args.global_rounds,
        "groups_per_round": args.groups_per_round, "samples_per_group": args.samples_per_group,
        "model_path": model_path, "model_revision": args.model_revision, "lora_rank": run.lora_rank,
        "parameter_mode": args.parameter_mode, **full_plan,
        "lora_targets": args.lora_targets, "event_tape": os.path.expanduser(args.event_tape),
        "out_dir": str(OUTPUT), "tis_upper": algo["tis_upper"], "wan_streams": args.wan_streams,
        "bcast_delta": bool(args.bcast_delta),
        "thresholds_key": ["verl", engine, "vllm-" + str(manifest["versions"].get("vllm")),
                           _threshold_card(manifest)],
        "engine": engine,
        **({"megatron": megatron["shape"], "hf_exports": megatron["hf_exports"],
            "store_note": store_note} if megatron is not None else {}),
        # S27 G-FL: absent unless --rl-ascend-plugin-stack asked for it
        **({"ascend_plugin_stack": megatron["ascend_plugin_stack"]}
           if megatron is not None and megatron["ascend_plugin_stack"] else {}),
        **({"separate": pools.to_dict()} if pools is not None else {}),  # S27 G-EK
        "algorithm_spec": spec_dict, "syncer_epoch": args.rl_syncer_epoch, "island_lease_s": args.rl_sync_island_lease_s,
        "test_exit_after_version": exit_after,
        # S27 task assignment (rl_driver_start); "dispense": the rows come from the dispenser (G-EU)
        "data_slot": "dispense" if dispense_plan else args.rl_data_slot,
        **({"dispense": dispense_plan} if dispense_plan else {}),
        **verify,  # S27 G-EU: verify_every / verify_every_round / debug_dump (absent when off)
        # S27 G-FS: the driver and every Ray worker register yeto_grpo, and the
        # Harbor environment is forwarded to the agent-loop workers (verl_main).
        "agentic": bool(run.agentic),
        "publish_selftest": os.environ.get("YETO_VERL_PUBLISH_SELFTEST") == "1", "run_config": run.to_dict(), "overrides": overrides,
        "asserted": {k: _override_value(overrides, k) for k in asserted_keys},
        "max_policy_age": int(args.rl_max_policy_age or 0), "fully_async": bool(limit),
        **({"rollout_ahead": ahead} if ahead else {}),  # S27 G-EV: separate.wire_rollout_ahead
        "resume": resume,
        "versions": manifest["versions"],
        **({"sample_buffer": buffer_section} if buffer_section else {}),
        "rollout_quant": quant_summary,
        **({eval_port.PLAN_KEY: eval_plan} if eval_plan else {}),
    }
    plan_path = work / f"plan-{lid}.json"
    plan_path.write_text(json.dumps(plan, indent=1, default=str))
    (OUTPUT / f"verl-plan-{lid}.json").write_text(json.dumps(plan, indent=1, default=str))
    env = dict(os.environ)
    workdir = str(Path(__file__).resolve().parents[4])
    env["PYTHONPATH"] = workdir + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["YETO_VERL_PLAN"] = str(plan_path)
    env.update(spot_env)
    if args.rl_echo_events:  # tape lines also on stdout (head launcher / relaunch-proof copy)
        from yeto.rl.event_echo import ECHO_ENV

        env[ECHO_ENV] = "1"
    env["YETO_VERL_READBACK_DIR"] = str(readback)
    if run.agentic:  # S27 G-FS: the recorder environment the agent-loop workers read
        env.update(agentic_records_env(run, model_path=model_path, card=_threshold_card(manifest)))
        # Names only, never values: the tunnel's path secret and the boat token are
        # in this environment. One line says whether the tunnel reached the island,
        # which is the only way to tell a missing callback port from a dead sandbox.
        from yeto.rl.harness.gateway import records_env as _renv

        present = sorted(k for k in env if k.startswith(("YETO_RECORDS_", "HARBOR_", "YETO_BOAT_",
                                                         "YETO_AGENTIC_", "YETO_HARBOR_")))
        print(f"[yeto-verl] island {lid} agentic env {present}", flush=True)
        for name in (_renv.PROXY_PORT_ENV, _renv.PROXY_PUBLIC_BASE_ENV, _renv.PROXY_SECRET_FILE_ENV, "BOAT_API_KEY"):
            if not env.get(name):
                print(f"[yeto-verl] island {lid} agentic WARNING {name} is not set "
                      "(the sandboxes cannot reach the records proxy, or cannot start)", flush=True)
    env["VERL_FILE_LOGGER_PATH"] = str(OUTPUT / f"verl-file-logger-{lid}.jsonl")
    readback.mkdir(parents=True, exist_ok=True)
    entry = "main_buffer_trainer" if buffer_section else "main_fully_async" if limit else "main"
    cmd = [sys.executable, "-c", f"from yeto.rl.adapters.verl.verl_main import {entry}; {entry}()", *overrides]
    print("[yeto-verl] " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    log = OUTPUT / f"verl-train-{lid}.log"
    stop_releaser = threading.Event()
    if args.rl_rollout_quant_tier == TIER_TRAINER_FP8:  # S26 E6 tier 2: BF16 releases -> FP8 releases
        start_trainer_releaser(buffer_section, Path(model_path), work / "fp8-template", stop_releaser)
    try:
        with log.open("a") as handle:
            proc = subprocess.Popen(cmd, env=env, cwd=verl_root(),
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in proc.stdout:
                sys.stdout.write(line)
                handle.write(line)
            code = proc.wait()
    finally:
        stop_releaser.set()
    print(f"[yeto-verl] island {lid} exit {code} after {time.time() - t0:.0f}s", flush=True)
    return code


def rollout_ahead_of(args) -> tuple[int, int]:
    """S27 G-EV (VERL-PARITY #10): ``(--rl-rollout-ahead, the fully_async limit)``.

    Rollout ahead runs verl's separate mode with its synchronous trainer: the
    rollout servers generate round r+1 with the published policy r while the
    trainer trains round r (``separate``, ``VerlRolloutPool.start_generate``).
    ``--rl-max-policy-age`` then bounds that one-version lag and does not select
    the fully_async path (the limit returned is 0). The launcher refuses the same
    combinations first (``policy_age.rollout_ahead_problems`` and the engine-neutral
    ``rollout_ahead.launch_problems``); the island refuses them again here."""
    from .policy_age import MAX_ROLLOUT_AHEAD

    ahead = int(getattr(args, "rl_rollout_ahead", 0) or 0)
    limit = int(args.rl_max_policy_age or 0)
    if not ahead:
        return 0, limit
    problems = []
    if not 0 < ahead <= MAX_ROLLOUT_AHEAD:
        problems.append(f"--rl-rollout-ahead 0..{MAX_ROLLOUT_AHEAD}")
    if args.parameter_mode != vconf.FULL_PARAMETER_MODE:
        problems.append("--tuning full (a LoRA island's fixed partition is the fully_async path)")
    if (args.rl_placement or "colocated") != "fixed-partition":
        problems.append("--rl-placement fixed-partition (the rollout servers on their own cards)")
    if limit < ahead:
        problems.append(f"--rl-max-policy-age >= {ahead} (every sample lags {ahead} version)")
    if args.rl_sample_role:
        problems.append("no --rl-sample-role")
    if problems:
        raise SystemExit("verl 岛的 --rl-rollout-ahead 需要：" + "; ".join(problems))
    return ahead, 0


def island_model_path(model: str, revision: str | None) -> str:
    """This island's model directory: a hub repo id at the pinned revision, or a
    directory that already exists on this node.

    S27 G-EQ2 (npu1): an on-prem cluster stages the weights itself and its nodes
    may have no route to the hub at all (npu1's n2-n4 do not), so a directory is
    used as is. The pinned revision then cannot be checked -- the plan keeps the
    flag's value and the island's identity still fingerprints ``config.json`` of
    the files actually loaded (``full_parameter_stream.plan_section``)."""
    local = Path(os.path.expanduser(model))
    if local.is_dir():
        if not (local / "config.json").is_file():
            raise SystemExit(f"--model {model} 是本机目录但没有 config.json")
        print(f"[yeto-verl] model from this node: {local} (--model-revision {revision} 未校验)", flush=True)
        return str(local)
    from huggingface_hub import snapshot_download

    return snapshot_download(model, revision=revision)


def model_supports_gradient_checkpointing(model_path: str, *, trust_remote_code: bool = False) -> bool | None:
    """Whether this model's HF implementation can recompute activations, or None
    when it cannot be determined (no transformers, an unmapped architecture).

    Only the model class is looked at, never the weights: ``supports_gradient_
    checkpointing`` is a class attribute of ``PreTrainedModel``."""
    try:
        from transformers import AutoConfig
        from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        model_class = MODEL_FOR_CAUSAL_LM_MAPPING[type(config)]
        return bool(getattr(model_class, "supports_gradient_checkpointing", True))
    except Exception as error:  # noqa: BLE001 - a probe never stops the island
        print(f"[yeto-verl] gradient checkpointing support unknown for {model_path}: "
              f"{type(error).__name__}: {error}", flush=True)
        return None


def gradient_checkpointing_for(args, model_path: str) -> bool:
    """Whether this island recomputes activations (S27 G-EQ2).

    Default on, as before. Turned off only when this model's implementation says
    it cannot: verl's FSDP2 engine calls ``gradient_checkpointing_enable()``
    unconditionally, and a model that refuses takes the island down after the
    whole model is built (npu1: HF NemotronH of transformers 5.3 sets
    ``supports_gradient_checkpointing = False``). ``--rl-activation-recompute``
    is an explicit request, so it is refused instead of silently dropped.

    S27 G-EQ4: a false answer here no longer means the island stores every
    activation. transformers' NemotronH has no recompute hook at all (its source
    never mentions ``gradient_checkpointing``), so an Ascend island installs one
    itself -- :mod:`yeto.rl.nemotron_h_recompute` wraps the block forward, which
    is what made a 64 GiB card fit more than one step. That hook is applied in
    the worker, next to the Mamba2 scan fix, and it leaves verl's own flag off."""
    supported = model_supports_gradient_checkpointing(
        model_path, trust_remote_code=bool(getattr(args, "trust_remote_code", False)))
    requested = bool(getattr(args, "rl_activation_recompute", False))
    if supported is False:
        if requested:
            raise SystemExit(f"--rl-activation-recompute：{model_path} 的 transformers 实现不支持"
                             "重算激活（supports_gradient_checkpointing = False）")
        print(f"[yeto-verl] activation recompute off (verl 的开关): {model_path} 的 transformers 实现不支持；"
              "昇腾岛改由 yeto.rl.nemotron_h_recompute 自己装重算钩子", flush=True)
        return False
    return True


def fsdp_separate_pools(args, ray_info: dict):
    """S27 G-EK (VERL-PARITY #13): the trainer and rollout pools of a full-parameter
    FSDP2 island with ``--rl-placement fixed-partition`` (verl's separate mode,
    :mod:`.separate`), None for a colocated island. A LoRA island's fixed partition
    is the fully_async path (``fully_async_plan``), not this one."""
    from .separate import FIXED_PARTITION, pools_from_args

    if args.parameter_mode != vconf.FULL_PARAMETER_MODE or (args.rl_placement or "colocated") != FIXED_PARTITION:
        return None
    return pools_from_args(args, island_gpus(ray_info), what="verl FSDP2 分离岛")


def fsdp_island_run(args, *, model_path: str, train_file: str, val_file: str, out_dir: str, algo: dict,
                    chat: dict, ray_info: dict, max_prompt: int, limit: int, pools=None):
    """The FSDP2 island's ``VerlRunConfig`` and Hydra overrides (unchanged by S27 G-DC,
    except that a spec clip range is refused: only the Megatron island maps it).
    ``pools`` (:func:`fsdp_separate_pools`, S27 G-EK): the trainer trains on its own
    cards and the rollout runs on standalone vLLM servers (``separate.with_separate_pools``)."""
    offload = getattr(args, "rl_fsdp_offload", "none") or "none"
    clips = [name for name in ("eps_clip", "eps_clip_high") if algo.get(name) is not None]
    if clips:
        raise SystemExit("verl FSDP2 岛未映射 " + ", ".join(clips) + "（只有 Megatron 岛映射）")
    full = args.parameter_mode == vconf.FULL_PARAMETER_MODE  # S27 B10
    run = vconf.VerlRunConfig(
        model_path=model_path, train_file=train_file, val_file=val_file, out_dir=out_dir,
        parameter_mode=args.parameter_mode,
        lora_rank=0 if full else args.lora_r, lora_alpha=0 if full else args.lora_r, lr=args.inner_lr,
        groups_per_round=args.groups_per_round, samples_per_group=args.samples_per_group,
        max_prompt_len=max_prompt, max_response_len=args.rollout_max_response_len, seed=args.seed,
        total_rounds=args.global_rounds, correction=algo["correction"],
        # limit > 0: the correction is mapped by offpolicy_correction in
        # fully_async_plan (lower > 0 warns there, D3); the sync path keeps its check.
        tis_lower=0.0 if limit else algo.get("tis_lower", 0.0), tis_upper=algo["tis_upper"],
        gpu_memory_utilization=(float(os.environ.get("YETO_VERL_GPU_MEM", "0.5")) if pools is None
                                else pools.gpu_memory_utilization),
        max_model_len=max_prompt + args.rollout_max_response_len,
        reward_path=str(Path(__file__).with_name("reward_fn.py")),
        reward_name=function_for(args.reward_function), chat_template_kwargs=chat,
        trust_remote_code=bool(getattr(args, "trust_remote_code", False)),
        gradient_checkpointing=gradient_checkpointing_for(args, model_path),
        param_offload=offload in ("param", "both"), optimizer_offload=offload in ("optimizer", "both"),
        # fsdp2 offloads through CPUOffloadPolicy; the two flags above are the FSDP1 path
        offload_policy=offload in ("param", "both"),
        n_gpus=island_gpus(ray_info) if pools is None else pools.trainer_gpus,
        experiment_name=f"yeto-verl-{args.learner_id}",
        **agentic_run_fields(args, out_dir=out_dir))
    overrides = vconf.build_overrides(run)
    if pools is not None and not limit:  # limit > 0: the fully_async pools (fully_async_plan)
        from .separate import with_separate_pools

        overrides = with_separate_pools(overrides, pools)
    overrides += [
        # S27 G-FS: the agentic island keeps the worker count its run config fixed
        # (one records proxy per worker, one tunnelled port), so it is not raised here.
        f"actor_rollout_ref.rollout.agent.num_workers={run.agent_loop_workers if run.agentic else 2}",
        "reward.num_workers=2",
    ]
    from yeto.hw.catalog import device_family

    if device_family() == "ascend":
        # S27 G-EQ2 (npu1): verl compiles the entropy/log-prob helpers with
        # torch.compile by default (actor.use_torch_compile, actor.yaml:110).
        # An Ascend image carries triton-ascend, which has no inductor backend:
        # the first training step dies in dynamo with "cannot import name
        # 'Language' from 'triton.backends.compiler'" (it even tried to load
        # triton's AMD backend). Eager is correct, only slower.
        # S27 G-EQ3: the *engine* workers read their own key
        # (``FSDPEngineConfig.use_torch_compile``, engine/fsdp/transformer_impl.py:183),
        # and the generated config hardcodes it to true instead of selecting the
        # actor's value, so the actor key alone left compute_log_prob compiling.
        overrides += ["actor_rollout_ref.actor.use_torch_compile=False",
                      "actor_rollout_ref.actor.fsdp_config.use_torch_compile=False",
                      "actor_rollout_ref.ref.fsdp_config.use_torch_compile=False"]
        overrides += ascend_memory_overrides(run.gradient_checkpointing)
    return run, overrides


def ascend_memory_overrides(gradient_checkpointing: bool) -> list[str]:
    """What an Ascend FSDP2 island changes to fit a step into a 64 GiB card
    (S27 G-EQ3, npu1; the NVIDIA reference card has 141 GiB).

    The first backward of Nemotron Nano 30B-A3B died with "NPU out of memory.
    Tried to allocate 5.04 GiB (60.96 GiB total capacity; 50.30 GiB already
    allocated)". Two costs dominate, and both are optional:

    * the **gradient of a stacked MoE expert tensor**. One Nano MoE layer holds
      ``mixer.experts.up_proj`` as a single (128, 3712, 2688) parameter, 1.28e9
      values; FSDP2's default ``reduce_dtype=fp32`` upcasts the whole unsharded
      gradient before the reduce-scatter, which is 4.8 GiB of transient device
      memory per layer. Reducing in bf16 halves it. The FP32 master parameters
      are untouched (``model_dtype=fp32``), so the full-parameter export and the
      sync boundary still read FP32 masters.
    * the **entropy of the log-prob head**: Nano's vocabulary is 131072, so a
      2560-token micro-batch has 2560x131072 logits; entropy over them, kept for
      the backward, is another 1.3 GiB per copy. Chunking computes it in slices
      and checkpointing recomputes it in the backward instead of keeping it.

    A model whose implementation supports activation recomputation does not need
    the entropy knobs (the recompute already bounds the head), so they are only
    added when it is off.

    S27 G-EQ4 checked the first bullet instead of assuming it: the fork holds two
    engine config classes, and only the VeOmni one takes ``mixed_precision`` as a
    bool (``workers/config/engine.py:418``), while the FSDP one takes the dict
    (:282). ``FSDPActorConfig.__post_init__`` sets ``engine = fsdp_config``, so the
    override does arrive -- composed and converted, the engine reports
    ``mixed_precision = {'param_dtype': 'bf16', 'reduce_dtype': 'bf16',
    'buffer_dtype': 'fp32'}`` and FSDP2's policy is ``reduce_dtype=torch.bfloat16``
    (``tools/probe_mp.py`` on npu1). The bf16 reduce was therefore **already** in
    force through every out-of-memory attempt, and it is not the missing saving.
    What was missing is activation recompute, which this implementation cannot do
    at all: see :mod:`yeto.rl.nemotron_h_recompute`, installed in the worker.

    What is deliberately *not* here: verl's ``enable_activation_offload``. It
    looked like the remaining lever, but on this model it fails in the backward
    of the first MoE layer -- ``activation_offload.tensor_push`` asserts that a
    saved tensor's tag is new, and ``transformers``' NemotronH router saves the
    same tensor twice (``route_tokens_to_experts``: ``.topk(2, dim=-1)[0]``),
    which trips that assertion (S27 G-EQ3, measured on npu1).
    """
    # the key is absent from the generated config (fsdp_config.mixed_precision defaults
    # to None in the dataclass), so it is added, not overridden: hydra needs the "+".
    overrides = ["+actor_rollout_ref.actor.fsdp_config.mixed_precision="
                 "{param_dtype: bf16, reduce_dtype: bf16, buffer_dtype: fp32}"]
    # One sample per micro-batch instead of packing up to the token budget (S27 G-EQ3,
    # npu1): with packing, what fits depends on which samples a round draws -- the same
    # island did a step with the groups at buffer position 0..3 and ran out of memory on
    # the groups after them (8.50 GiB short of 60.96 GiB). verl requires the actor and
    # the log-prob pass to agree on this, so both are switched.
    overrides += ["actor_rollout_ref.actor.use_dynamic_bsz=False",
                  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
                  "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False",
                  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1"]
    if not gradient_checkpointing:
        overrides += ["actor_rollout_ref.actor.entropy_from_logits_with_chunking=True",
                      "actor_rollout_ref.actor.entropy_checkpointing=True"]
    return overrides


def megatron_resume(args, resume: dict | None) -> tuple[None, str | None]:
    """The Megatron island has no round cuts. The launcher sends a no-sync island with
    a checkpoint store ``--rl-resume-store`` as well; the island then uses the store
    for its HF exports only and says so (plan ``store_note`` and the island log).
    An explicit cut flag is refused."""
    from .round_cuts import RESUME_FLAGS, _flag

    given = [_flag(n) for n in RESUME_FLAGS if getattr(args, n, None) is not None]
    if given:
        raise SystemExit("verl Megatron 岛没有接切点：" + ", ".join(given))
    if resume is None:
        return None, None
    note = (f"--rl-resume-store {resume['store']}: round cuts are not wired for the verl Megatron "
            "island; the store holds the HF exports only, and a relaunched island starts from the base model")
    print(f"[yeto-verl] WARNING: {note}", file=sys.stderr, flush=True)
    return None, note


def megatron_island_plan(args, *, model_path: str, train_file: str, val_file: str, out_dir: str, algo: dict,
                         chat: dict, ray_info: dict, reward_path: str, reward_name: str,
                         buffer_trainer: bool = False) -> dict:
    """S27 G-DC: the Megatron island (``megatron_engine``): full parameters, trainer and
    rollout pools apart, age 0, no outer sync (S27 G-FG: or the full-parameter
    outer sync, ``megatron_engine.sync_problems``). Returns run, overrides, the keys the
    island re-reads and the plan sections ``megatron`` (the shape) and ``hf_exports``;
    every flag the island cannot honour is refused here, before verl starts.

    ``buffer_trainer`` (S27 G-FN, blockers B1-B3): the island runs as a buffer trainer
    (``--rl-sample-role trainer``): it consumes samples from the store and publishes
    its weights there, and it holds no rollout engine. Three requirements of the
    separate mode are then lifted, because each of them exists only to place a rollout
    pool: the fixed partition (B3, ruled by the main agent 07:4xZ), the policy age of 0
    (B2: a buffer trainer is on the fully_async path and needs age >= 1) and the
    refusal of ``--rl-sample-role`` itself (B1). ``--rl-rollout-quant-tier`` stays
    refused on both roles. Every card the island's Ray reports is a training card."""
    from .megatron_engine import MegatronShape, asserted_keys, build_overrides

    problems = []
    if (args.parameter_mode or "lora") != "full":
        problems.append("--tuning full (parameter mode full)")
    synced = not args.rl_single_island_no_sync  # S27 G-FG: the full-parameter outer sync
    if synced:
        problems += megatron_engine.sync_problems(args)
    placement = args.rl_placement or "colocated"
    if buffer_trainer:
        # S27 G-FN (B3): no rollout cards, so nothing to keep apart from the training
        # cards; a fixed partition would carve a rollout pool out of the training cards.
        if placement != "colocated":
            problems.append("--rl-placement colocated（缓冲训练角色没有 rollout 卡，不分池）")
        if int(args.rl_max_policy_age or 0) < 1:
            problems.append("--rl-max-policy-age >= 1（缓冲训练角色走 fully_async 路径）")
    else:
        if placement != "fixed-partition":
            problems.append("--rl-placement fixed-partition (trainer and rollout pools apart)")
        if int(args.rl_max_policy_age or 0) != 0 and not int(args.rl_rollout_ahead or 0):
            problems.append("--rl-max-policy-age 0, or --rl-rollout-ahead 1 (the fully_async path is "
                            "not wired), or --rl-sample-role trainer (the buffer trainer role)")
        if args.rl_sample_role:
            problems.append("no --rl-sample-role")
    if args.rl_rollout_quant_tier:
        problems.append("no --rl-rollout-quant-tier")
    if args.rl_data_slot and not synced:
        problems.append("no --rl-data-slot (one island)")
    if problems:
        raise SystemExit("verl Megatron 岛需要：" + "; ".join(problems))
    from .separate import pools_from_args, trainer_only_pools

    cards = island_gpus(ray_info)
    # S27 G-EK: shared with the FSDP2 separate island; S27 G-FN: the buffer trainer
    # role takes every card as a training card (no rollout pool to carve out).
    pools = (trainer_only_pools(args, cards, what="verl Megatron 缓冲训练岛") if buffer_trainer
             else pools_from_args(args, cards, what="verl Megatron 岛"))
    trainer_gpus = pools.trainer_gpus
    with open(Path(model_path) / "config.json") as handle:
        model_config = json.load(handle)
    num_experts = model_config.get("n_routed_experts")
    # S27 G-FN (B12): a hybrid model that declares no MTP layers must say so, or the
    # bridge builds its MTP branch from a default pattern and dies
    # (megatron_engine.MTP_OFF_PATTERN)
    close_mtp = bool(model_config.get("hybrid_override_pattern")) and not any(
        key in model_config for key in megatron_engine.MTP_CONFIG_KEYS)
    # S28 G-CPW: a Mamba model needs seq_idx in verl's THD packing; probe the verl this
    # process imports (the workers probe their own at start, megatron_ports)
    mamba = megatron_engine.has_mamba(model_config)
    shape = MegatronShape(
        trainer_nodes=pools.trainer_nodes, trainer_gpus_per_node=pools.trainer_gpus_per_node,
        rollout_nodes=pools.rollout_nodes, rollout_gpus_per_node=pools.rollout_gpus_per_node,
        rollout_tp=pools.rollout_tp, tp=int(args.tensor_parallel or 1),
        pp=int(args.pipeline_parallel or 1), ep=int(args.expert_parallel or 1),
        cp=int(args.context_parallel or 1), mamba=mamba,
        seq_idx=megatron_engine.probe_seq_idx(int(args.context_parallel or 1)) if mamba else None,
        max_tokens_per_gpu=int(args.rl_max_tokens_per_gpu or 8192),
        router_replay="R3" if args.use_rollout_routing_replay else "disabled",
        gpu_memory_utilization=pools.gpu_memory_utilization,
        num_experts=int(num_experts) if num_experts else None,
        grad_nan_check=not args.rl_no_grad_nan_check,
        rollout_pool=not buffer_trainer,
        optimizer_offload_fraction=float(getattr(args, "rl_optimizer_offload_fraction", 0.0) or 0.0),
        bf16_moments=bool(getattr(args, "rl_megatron_bf16_moments", False)),
        first_stage_layers=getattr(args, "rl_megatron_first_stage_layers", None),
        last_stage_layers=getattr(args, "rl_megatron_last_stage_layers", None),
        recompute_full_layers=getattr(args, "rl_megatron_recompute_full_layers", None),
        fused_logprob_kernels=bool(getattr(args, "rl_megatron_fused_logprob_kernels", False)),
        close_mtp_pattern=close_mtp)
    max_prompt = max(512, int(args.seq_len) - int(args.rollout_max_response_len))
    run = vconf.VerlRunConfig(
        model_path=model_path, train_file=train_file, val_file=val_file, out_dir=out_dir,
        parameter_mode=vconf.FULL_PARAMETER_MODE, lora_rank=0, lora_alpha=0, lr=args.inner_lr,
        groups_per_round=args.groups_per_round, samples_per_group=args.samples_per_group,
        max_prompt_len=max_prompt, max_response_len=args.rollout_max_response_len, seed=args.seed,
        total_rounds=args.global_rounds, correction=algo["correction"],
        tis_lower=algo.get("tis_lower", 0.0), tis_upper=algo["tis_upper"],
        max_model_len=max_prompt + args.rollout_max_response_len, reward_path=reward_path,
        reward_name=reward_name, chat_template_kwargs=chat, n_gpus=trainer_gpus,
        experiment_name=f"yeto-verl-{args.learner_id}")
    overrides = build_overrides(run, shape, eps_clip=algo.get("eps_clip"),
                                eps_clip_high=algo.get("eps_clip_high")) + [
        "actor_rollout_ref.rollout.agent.num_workers=8",
        "reward.num_workers=8",
    ]
    from yeto.rl.engine.hf_export import ExportRounds, export_store_path

    rounds = ExportRounds.parse(args.rl_export_hf_rounds)
    if bool(rounds) != bool(args.rl_export_hf_store):
        raise SystemExit("--rl-export-hf-rounds and --rl-export-hf-store go together")
    exports = ({"rounds": str(rounds), "store": export_store_path(str(args.rl_export_hf_store),
                                                                  learner_id=int(args.learner_id))}
               if rounds else None)
    if args.rl_lr_schedule not in (None, "constant"):
        raise SystemExit(f"verl Megatron 岛只接 --rl-lr-schedule constant（得到 {args.rl_lr_schedule}）")
    return {"run": run, "overrides": overrides, "shape": shape.to_dict(), "hf_exports": exports,
            "asserted_keys": asserted_keys(shape),
            "ascend_plugin_stack": ascend_plugin_stack_section(args, engine=megatron_engine.ENGINE_MEGATRON)}


def ascend_plugin_stack_section(args, *, engine: str) -> dict | None:
    """S27 G-FL: the plan's ``ascend_plugin_stack`` section; None without the flag.

    Only an island with model parallelism takes it: those plugins patch Megatron-Core
    (and the model code), which the FSDP2 island does not use."""
    root = getattr(args, "rl_ascend_plugin_stack", None)
    if not root:
        return None
    if engine != megatron_engine.ENGINE_MEGATRON:
        raise SystemExit("--rl-ascend-plugin-stack 只接有模型并行的 verl 岛（TP/PP/EP > 1）："
                         "那组插件改的是 Megatron-Core，FSDP2 岛用不上")
    from yeto.rl import ascend_plugin_stack

    try:
        return ascend_plugin_stack.plan(str(root))
    except ascend_plugin_stack.AscendPluginStackError as error:
        raise SystemExit(f"--rl-ascend-plugin-stack: {error}") from None


def rollout_quant_overrides(tier: int, model_path: Path, *, colocated: bool) -> tuple[list[str], dict]:
    """S26 E6: rollout quantization on a verl island; returns the vLLM overrides (none) and
    the plan record. Tier 1 on a colocated island is refused before verl starts
    (``tiers.VERL_TIER1_REFUSAL``: LoRA syncs never re-quantize the base). On a buffer
    trainer the rollout runs on the sampler island, so nothing is added; tier 2 starts
    the releaser instead. ``model_path`` is kept for the tier 1 plan once a full-parameter
    verl path exists (``tiers.vllm_online_overrides``)."""
    from yeto.rl.quant import tiers

    summary: dict = {"tier": tier, "name": tiers.TIER_NAMES[tier]}
    if tier == tiers.TIER_ONLINE_FP8 and colocated:
        raise SystemExit(f"[yeto-verl] {tiers.VERL_TIER1_REFUSAL}")
    return [], summary


def start_trainer_releaser(buffer_section: dict | None, base_dir: Path, template_dir: Path,
                           stop: threading.Event) -> threading.Thread:
    """S26 E6 tier 2 on a buffer trainer: convert every BF16 release of the store into an
    FP8 release (``yeto.rl.quant.release.scoped_releaser``) in a thread of this process."""
    from yeto.rl.engine.resume import store_for
    from yeto.rl.quant.release import scoped_releaser, start_scoped_releaser
    from yeto.rl.quant.schemes import FP8_FFN_CHANNEL

    if not buffer_section:
        raise SystemExit("--rl-rollout-quant-tier 2 needs --rl-sample-role trainer on a verl island")
    store = store_for(buffer_section.get("release_store") or buffer_section["store"])
    releaser = scoped_releaser(store, base_dir, template_dir, FP8_FFN_CHANNEL)
    return start_scoped_releaser(releaser, stop)


def sample_buffer_section(args) -> dict | None:
    """The plan's ``sample_buffer`` section of a buffer trainer (S25 C2); None for
    role ``both``. A verl island is never a sampler: a sampler island runs the
    engine sampler (``python -m yeto.rl.engine.engine_sampler``) on any engine."""
    from yeto.rl.engine import sample_roles

    role = sample_roles.sample_role(args)
    if role == sample_roles.BOTH:
        return None
    if role == sample_roles.SAMPLER:
        return None  # main() runs the engine sampler (.sampler_entry) instead of verl
    if role != sample_roles.TRAINER:
        raise SystemExit(f"--rl-sample-role {role}: a verl island trains from the buffer or "
                         "runs the engine sampler (role sampler)")
    if int(args.rl_max_policy_age or 0) < 1:
        raise SystemExit("--rl-sample-role trainer needs --rl-max-policy-age >= 1 (the fully_async path)")
    try:
        sample_roles.check_sample_store(args)
        lane = sample_roles.buffer_lane(args)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    config = sample_roles.resolve(args, groups_per_round=args.groups_per_round)
    return {"store": config.store, "wait_s": config.wait_s, "lane": None if lane is None else lane.env(),
            # S27 G-EQ3: staged bf16 delta releases (0: the release goes into the store whole)
            "release_delta_full_every": int(config.release_delta_full_every or 0),
            **({"no_release": True} if getattr(args, "rl_measure_no_release", False) else {}),
            **({"release_only": list(args.rl_release_only_versions)}
               if getattr(args, "rl_release_only_versions", None) is not None else {}),
            **({"shards": list(args.rl_buffer_shards)} if getattr(args, "rl_buffer_shards", None) else {})}


def spot_rollout_env(args, buffer_section: dict | None) -> dict[str, str]:
    """S27 G-EV (SPOT-ROLLOUT.md §12.9 item 1): the environment of the spot-rollout
    channel on a verl island ({} when off; ``YETO_`` variables reach verl's Ray actors).

    * ``--rl-spot-subscribers N``: island 0 publishes its fragment layout on the relay
      board (the engine-neutral stream does it, ``layout_manifest.maybe_publish``);
    * ``--rl-sample-relay`` on a full-parameter buffer trainer: the relay variables
      (``trainer_relay_env``); the buffer trainer then reads the relay
      (``relay_trainer``). Any other island refuses it (it would be ignored)."""
    env: dict[str, str] = {}
    if int(getattr(args, "rl_spot_subscribers", 0) or 0) > 0:
        from yeto.rl.spot_rollout.layout_manifest import PUBLISH_ENV

        if args.parameter_mode != vconf.FULL_PARAMETER_MODE or args.rl_single_island_no_sync:
            raise SystemExit("--rl-spot-subscribers 需要同步的 verl 全参数岛（订阅者收合并后的全参数分片）")
        env[PUBLISH_ENV] = "1"
    if getattr(args, "rl_sample_relay", False):
        from yeto.rl.spot_rollout.relay_buffer import trainer_relay_env

        from .full_parameter_stream import syncer_addresses
        from .relay_trainer import RELAY_SECTION_KEY

        if (buffer_section is None or args.parameter_mode != vconf.FULL_PARAMETER_MODE
                or args.rl_single_island_no_sync or not args.syncer):
            raise SystemExit("--rl-sample-relay 只接同步的 verl 全参数缓冲训练岛（--rl-sample-role trainer，"
                             "全参数，有 syncer）")
        addresses = syncer_addresses(args.syncer)
        if len(addresses) != 1:
            raise SystemExit("--rl-sample-relay 需要不分片的 syncer（--syncer-shards 1）")
        buffer_section[RELAY_SECTION_KEY] = {"channel": int(args.learner_id)}
        env.update(trainer_relay_env(addresses[0], int(args.learner_id), groups_per_round=int(args.groups_per_round),
                                     prefetch=args.rl_sample_relay_prefetch))
    return env


def fully_async_plan(limit: int, overrides: list[str], spec_dict: dict | None, args, sync: str, *,
                     pools=None, trainer_cards: int | None = None):
    """Limit > 0: fully_async overrides + asserted keys; the algorithm contract is
    checked here (bounded staleness needs ``execution.max_policy_staleness >= limit``
    and a TIS correction) so a wrong spec fails before verl starts.

    ``trainer_cards`` (S27 G-EZ2): a buffer trainer's island cards; its trainer
    pool is all of them over ``--actor-num-nodes`` nodes
    (``fully_async_round.buffer_trainer_shape``). None keeps the in-island
    rollout split (one rollout card, one trainer card).

    The ``algorithm.rollout_correction`` overrides come from the correction
    mapping (rl-offpolicy-correction-passthrough D2/D3): decoupled PPO with
    ``bypass_mode=False`` written over the fully_async yaml default; a
    forbidden setting is refused here."""
    from yeto.rl.engine.algorithm import AlgorithmSpec
    from yeto.rl.engine.execution_profile import check_algorithm_contract

    from .fully_async_round import (FULLY_ASYNC_ASSERTED_KEYS, VerlTranslationError, buffer_trainer_shape,
                                    execution_profile_for, fully_async_run_overrides)
    from .offpolicy_correction import apply_to_overrides, correction_mapping
    from .policy_age import SUPPORT

    SUPPORT.check(limit)
    spec = AlgorithmSpec.from_dict(spec_dict) if spec_dict else AlgorithmSpec()
    profile = execution_profile_for(spec, limit, groups_per_round=args.groups_per_round,
                                    samples_per_group=args.samples_per_group, sync=sync)
    check_algorithm_contract(profile, spec)
    # S27 G-EV (VERL-PARITY #11): a full-parameter island's pools from the launcher's
    # fixed-partition flags (separate.pools_from_args); a LoRA island keeps 1 + 1.
    # S27 G-EZ2: a buffer trainer has no rollout pool, so its trainer takes every card
    # of the island over --actor-num-nodes nodes (buffer_trainer_shape). An island never
    # reaches here with both -- fixed-partition pools on a buffer trainer are refused
    # (G-EW) -- and when both are passed anyway, trainer_cards wins whole: the role with
    # no rollout pool claims no rollout cards (G-FN, test_s27_verl_megatron_buffer.py).
    sizes = {} if pools is None else {"rollout_gpus": pools.rollout_gpus, "trainer_gpus": pools.trainer_gpus}
    if trainer_cards is not None:  # S27 G-EZ2: a buffer trainer has no rollout pool
        try:
            nodes, per_node = buffer_trainer_shape(trainer_cards, getattr(args, "actor_num_nodes", 1),
                                                   getattr(args, "actor_num_gpus_per_node", None))
        except VerlTranslationError as exc:
            raise SystemExit(str(exc)) from exc
        # S27 G-FN: no rollouter on this role, so no weight-transport backend either
        sizes = {"trainer_gpus": per_node, "trainer_nodes": nodes, "rollouter": False}
    out = fully_async_run_overrides(overrides, limit, groups_per_round=args.groups_per_round,
                                    rounds=args.global_rounds, **sizes)
    out = apply_to_overrides(out, correction_mapping(spec))
    from yeto.hw.catalog import device_family

    if device_family() == "ascend":
        # fully_async_ppo_trainer.yaml defaults trainer.device=cuda; on an NPU
        # node its resource pools then ask Ray for "GPU" and never schedule
        # (S20 ModelArts 2x910B4, runBA_try2_gpures).
        out = [o for o in out if not o.startswith("trainer.device=")] + ["trainer.device=npu"]
    return out, tuple(vconf.ASSERTED_KEYS) + FULLY_ASYNC_ASSERTED_KEYS


def append_correction_record(args, spec_dict: dict | None) -> None:
    """rl-offpolicy-correction-passthrough D8 (as Miles' island entry): the island
    tape records the full correction group, the verl values and ``prox_anchor``.
    A field verl cannot map refuses the start (phase 2, L6b.2), after the record
    and its ``correction_config_warning`` events are on the tape."""
    from yeto.rl.engine.algorithm import AlgorithmSpec
    from yeto.rl.engine.events import append_tape_event
    from yeto.rl.engine.offpolicy_correction import (CORRECTION_RECORD_EVENT, correction_record,
                                                     refuse_correction_warnings)

    from .offpolicy_correction import BACKEND, correction_mapping

    spec = AlgorithmSpec.from_dict(spec_dict) if spec_dict else AlgorithmSpec()
    if spec.correction.is_default():
        return  # default tapes unchanged
    mapping = correction_mapping(spec)
    tape = SimpleNamespace(yeto_rl_event_tape=args.event_tape,
                           yeto_rl_learner_id=args.learner_id)
    append_tape_event(tape, {"event": CORRECTION_RECORD_EVENT,
                             **correction_record(spec.correction, mapping, backend=BACKEND)})
    warnings = [warning.event() for warning in mapping.warnings]
    for warning in warnings:
        append_tape_event(tape, warning)
    refuse_correction_warnings(warnings)


def _override_value(overrides, key):
    """The value yeto wrote for ``key``, or None when it wrote no override for it.

    S28 G-120C: Hydra's "add this key" prefixes (``+``, ``++``) are part of the
    override string but not part of the config path, so a plain ``startswith``
    never matched an asserted key written through one. Every asserted key under
    ``override_transformer_config`` therefore read back with ``want: null``, and
    ``check_asserted`` skips a None -- the key was re-read but never compared
    (S28 G-120 handoff section 6 item 12 saw this on
    ``gradient_accumulation_fusion`` and called it "a read-back, not an
    assertion"). Strip the prefix before comparing so those keys assert.
    """
    for item in overrides:
        if item.lstrip("+").startswith(key + "="):
            return item.split("=", 1)[1]
    return None


if __name__ == "__main__":
    from yeto.island_credential_guard import check_island_credentials

    check_island_credentials()  # secret-handling-hardening D4
    sys.exit(main())

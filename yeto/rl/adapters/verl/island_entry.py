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
from . import run_config_rules
from .pins import VERL_COMMIT, VERL_ROOT, expected_versions
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
    p.add_argument("--rl-algorithm-spec", default=None)
    p.add_argument("--rl-island-pool", action="store_true")
    p.add_argument("--rl-syncer-epoch", type=int, default=0)
    p.add_argument("--rl-sync-island-lease-s", type=float, default=None)
    p.add_argument("--rl-echo-events", action="store_true")
    p.add_argument("--rl-max-policy-age", type=int, default=0)
    # S25 C2: a trainer island over the sample buffer (sample_roles; the sampler is
    # the engine sampler island, yeto.rl.engine.engine_sampler). --rl-resume-store
    # names the buffer's store there, and the round-cut store otherwise (N7.6b)
    p.add_argument("--rl-sample-role", default=None)
    p.add_argument("--rl-resume-store", default=None)
    p.add_argument("--rl-sample-wait-s", type=float, default=None)
    # S26 E9.4: the pair flags the launcher forwards per island (sample_roles.island_flags);
    # a sampler island of a verl launch runs the engine sampler (.sampler_entry)
    p.add_argument("--rl-sampler-node", default=None)
    p.add_argument("--rl-buffer-shards", default=None, type=lambda text: [s for s in text.split(",") if s])
    p.add_argument("--rl-sampler-data-slot", default=None)
    # S27 task assignment: this training island's row slot of --data (data_split)
    p.add_argument("--rl-data-slot", default=None)
    p.add_argument("--rl-sampler-max-backlog-rounds", type=int, default=None)
    p.add_argument("--rl-sampler-min-round-seconds", type=float, default=None)
    # S25 C3: a synced trainer island (syncer, learner m of N) reads lane m of the
    # buffer in this store; the launcher forwards both (yeto.rl.engine.merged_release)
    p.add_argument("--rl-sample-store", default=None)
    p.add_argument("--rl-sample-lanes", type=int, default=None)
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
    # S27 B10: a full-parameter island and its outer sync (Miles island spellings)
    from .full_parameter_stream import add_island_flags

    add_island_flags(p)
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
    out = {"correction": method, "tis_upper": 2.0}
    if method == "tis":
        out["tis_upper"] = float(correction.tis_clip if correction.tis_clip is not None else 2.0)
        out["tis_lower"] = float(correction.tis_clip_low or 0.0)
    if problems:
        raise vconf.VerlConfigError("verl 后端未映射的算法字段: " + ", ".join(problems))
    return out


def runtime_manifest(learner_id: int, device_family: str | None = None) -> dict:
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
    expected = expected_versions(family)
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


def prepare_data(data: str, revision: str | None, root: Path, *, val_rows: int = 64,
                 slot: Any = None) -> tuple[str, str]:
    """``data``'s train and test splits as verl parquet files. ``slot``
    (``data_split.DataSlot``, S27 task assignment): the train file keeps only the
    slot's rows, in data order, each with its row number in the unsplit data as
    ``extra_info["yeto_data_row"]``; the validation rows stay whole."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    from yeto.rl.engine.data_split import DATA_ROW_KEY

    root.mkdir(parents=True, exist_ok=True)
    out = {}
    for split in ("train", "test"):
        src = hf_hub_download(data, f"{split}.parquet", repo_type="dataset", revision=revision)
        frame = pd.read_parquet(src)
        rows = [{
            "data_source": "yeto/" + data,
            "prompt": [{"role": m["role"], "content": m["content"]} for m in row["messages"]],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": str(row["label"])},
            "extra_info": {"split": split, "index": int(i)},
        } for i, row in frame.iterrows()]
        if split == "test":
            rows = rows[:val_rows]
        elif slot is not None and not slot.whole:
            rows = [{**row, "extra_info": {**row["extra_info"], DATA_ROW_KEY: number}}
                    for number, row in slot.numbered(rows)]
        path = root / f"{split}.parquet"
        pd.DataFrame(rows).to_parquet(path)
        out[split] = str(path)
    return out["train"], out["test"]


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


def island_gpus(ray_info: Mapping[str, Any]) -> int:
    """GPUs of this single-node island: the island-local Ray's ``GPU`` resource
    (the launcher starts it with every card of the container), else 1.

    S26 E9 G3: the sync island used ``n_gpus=1`` always, so a 2-card container
    trained on one rank (e9-g3-a); every card now joins the FSDP2 group."""
    resources = ray_info.get("after") or ray_info.get("before") or {}
    return max(1, int(resources.get("GPU", 0)))


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
    gpus = int(before.get("GPU", 0))
    cmd = (f'ray start --head --node-ip-address={host} --port={port} --num-cpus={min_cpus} '
           f'--num-gpus={gpus} --include-dashboard=false --temp-dir={temp}')
    info["restart"] = cmd
    info["restart_rc"] = subprocess.run(cmd, shell=True).returncode
    info["after"] = _ray_resources(address)
    return info


def main(argv=None) -> int:
    args, ignored = parse_args(sys.argv[1:] if argv is None else argv)
    run_config_rules.refuse_agentic_argv(ignored)  # S25 N10.4: never ignore agentic flags
    buffer_section = sample_buffer_section(args)  # S25 C2: refused roles stop before any work
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lid = args.learner_id
    t0 = time.time()
    manifest = runtime_manifest(lid)
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

    from huggingface_hub import snapshot_download

    model_path = snapshot_download(args.model, revision=args.model_revision)
    work = Path(os.path.expanduser("~/yeto-verl"))
    train_file, val_file = prepare_data(args.data, args.data_revision, work / "data",
                                        slot=island_data_slot(args))
    spec_dict = json.loads(Path(os.path.expanduser(args.rl_algorithm_spec)).read_text()) \
        if args.rl_algorithm_spec else None
    algo = algorithm_options(spec_dict)
    chat = json.loads(args.apply_chat_template_kwargs) if args.apply_chat_template_kwargs else {}
    max_prompt = 512
    limit = int(args.rl_max_policy_age or 0)
    full = args.parameter_mode == vconf.FULL_PARAMETER_MODE
    if full and (limit or buffer_section is not None or args.rl_rollout_quant_tier):
        raise SystemExit("verl 全参数岛只接同步共置路径：不支持 --rl-max-policy-age > 0、"
                         "--rl-sample-role trainer 与 --rl-rollout-quant-tier")
    run = vconf.VerlRunConfig(
        model_path=model_path, train_file=train_file, val_file=val_file, out_dir=str(work / "out"),
        parameter_mode=args.parameter_mode,
        lora_rank=0 if full else args.lora_r, lora_alpha=0 if full else args.lora_r, lr=args.inner_lr,
        groups_per_round=args.groups_per_round, samples_per_group=args.samples_per_group,
        max_prompt_len=max_prompt, max_response_len=args.rollout_max_response_len, seed=args.seed,
        total_rounds=args.global_rounds, correction=algo["correction"],
        # limit > 0: the correction is mapped by offpolicy_correction in
        # fully_async_plan (lower > 0 warns there, D3); the sync path keeps its check.
        tis_lower=0.0 if limit else algo.get("tis_lower", 0.0), tis_upper=algo["tis_upper"],
        gpu_memory_utilization=float(os.environ.get("YETO_VERL_GPU_MEM", "0.5")),
        max_model_len=max_prompt + args.rollout_max_response_len,
        reward_path=str(Path(__file__).with_name("reward_fn.py")),
        reward_name=function_for(args.reward_function), chat_template_kwargs=chat,
        n_gpus=island_gpus(ray_info), experiment_name=f"yeto-verl-{lid}")
    overrides = vconf.build_overrides(run) + [
        "actor_rollout_ref.rollout.agent.num_workers=2",
        "reward.num_workers=2",
    ]
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
    if full:  # S27 B10: model identity, and the outer full-parameter sync when synced
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
    if full:  # S27 G-EC: the Miles training options verl can express (.train_options)
        from .train_options import apply_training_options

        try:
            overrides, asserted_keys = apply_training_options(
                args, overrides, asserted_keys, max_prompt_len=max_prompt,
                max_response_len=args.rollout_max_response_len)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    quant_overrides, quant_summary = rollout_quant_overrides(
        args.rl_rollout_quant_tier, Path(model_path), colocated=buffer_section is None)
    overrides += quant_overrides
    if limit:  # agentic-rollout-utilization 6.4b: fully_async path
        overrides, asserted_keys = fully_async_plan(limit, overrides, spec_dict, args, sync)
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
        "thresholds_key": ["verl", "fsdp2", "vllm-" + str(manifest["versions"].get("vllm")),
                           _threshold_card(manifest)],
        "algorithm_spec": spec_dict, "syncer_epoch": args.rl_syncer_epoch, "island_lease_s": args.rl_sync_island_lease_s,
        "test_exit_after_version": exit_after,
        "data_slot": args.rl_data_slot,  # S27 task assignment (rl_driver_start)
        "publish_selftest": os.environ.get("YETO_VERL_PUBLISH_SELFTEST") == "1", "run_config": run.to_dict(), "overrides": overrides,
        "asserted": {k: _override_value(overrides, k) for k in asserted_keys},
        "max_policy_age": limit, "fully_async": bool(limit),
        "resume": resume,
        "versions": manifest["versions"],
        **({"sample_buffer": buffer_section} if buffer_section else {}),
        "rollout_quant": quant_summary,
    }
    plan_path = work / f"plan-{lid}.json"
    plan_path.write_text(json.dumps(plan, indent=1, default=str))
    (OUTPUT / f"verl-plan-{lid}.json").write_text(json.dumps(plan, indent=1, default=str))
    env = dict(os.environ)
    workdir = str(Path(__file__).resolve().parents[4])
    env["PYTHONPATH"] = workdir + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["YETO_VERL_PLAN"] = str(plan_path)
    if args.rl_echo_events:  # tape lines also on stdout (head launcher / relaunch-proof copy)
        from yeto.rl.event_echo import ECHO_ENV

        env[ECHO_ENV] = "1"
    env["YETO_VERL_READBACK_DIR"] = str(readback)
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
            proc = subprocess.Popen(cmd, env=env, cwd=VERL_ROOT if Path(VERL_ROOT).is_dir() else None,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in proc.stdout:
                sys.stdout.write(line)
                handle.write(line)
            code = proc.wait()
    finally:
        stop_releaser.set()
    print(f"[yeto-verl] island {lid} exit {code} after {time.time() - t0:.0f}s", flush=True)
    return code


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
            **({"shards": list(args.rl_buffer_shards)} if getattr(args, "rl_buffer_shards", None) else {})}


def fully_async_plan(limit: int, overrides: list[str], spec_dict: dict | None, args, sync: str):
    """Limit > 0: fully_async overrides + asserted keys; the algorithm contract is
    checked here (bounded staleness needs ``execution.max_policy_staleness >= limit``
    and a TIS correction) so a wrong spec fails before verl starts.

    The ``algorithm.rollout_correction`` overrides come from the correction
    mapping (rl-offpolicy-correction-passthrough D2/D3): decoupled PPO with
    ``bypass_mode=False`` written over the fully_async yaml default; a
    forbidden setting is refused here."""
    from yeto.rl.engine.algorithm import AlgorithmSpec
    from yeto.rl.engine.execution_profile import check_algorithm_contract

    from .fully_async_round import (FULLY_ASYNC_ASSERTED_KEYS, execution_profile_for,
                                    fully_async_run_overrides)
    from .offpolicy_correction import apply_to_overrides, correction_mapping
    from .policy_age import SUPPORT

    SUPPORT.check(limit)
    spec = AlgorithmSpec.from_dict(spec_dict) if spec_dict else AlgorithmSpec()
    profile = execution_profile_for(spec, limit, groups_per_round=args.groups_per_round,
                                    samples_per_group=args.samples_per_group, sync=sync)
    check_algorithm_contract(profile, spec)
    out = fully_async_run_overrides(overrides, limit, groups_per_round=args.groups_per_round,
                                    rounds=args.global_rounds)
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
    for item in overrides:
        if item.startswith(key + "="):
            return item.split("=", 1)[1]
    return None


if __name__ == "__main__":
    from yeto.island_credential_guard import check_island_credentials

    check_island_credentials()  # secret-handling-hardening D4
    sys.exit(main())

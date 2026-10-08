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
import time
from pathlib import Path

from . import config as vconf
from .pins import EXPECTED_VERSIONS, VERL_COMMIT, VERL_ROOT
from .reward_fn import function_for

OUTPUT = Path(os.path.expanduser("~/yeto-output"))
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
    p.add_argument("--apply-chat-template-kwargs", default=None)
    p.add_argument("--rl-algorithm-spec", default=None)
    p.add_argument("--rl-island-scheduling", default="legacy")
    p.add_argument("--rl-syncer-epoch", type=int, default=0)
    p.add_argument("--rl-echo-events", action="store_true")
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


def runtime_manifest(learner_id: int) -> dict:
    import importlib.metadata as md

    versions = {}
    for name in EXPECTED_VERSIONS:
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = None
    try:
        import torch

        versions["torch"] = torch.__version__
    except ImportError:
        pass
    sha_file = Path("/workspace/verl_sha.txt")
    commit = sha_file.read_text().strip() if sha_file.is_file() else None
    smi = subprocess.run("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader",
                         shell=True, capture_output=True, text=True).stdout.strip()
    manifest = {"schema": "yeto-verl-runtime-v1", "verl_commit": commit, "versions": versions,
                "expected_versions": EXPECTED_VERSIONS, "nvidia_smi": smi, "learner_id": learner_id,
                "python": sys.version.split()[0]}
    problems = []
    if commit != VERL_COMMIT:
        problems.append(f"verl commit {commit} != pin {VERL_COMMIT}")
    for name, want in EXPECTED_VERSIONS.items():
        if versions.get(name) != want:
            problems.append(f"{name} {versions.get(name)} != {want}")
    manifest["problems"] = problems
    return manifest


def prepare_data(data: str, revision: str | None, root: Path, *, val_rows: int = 64) -> tuple[str, str]:
    import pandas as pd
    from huggingface_hub import hf_hub_download

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
        path = root / f"{split}.parquet"
        pd.DataFrame(rows).to_parquet(path)
        out[split] = str(path)
    return out["train"], out["test"]


def main(argv=None) -> int:
    args, ignored = parse_args(sys.argv[1:] if argv is None else argv)
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
    if manifest["problems"]:
        print(f"[yeto-verl] runtime check failed: {manifest['problems']}", file=sys.stderr, flush=True)
        return 2

    from huggingface_hub import snapshot_download

    model_path = snapshot_download(args.model, revision=args.model_revision)
    work = Path(os.path.expanduser("~/yeto-verl"))
    train_file, val_file = prepare_data(args.data, args.data_revision, work / "data")
    spec_dict = json.loads(Path(os.path.expanduser(args.rl_algorithm_spec)).read_text()) \
        if args.rl_algorithm_spec else None
    algo = algorithm_options(spec_dict)
    chat = json.loads(args.apply_chat_template_kwargs) if args.apply_chat_template_kwargs else {}
    max_prompt = 512
    run = vconf.VerlRunConfig(
        model_path=model_path, train_file=train_file, val_file=val_file, out_dir=str(work / "out"),
        lora_rank=args.lora_r, lora_alpha=args.lora_r, lr=args.inner_lr,
        groups_per_round=args.groups_per_round, samples_per_group=args.samples_per_group,
        max_prompt_len=max_prompt, max_response_len=args.rollout_max_response_len, seed=args.seed,
        total_rounds=args.global_rounds, correction=algo["correction"],
        tis_lower=algo.get("tis_lower", 0.0), tis_upper=algo["tis_upper"],
        gpu_memory_utilization=float(os.environ.get("YETO_VERL_GPU_MEM", "0.5")),
        reward_path=str(Path(__file__).with_name("reward_fn.py")),
        reward_name=function_for(args.reward_function), chat_template_kwargs=chat,
        experiment_name=f"yeto-verl-{lid}")
    overrides = vconf.build_overrides(run) + [
        "actor_rollout_ref.rollout.agent.num_workers=2",
        "reward.num_workers=2",
    ]
    if args.rl_single_island_no_sync:
        sync = "none"
    elif args.rl_island_scheduling == "elastic":
        sync = "elastic"
    else:
        sync = "strict"
    if sync != "none" and not args.syncer:
        raise SystemExit("--syncer is required unless --rl-single-island-no-sync")
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
        "model_path": model_path, "model_revision": args.model_revision, "lora_rank": args.lora_r,
        "lora_targets": args.lora_targets, "event_tape": os.path.expanduser(args.event_tape),
        "out_dir": str(OUTPUT), "tis_upper": algo["tis_upper"], "wan_streams": args.wan_streams,
        "thresholds_key": ["verl", "fsdp2", "vllm-" + str(manifest["versions"].get("vllm")),
                           "H100" if "H100" in manifest["nvidia_smi"] else manifest["nvidia_smi"]],
        "algorithm_spec": spec_dict, "syncer_epoch": args.rl_syncer_epoch,
        "test_exit_after_version": exit_after,
        "publish_selftest": os.environ.get("YETO_VERL_PUBLISH_SELFTEST") == "1", "run_config": run.to_dict(), "overrides": overrides,
        "asserted": {k: _override_value(overrides, k) for k in vconf.ASSERTED_KEYS},
        "versions": manifest["versions"],
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
    cmd = [sys.executable, "-c", "from yeto.rl.adapters.verl.verl_main import main; main()", *overrides]
    print("[yeto-verl] " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    log = OUTPUT / f"verl-train-{lid}.log"
    with log.open("a") as handle:
        proc = subprocess.Popen(cmd, env=env, cwd=VERL_ROOT if Path(VERL_ROOT).is_dir() else None,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            handle.write(line)
        code = proc.wait()
    print(f"[yeto-verl] island {lid} exit {code} after {time.time() - t0:.0f}s", flush=True)
    return code


def _override_value(overrides, key):
    for item in overrides:
        if item.startswith(key + "="):
            return item.split("=", 1)[1]
    return None


if __name__ == "__main__":
    sys.exit(main())

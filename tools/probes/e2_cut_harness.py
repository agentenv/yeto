#!/usr/bin/env python3
"""E2 GPU harness, host side (rl-infra-spec 4.2-4.5, plan-v3). DRY-RUN ONLY.

Writes, for every run of plan-v3 in order, a self-contained run directory:

  <root>/<prefix>-<run>/args.txt          yeto launch arguments (Modal, --controller local)
  <root>/<prefix>-<run>/harness.json      in-learner harness plan (C1/C2 only; copied to the
                                          snapshot root as yeto-rl-e2-harness.json)
  <root>/<prefix>-<run>/run.sh            launch + independent watchdog + puller + (C3) the
                                          controller rebuild-trainer trigger
  <root>/<prefix>-<run>/spec.json         what the run must show (criteria ids of plan-v3)

and a top-level ``plan.json``. It never creates a cloud resource: ``run.sh`` is
written with a guard line that exits unless ``YETO_E2_GPU_APPROVED=1`` is set
by a human, and this tool itself refuses ``--execute``. ``--launcher-dry-run``
additionally runs ``python -m yeto.cli <args> --dry-run`` for each run against
a yeto checkout (CPU only; the launcher's dry-run plan calls neither sky nor
Modal).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

PLAN_VERSION = "plan-v3"
IMAGE_DIGEST = "sha256:17d428a2e955a1d43525b59b8785bb786b8e48852fe00c6e3e90dad798f0bcef"
IMAGE = f"ghcr.io/michaellchung/yeto-miles-ports@{IMAGE_DIGEST}"
MILES_COMMIT = "5c1b49ebccbc7508c1d9ef89eacc2db3e448b6ba"
MODELS = {
    "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
    "Qwen/Qwen3-1.7B": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
}
DATA = ("zhuzilin/gsm8k", "0cbd9f31d91ac21a7613dcbc7fef992adac459ae")
HARNESS_PLAN_FILE = "yeto-rl-e2-harness.json"
STATE_DIR = "~/yeto-rl/elastic-state"
MODAL = "/tmp/modal-venv/bin/modal"
HEAD_PY = "/home/michael/work/gpu-head/venv/bin/python"


def common_args(model: str, *, gpu: str, prefix: str) -> list[str]:
    return [
        "launch", "--controller", "local", "--training-mode", "rl", "--rl-engine", "ports",
        "--rl-single-island-no-sync", "--gpu", gpu, "--modal-gpu-exact", "--cluster-prefix", prefix,
        "--model", model, "--model-revision", MODELS[model],
        "--data", DATA[0], "--data-revision", DATA[1], "--reward-function", "gsm8k_reward:score",
        "--tuning", "lora", "--lora-r", "16", "--lora-alpha", "32", "--lora-targets", "all-linear",
        "--fragments", "1", "--pipeline", "1",
        # GBS 16 = 2 prompts x 8 samples, one optimizer step per rollout
        "--rollout-batch-size", "2", "--n-samples-per-prompt", "8",
        "--rollout-max-response-len", "384", "--seq-len", "1024", "--inner-lr", "1e-5",
        "--seed", "1234", "--apply-chat-template-kwargs", '{"enable_thinking": false}',
        "--trust-remote-code", "--rl-deterministic-trainer",
    ]


@dataclass
class Run:
    name: str
    config: str  # C1 / C2 / C3
    gpu: str
    model: str
    criteria: list[str]
    extra: list[str] = field(default_factory=list)
    harness: dict | None = None
    rebuild_trigger: bool = False  # C3: controller rebuild-trainer before round 3
    blocked: str | None = None  # why it cannot produce evidence yet (still generated)
    hard_s: int = 5400  # plan-v3 §0: 90 min

    def args(self, prefix: str) -> list[str]:
        return common_args(self.model, gpu=self.gpu, prefix=f"{prefix}-{self.name}") + self.extra


def _harness(config: str, dp: int) -> dict:
    return {"schema": "yeto.e2_harness_plan/v1", "config": config, "warmup_rounds": 2,
            "out_dir": "~/yeto-rl/e2-harness", "expected_dp": dp, "lora_dropout": 0.05,
            "plan": PLAN_VERSION}


def _elastic_c3(extra: list[str]) -> list[str]:
    return [
        "--total-steps", "6", "--rl-placement", "fixed-partition", "--rl-rollout-gpus", "1",
        "--rl-elastic", "--rl-elastic-resources", "{RESOURCES}", "--rl-elastic-initial-config", "T2R1S0",
        "--rl-elastic-cells", "c0", *extra,
    ]


def plan_runs() -> list[Run]:
    """plan-v3 order: G-4.2+G-4.3 on C1 then C2 (harness), G-4.4 on C3, then G-4.5 rows on C3."""
    g42 = [f"G-4.2({c})" for c in "abcdeg"]
    g43 = ["G-4.3(1)", "G-4.3(2)", "G-4.3(3)", "G-4.3(4)", "L2"]
    runs = [
        Run("c1", "C1", "modal:1xh100", "Qwen/Qwen3-0.6B", g42 + g43,
            extra=["--total-steps", "6"], harness=_harness("C1", 1)),
        Run("c2", "C2", "modal:2xh100", "Qwen/Qwen3-0.6B", g42 + ["G-4.2(f)"] + g43,
            extra=["--total-steps", "6"], harness=_harness("C2", 2)),
        Run("c3-b1", "C3", "modal:3xh100", "Qwen/Qwen3-1.7B", ["G-4.4 baseline"], extra=_elastic_c3([])),
        Run("c3-rb", "C3", "modal:3xh100", "Qwen/Qwen3-1.7B", ["G-4.4"], extra=_elastic_c3([]),
            rebuild_trigger=True),
        Run("c3-rbold", "C3", "modal:3xh100", "Qwen/Qwen3-1.7B", ["G-4.4 REBUILD_OLD", "G-4.5 row4"],
            extra=_elastic_c3(["--rl-test-inject-rebuild-fail"]), rebuild_trigger=True),
        Run("c3-f1", "C3", "modal:3xh100", "Qwen/Qwen3-1.7B", ["G-4.5 row1"],
            extra=_elastic_c3(["--rl-test-inject-cut-restore-kill-rank", "1"]), rebuild_trigger=True),
        Run("c3-f2", "C3", "modal:3xh100", "Qwen/Qwen3-1.7B", ["G-4.5 row2"],
            extra=_elastic_c3(["--rl-distributed-timeout-minutes", "5",
                               "--rl-test-inject-cut-restore-sleep", "1:420"]), rebuild_trigger=True),
        Run("c3-f3", "C3", "modal:3xh100", "Qwen/Qwen3-1.7B", ["G-4.5 row3"],
            extra=_elastic_c3(["--rl-test-inject-cut-save-kill-rank", "1"]), rebuild_trigger=True),
        Run("c3-f5", "C3", "modal:3xh100", "Qwen/Qwen3-1.7B", ["G-4.5 row5"],
            extra=_elastic_c3(["--rl-test-inject-rebuild-cursor-shift", "1"]), rebuild_trigger=True,
            blocked="E1 data_cursor() is the last batch's cursor, not a live read: a rewind "
                    "during the rebuild is not visible before the next generate"),
        Run("c3-f6a", "C3", "modal:3xh100", "Qwen/Qwen3-1.7B", ["G-4.5 row6 before CAS"],
            extra=_elastic_c3(["--rl-test-kill-learner-at", "REBUILDING_TRAINER",
                               "--rl-elastic-restart-attempts", "1"]), rebuild_trigger=True),
        Run("c3-f6b", "C3", "modal:3xh100", "Qwen/Qwen3-1.7B", ["G-4.5 row6 after CAS"],
            extra=_elastic_c3(["--rl-test-kill-learner-at", "COMMITTED",
                               "--rl-elastic-restart-attempts", "1"]), rebuild_trigger=True),
    ]
    return runs


RESOURCES_T2R1S0 = {  # same schema as the B1 cfg/resources-*.json
    "configs": {"T2R1S0": {"trainer": 2, "rollout": 1, "standby": 0, "rollout_engine_gpus": 1}},
    "edges": [],
}


def check_pins(repo: Path) -> list[str]:
    """The checkout's pins must be plan-v3's (environment only; criteria unchanged)."""
    text = (repo / "yeto" / "rl" / "__init__.py").read_text(encoding="utf-8")
    out = []
    if MILES_COMMIT not in text:
        out.append(f"MILES_NEXT_COMMIT is not {MILES_COMMIT}")
    if IMAGE_DIGEST.split(":", 1)[1] not in text:
        out.append(f"MILES_NEXT_IMAGE is not {IMAGE_DIGEST}")
    return out


GUARD = (
    'if [ "${YETO_E2_GPU_APPROVED:-}" != 1 ]; then\n'
    '  echo "refused: GPU runs are paused; set YETO_E2_GPU_APPROVED=1 only with user approval" >&2\n'
    "  exit 2\nfi\n"
)


def run_script(run: Run, rdir: Path, *, yeto_sha: str, source_repo: Path) -> str:
    """launch + watchdog + puller (+ rebuild trigger), in the pattern of the B1 mrun.sh."""
    app = f"yeto-{rdir.name}"
    harness_copy = (f"cp {rdir}/harness.json $R/yeto/{HARNESS_PLAN_FILE}\n" if run.harness else "")
    trigger = ""
    if run.rebuild_trigger:
        trigger = f"""
# controller rebuild-trainer once two rounds are on the tape (G-4.4 / G-4.5)
setsid nohup bash -c '
until [ -f $R/rc.txt ]; do
  n=$(grep -c "\\"rl_round_trained\\"" $R/pulled/rl-island-0.jsonl 2>/dev/null || echo 0)
  if [ "$n" -ge 2 ] && [ ! -f $R/rebuild.sent ]; then
    c=$(cat $R/pulled/container_id.txt)
    HOME=$R/home timeout 1200 {MODAL} container exec $c -- sh -c "cd ~/sky_workdir && PYTHONPATH=~/miles:~/sglang/python:. python3 -m yeto.rl.engine.controller --state-dir {STATE_DIR} rebuild-trainer rb1 --expected-epoch 0 --deadline-s 900" > $R/rebuild.out 2>&1
    touch $R/rebuild.sent
  fi
  sleep 10
done' > $R/trigger.log 2>&1 &
"""
    pull_extra = ("timeout 120 $M container exec $c -- sh -c 'cd ~/yeto-rl && tar czf - e2-harness 2>/dev/null | base64 -w0' > $R/pulled/.h && [ -s $R/pulled/.h ] && mv $R/pulled/.h $R/pulled/e2-harness.tgz.b64\n"
                  if run.harness else
                  "timeout 120 $M container exec $c -- sh -c 'cd ~/yeto-rl && tar czf - --exclude=elastic-state/cuts/*/trainer_* elastic-state 2>/dev/null | base64 -w0' > $R/pulled/.s && [ -s $R/pulled/.s ] && mv $R/pulled/.s $R/pulled/elastic-state.tgz.b64\n")
    return f"""#!/bin/bash
# {PLAN_VERSION} {run.name} ({run.config}); generated by tools/probes/e2_cut_harness.py -- DRY-RUN ARTIFACT
{GUARD}set -u
R={rdir}; M={MODAL}
mkdir -p $R/home $R/runs $R/pulled $R/yeto
cp ~/.modal.toml $R/home/
git -C {source_repo} archive {yeto_sha} | tar x -C $R/yeto
cp /home/michael/work/gpu-default-modal/yeto/gsm8k_reward.py $R/yeto/
touch $R/yeto/yeto-rl-echo-events
{harness_copy}# independent watchdog: stop the app after the hard timeout plus 5 min
setsid nohup bash -c "sleep {run.hard_s + 300}; HOME=$R/home $M app stop -y {app} > $R/watchdog.out 2>&1" >/dev/null 2>&1 &
echo $! > $R/watchdog.pid
setsid nohup bash -c '
export HOME=$R/home
while [ ! -f $R/rc.txt ]; do
  for c in $($M container list --json 2>/dev/null | /usr/bin/python3 -c "import json,sys
try: d=json.load(sys.stdin)
except Exception: d=[]
[print(x[\\"container_id\\"]) for x in d if x.get(\\"app_name\\")==\\"{app}\\"]"); do
    echo $c > $R/pulled/container_id.txt
    timeout 60 $M container exec $c -- sh -c "cat /root/yeto-output/rl-island-0.jsonl 2>/dev/null" > $R/pulled/.t && [ -s $R/pulled/.t ] && mv $R/pulled/.t $R/pulled/rl-island-0.jsonl
    [ -s $R/pulled/gpu.txt ] || timeout 60 $M container exec $c -- sh -c "nvidia-smi --query-gpu=index,uuid,name,driver_version --format=csv,noheader" > $R/pulled/gpu.txt
    {pull_extra.strip()}
  done
  sleep 10
done' > $R/puller.log 2>&1 &
echo $! > $R/puller.pid
{trigger}
(
export HOME=$R/home YETO_RUNS_DIR=$R/runs PYTHONPATH=$R/yeto
cd $R/yeto
date -u +%FT%TZ > $R/start_utc.txt
eval "timeout {run.hard_s} {HEAD_PY} -m yeto.cli $(cat $R/args.txt)" > $R/launch.log 2>&1
echo "rc=$?" > $R/rc.txt.tmp
date -u +%FT%TZ > $R/end_utc.txt
sleep 15; mv $R/rc.txt.tmp $R/rc.txt
)
HOME=$R/home $M app stop -y {app} > $R/stop.out 2>&1
HOME=$R/home $M app list > $R/app_list_after.txt 2>&1
"""


def write_plan(root: Path, prefix: str, *, yeto_sha: str, source_repo: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    summary = {"plan": PLAN_VERSION, "image": IMAGE, "miles_commit": MILES_COMMIT, "models": MODELS,
               "yeto_sha": yeto_sha, "runs": []}
    resources = json.dumps(RESOURCES_T2R1S0, sort_keys=True)
    for run in plan_runs():
        rdir = root / f"{prefix}-{run.name}"
        rdir.mkdir(parents=True, exist_ok=True)
        args = [a.replace("{RESOURCES}", str(rdir / "resources.json")) for a in run.args(prefix)]
        if "--rl-elastic" in args:
            (rdir / "resources.json").write_text(resources, encoding="utf-8")
        (rdir / "args.txt").write_text(" ".join(shlex.quote(a) for a in args) + "\n", encoding="utf-8")
        if run.harness:
            (rdir / "harness.json").write_text(json.dumps(run.harness, indent=1), encoding="utf-8")
        spec = {"run": run.name, "config": run.config, "gpu": run.gpu, "model": run.model,
                "model_revision": MODELS[run.model], "criteria": run.criteria, "blocked": run.blocked,
                "hard_timeout_s": run.hard_s, "rebuild_trigger": run.rebuild_trigger}
        (rdir / "spec.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
        script = rdir / "run.sh"
        script.write_text(run_script(run, rdir, yeto_sha=yeto_sha, source_repo=source_repo), encoding="utf-8")
        script.chmod(0o755)
        summary["runs"].append({**spec, "dir": str(rdir), "args": args})
    (root / "plan.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def launcher_dry_run(args: list[str], *, repo: Path, python: str = HEAD_PY) -> tuple[int, str]:
    """``yeto.cli <args> --dry-run`` on CPU with placeholder Modal credentials."""
    env = {**os.environ, "PYTHONPATH": str(repo), "MODAL_TOKEN_ID": "ak-dryrun",
           "MODAL_TOKEN_SECRET": "as-dryrun"}
    proc = subprocess.run([python, "-m", "yeto.cli", *args, "--dry-run"], cwd=repo, env=env,
                          capture_output=True, text=True, timeout=300)
    return proc.returncode, (proc.stdout + proc.stderr)[-4000:]


LEARNER_PREFLIGHT = r"""
import json, os, shlex, sys
from pathlib import Path
from types import SimpleNamespace
cmd, harness_path = sys.argv[1], sys.argv[2]
argv = shlex.split(cmd.split("python3 -m yeto.rl.learner", 1)[1].replace("$LEARNER_ID", "0"))
from yeto.rl import learner
from yeto.rl.engine.run_config import resolve_rl_run_config
out = {"checks": []}
args = learner.parse_args(argv)
out["checks"].append("learner.parse_args")
learner._check_ports_algorithm_options(args, outer_sync=False); out["checks"].append("_check_ports_algorithm_options")
learner._check_single_island_no_sync(args); out["checks"].append("_check_single_island_no_sync")
learner._check_ports_infra_switches(args); out["checks"].append("_check_ports_infra_switches")
learner._require_ports_supported(args); out["checks"].append("_require_ports_supported")
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id=args.model, revision=args.model_revision,
                         allow_patterns=["*.json", "*.txt", "*.model", "*.py"])
cfg = json.loads((Path(path) / "config.json").read_text())
provider = SimpleNamespace(
    hidden_size=cfg["hidden_size"], num_attention_heads=cfg["num_attention_heads"],
    num_layers=cfg["num_hidden_layers"], ffn_hidden_size=cfg["intermediate_size"],
    num_query_groups=cfg["num_key_value_heads"], kv_channels=cfg.get("head_dim"),
    seq_length=cfg["max_position_embeddings"], vocab_size=cfg["vocab_size"],
    layernorm_epsilon=cfg["rms_norm_eps"], normalization="RMSNorm",
    qk_layernorm=True, share_embeddings_and_output_weights=cfg.get("tie_word_embeddings", False),
)
from yeto.rl.export import derive_peft_lora_specs, adapter_targets
specs = derive_peft_lora_specs(path, None, rank=args.lora_r, targets=args.lora_targets,
                               trust_remote_code=args.trust_remote_code)
run_config = resolve_rl_run_config(args, model_path=path, prompt_path="/tmp/prompts.jsonl",
                                   provider=provider, target_modules=adapter_targets(specs),
                                   yeto_policy_sync=False)
out["checks"].append("resolve_rl_run_config (provider from HF config.json; the container uses Megatron-Bridge)")
launch = learner.build_ports_launch(args, run_config, ())
out["checks"].append("build_ports_launch (translate_run_config)")
miles_argv = list(launch.argv)
out["miles_argv"] = miles_argv
problems = []
if "--load" in miles_argv:
    problems.append("--load in the Miles argv (same-shape rebuild refused)")
if getattr(args, "rl_deterministic_trainer", False) and "--deterministic-mode" not in miles_argv:
    problems.append("--rl-deterministic-trainer did not reach the Miles argv")
if os.path.exists(harness_path):
    from yeto.rl.engine.miles_adapter.e2_harness import plan_problems
    plan = json.loads(Path(harness_path).read_text())
    problems += plan_problems(plan)
    i = miles_argv.index("--lora-dropout") if "--lora-dropout" in miles_argv else None
    dropout = float(miles_argv[i + 1]) if i is not None else None
    if plan.get("lora_dropout") is not None and dropout != plan["lora_dropout"]:
        problems.append(f"Miles --lora-dropout {dropout} != plan {plan['lora_dropout']} "
                        "(the in-learner harness would stop as environment_blocked)")
    from yeto.rl.engine.miles_adapter.cut_plugin import config_problems
    ns = SimpleNamespace(fp16="--fp16" in miles_argv,
                         use_precision_aware_optimizer="--use-precision-aware-optimizer" in miles_argv)
    problems += config_problems(ns)
out["problems"] = problems
print(json.dumps(out))
sys.exit(1 if problems else 0)
"""


def learner_preflight(learner_command: str, harness_path: Path, *, repo: Path,
                      python: str = "/tmp/yeto-venv/bin/python") -> tuple[int, str]:
    """The learner's own pre-GPU checks and the ports translation, run locally (review lesson:
    a learner/Miles-arg refusal must show up here, not in the container). Still image-only:
    Megatron-Bridge provider, Miles parse_args/validate, runtime manifest."""
    env = {**os.environ, "PYTHONPATH": str(repo), "OMP_NUM_THREADS": "1"}
    proc = subprocess.run([python, "-c", LEARNER_PREFLIGHT, learner_command, str(harness_path)],
                          cwd=repo, env=env, capture_output=True, text=True, timeout=600)
    return proc.returncode, (proc.stdout + proc.stderr)[-6000:]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--prefix", default="infra-e2-g4")
    p.add_argument("--yeto-sha", required=True)
    p.add_argument("--source-repo", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--launcher-dry-run", action="store_true")
    p.add_argument("--python", default=HEAD_PY, help="interpreter with sky installed (launcher dry-run)")
    p.add_argument("--execute", action="store_true", help="refused: GPU runs are paused")
    ns = p.parse_args(argv)
    if ns.execute:
        print("refused: this tool is dry-run only (GPU/cloud runs are paused by the user)", file=sys.stderr)
        return 2
    problems = check_pins(ns.source_repo)
    if problems:
        print("pins do not match " + PLAN_VERSION + ": " + "; ".join(problems), file=sys.stderr)
        return 3
    summary = write_plan(ns.root, ns.prefix, yeto_sha=ns.yeto_sha, source_repo=ns.source_repo)
    rc = 0
    if ns.launcher_dry_run:
        for run in summary["runs"]:
            code, out = launcher_dry_run(run["args"], repo=ns.source_repo, python=ns.python)
            (Path(run["dir"]) / "launcher-dry-run.txt").write_text(out, encoding="utf-8")
            run["launcher_dry_run_rc"] = code
            rc = rc or code
            print(f"{run['run']}: launcher --dry-run rc={code}")
            if code == 0:
                plan = json.loads(out[out.index("{"):out.rindex("}") + 1])
                command = plan["island_requests"][0]["learner_command"]
                code2, out2 = learner_preflight(command, Path(run["dir"]) / "harness.json",
                                                repo=ns.source_repo)
                (Path(run["dir"]) / "learner-preflight.txt").write_text(out2, encoding="utf-8")
                run["learner_preflight_rc"] = code2
                rc = rc or code2
                print(f"{run['run']}: learner preflight rc={code2}")
        (ns.root / "plan.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    for run in summary["runs"]:
        flag = f"  BLOCKED: {run['blocked']}" if run["blocked"] else ""
        print(f"{run['run']:10s} {run['config']} {run['gpu']:14s} {','.join(run['criteria'])}{flag}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

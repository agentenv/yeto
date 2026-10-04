"""E2 learner preflight INSIDE the pinned image (CPU container; no GPU, no Ray, no training).

usage: python e2_container_preflight.py <run-name> <learner-args-file> <harness.json|-> <out.json>

Runs the real ``yeto.rl.learner.main`` with ``--rl-print-attestation-fingerprint``
(INFRA-E1: builds and verifies the ports launch exactly like a real run --
Megatron-Bridge provider, translate_run_config, Miles parse_args + validate,
verify_ports_algorithm -- then exits before Ray/GPU/sync), captures the parsed
Miles namespace, and applies the E2 harness/cut static checks to it:
same-shape rebuild preconditions, cut config_problems, Megatron
deterministic_mode, LoRA dropout vs the harness plan.
"""
import io
import json
import shlex
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path

run, args_file, harness_file, out_file = sys.argv[1:5]
argv = shlex.split(Path(args_file).read_text()) + ["--rl-print-attestation-fingerprint"]
result = {"run": run, "stage": "import", "problems": []}
captured = {}
try:
    from yeto.rl import learner

    original = learner.verify_ports_algorithm

    def capture(args, miles_args, launch):
        original(args, miles_args, launch)
        captured["miles_args"] = miles_args

    learner.verify_ports_algorithm = capture
    result["stage"] = "learner.main"
    buf = io.StringIO()
    with redirect_stdout(buf):
        learner.main(argv)
    lines = [json.loads(l) for l in buf.getvalue().splitlines() if l.startswith("{")]
    fp = [l for l in lines if l.get("event") == "rl_attestation_fingerprint"]
    if not fp or "miles_args" not in captured:
        raise RuntimeError("learner did not reach the attestation fingerprint")
    result["runtime_fingerprint"] = fp[0]["runtime_fingerprint"]
    result["miles_argv"] = fp[0]["miles_argv"]
    result["stage"] = "e2_checks"
    ma = captured["miles_args"]
    from yeto.rl.engine.miles_adapter.cut_plugin import config_problems
    from yeto.rl.engine.miles_adapter.trainer_rebuild import rebuild_preconditions

    problems = [f"rebuild: {p}" for p in rebuild_preconditions(ma)]
    problems += [f"cut: {p}" for p in config_problems(ma)]
    if not getattr(ma, "deterministic_mode", False):
        problems.append("Megatron deterministic_mode is off in the parsed Miles args")
    result["parsed"] = {k: getattr(ma, k, None) for k in (
        "requested_load", "load", "ref_load", "start_rollout_id", "deterministic_mode", "lora_dropout",
        "bf16", "fp16", "use_distributed_optimizer", "tensor_model_parallel_size",
        "pipeline_model_parallel_size", "context_parallel_size", "expert_model_parallel_size",
        "actor_num_gpus_per_node", "global_batch_size", "megatron_to_hf_mode")}
    if harness_file != "-":
        from yeto.rl.engine.miles_adapter.e2_harness import plan_problems

        plan = json.loads(Path(harness_file).read_text())
        problems += [f"harness plan: {p}" for p in plan_problems(plan)]
        if plan.get("lora_dropout") is not None and getattr(ma, "lora_dropout", None) != plan["lora_dropout"]:
            problems.append(f"lora_dropout {getattr(ma, 'lora_dropout', None)} != plan {plan['lora_dropout']}")
    result["problems"] = problems
    result["stage"] = "done"
except BaseException as exc:  # noqa: BLE001 - recorded
    result["error"] = f"{type(exc).__name__}: {exc}"
    result["traceback"] = traceback.format_exc()[-4000:]
result["pass"] = result["stage"] == "done" and not result["problems"]
Path(out_file).write_text(json.dumps(result, indent=1, default=repr))
print(json.dumps({k: result.get(k) for k in ("run", "stage", "pass", "problems", "error")}, default=repr))

"""Run one harness phase inside the container with the learner's own argument pipeline.

    python learner_shim.py --phase gen|arm|dry --work DIR [--arm A1] -- <yeto.rl.learner flags>

``yeto.rl.learner.main`` does everything a real island does before Ray
(source/reward/Miles pin checks, model + data download, run config ->
Miles argv -> ``parse_args``) and then calls
``entry.run_ports_island(miles_args, launch, algorithm, ...)``. The shim
replaces that one function with the harness phase, so the harness gets the
exact ``miles_args`` a production island would use; learner, driver, entry
and launcher code are unchanged (no patch needed).

``--phase dry`` stops right there and writes ``miles_args.json`` (what the
arms would run with, and the reshard refusals for DP 1<->2) -- the last
step before any GPU process.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

FROZEN_ROLLOUTS = 8


def frozen_template(work: Path) -> str:
    return str(Path(work) / "frozen" / "rollout_{rollout_id}.pt")


PARITY_KEYS = ("balance_data", "balance_by_flops", "use_dynamic_batch_size", "use_dynamic_global_batch_size",
               "allow_partial_train_step", "calculate_per_token_loss", "normalize_advantages", "indep_dp",
               "multimodal_keys", "lora_dropout", "hidden_dropout", "attention_dropout", "global_batch_size",
               "micro_batch_size", "fp16", "virtual_pipeline_model_parallel_size")


def argv_check(argv, algorithm, overrides) -> dict:
    """THE pre-GPU check; run identically by ``local_dry.py`` and by every container phase."""
    from yeto.rl.engine.miles_adapter.reshard import argv_profile, argv_reshard_problems

    profile = argv_profile(argv, overrides)
    return {"algorithm_spec_sha256": algorithm.sha256(), "overrides": dict(overrides),
            "argv_profile": {k: getattr(profile, k, None) for k in PARITY_KEYS},
            "argv_reshard_problems": argv_reshard_problems(argv, algorithm, overrides)}


def phase_summary(miles_args, launch, algorithm, overrides=None) -> dict:
    """Container: the argv check (same as local) plus the check on the parsed args and their agreement."""
    from yeto.rl.engine.miles_adapter.reshard import ReshardPlan, reshard_problems

    overrides = dict(overrides or {})
    summary = argv_check(list(getattr(launch, "argv", ())), algorithm, overrides)
    gbs = int(miles_args.global_batch_size)
    mbs = int(getattr(miles_args, "micro_batch_size", 1) or 1)
    layout = lambda dp: {"world": dp, "tp": 1, "pp": 1, "cp": 1, "ep": 1, "dp": dp}  # noqa: E731
    parsed = {}
    for src, dst in ((1, 2), (2, 1)):
        args = argparse.Namespace(**{**vars(miles_args), "actor_num_gpus_per_node": dst})
        parsed[f"{src}->{dst}"] = reshard_problems(ReshardPlan(layout(src), layout(dst), gbs, mbs),
                                                   args=args, spec=algorithm)
    summary["parsed_reshard_problems"] = parsed
    summary["parsed_profile"] = {k: getattr(miles_args, k, None) for k in PARITY_KEYS}
    summary["parity_mismatch"] = sorted(
        k for k in PARITY_KEYS
        if summary["argv_profile"].get(k) != summary["parsed_profile"].get(k)
        and not (summary["argv_profile"].get(k) in (None, False, 0) and summary["parsed_profile"].get(k) in (None, False, 0)))
    summary["argv"] = list(getattr(launch, "argv", ()))
    return summary


def summary_problems(summary: dict) -> list[str]:
    out = [f"argv {e}: {p}" for e, ps in summary["argv_reshard_problems"].items() for p in ps]
    out += [f"parsed {e}: {p}" for e, ps in summary.get("parsed_reshard_problems", {}).items() for p in ps]
    out += [f"argv/parsed disagree on {k}" for k in summary.get("parity_mismatch", [])]
    return out


def parse_overrides(items: list[str]) -> dict:
    out = {}
    for item in items:
        key, _, raw = item.partition("=")
        if not key or not raw:
            raise SystemExit(f"--set needs ARG=JSON, got {item!r}")
        out[key] = json.loads(raw)
    return out


def make_phase(ns):
    work = Path(ns.work)

    def run_ports_island(miles_args, launch, algorithm, **_kw):
        work.mkdir(parents=True, exist_ok=True)
        overrides = parse_overrides(getattr(ns, "set", None) or [])
        for key, value in overrides.items():
            setattr(miles_args, key, value)
        summary = phase_summary(miles_args, launch, algorithm, overrides)
        (work / f"miles_args.{ns.phase}{'.' + ns.arm if ns.arm else ''}.json").write_text(
            json.dumps(summary, indent=1, sort_keys=True, default=repr))
        problems = summary_problems(summary)
        if problems:
            raise SystemExit(f"E3 profile refused before any GPU process: {problems}")
        if ns.phase == "dry":
            return None
        from miles_backend import MilesBackend

        from yeto.rl import MILES_NEXT_COMMIT
        from yeto.rl.engine.miles_adapter.entry import runtime_fingerprint

        backend = MilesBackend(miles_args, algorithm, frozen_template=frozen_template(work),
                               fingerprint=runtime_fingerprint(launch, MILES_NEXT_COMMIT))
        if ns.phase == "gen":
            (work / "frozen").mkdir(parents=True, exist_ok=True)
            backend.generate_frozen(FROZEN_ROLLOUTS)
            return None
        from harness import ARM_BY_NAME, run_arm

        run_arm(ARM_BY_NAME[ns.arm], backend, work)
        return None

    return run_ports_island


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        raise SystemExit("usage: learner_shim.py --phase ... --work DIR [--arm A] -- <learner flags>")
    split = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("dry", "gen", "arm"), required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--arm")
    ap.add_argument("--set", action="append", default=[], metavar="ARG=JSON",
                    help="profile override applied to miles_args after parse (plan-v3 §0: dropouts 0, "
                         "deterministic mode); recorded in miles_args.*.json")
    ns = ap.parse_args(argv[:split])
    if ns.phase == "arm" and not ns.arm:
        raise SystemExit("--phase arm needs --arm")
    from yeto.rl import learner
    from yeto.rl.engine.miles_adapter import entry

    entry.run_ports_island = make_phase(ns)
    learner.main(argv[split + 1:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

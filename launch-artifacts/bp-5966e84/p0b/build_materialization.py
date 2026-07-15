#!/usr/bin/env python3
"""Build the exact P0b materialization and execute argv from frozen inputs."""

from __future__ import annotations

import json
import importlib.util
import shutil
import subprocess
from pathlib import Path


PACKET = Path(__file__).resolve().parent
SOURCE_TREE = Path("/tmp/yeto-prod-8d58208")
REMOTE_PYTHON = "/home/shou/venv/bin/python"
SOURCE_COMMIT = "8d58208cacafef12cb95f2642b4fa700531151b4"
REMOTE_REPO = "/tmp/yeto-best-paper"
RUN_ID = "bp-p0b-5966e84-20260715a"
SCIENCE_ROOT = f"/tmp/runs/{RUN_ID}"
PHASE_MAP = Path(SCIENCE_ROOT) / "phase-map"
PARENT = Path(SCIENCE_ROOT) / "parent"


COMMON = [
    "--study-id",
    "bp-phase-map-p0b",
    "--study-phase",
    "p0b_canary",
    "--run-dir",
    str(PHASE_MAP),
    "--artifact-uri",
    f"gs://yeto-exp2-52-model-training-497007/{RUN_ID}/phase-map",
    "--git-commit",
    SOURCE_COMMIT,
    "--python-executable",
    REMOTE_PYTHON,
    "--command-repo-root",
    REMOTE_REPO,
    "--image-digest",
    "038098c2b5356c9117f1019bf0d19c8999ab50f259dceb041a57fcf657d2620f",
    "--image-numeric-id",
    "7290368630472593484",
    "--model-path",
    f"{SCIENCE_ROOT}/inputs/model",
    "--model-revision",
    "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
    "--data",
    f"{SCIENCE_ROOT}/inputs/train.parquet",
    "--expected-model-hash",
    "43f9494fad3335a9237f7a3093ae1401b7c4d3164c7486542070e2cc04837132",
    "--expected-data-hash",
    "970f88b3f2fa6758f3b5f94052f4e91b872541a2ba530223b44a779168c51409",
    "--expected-train-rows-hash",
    "55f92573ceae6ca77ed58b58afd6554ae4fbde8e0d47ed83dea35a4862f97cfd",
    "--expected-development-eval-rows-hash",
    "533838a0564b13519956a044d23ed8db6705ddc7ae5f0ddb96538f49460bcebc",
    "--expected-development-eval-packed-hash",
    "865be921dfa890e26f5ddd2b268fdecfb6e90bca4048d0760012b7cb60b25cb4",
    "--expected-development-eval-example-ids-hash",
    "4378783ade97d6529729082302d2794de6a559fc618d7acfbe8d3c3a6968baf2",
    "--expected-development-eval-token-ids-hash",
    "fe488ba475717621b7559e9406924bb2543a5307317dc6b224a99045e1eee642",
    "--expected-development-eval-source-indices-hash",
    "58108d24b4976523f033a8628384e962195a6f2a68b2bad2b32c77c0c7603740",
    "--expected-audit-eval-rows-hash",
    "d71b90040a57731f25c78a2d191017ce90a12c1bb79f55a1cd2f3d085a706d7b",
    "--expected-audit-eval-packed-hash",
    "c08d196e15a0b1ee88e64da11521564de2c42d56857ef899b3ba91478ba47f7f",
    "--expected-audit-eval-example-ids-hash",
    "c7be2d71515850da85a3b9d9fa0bf27b56310a25f4b5d009d46ebe887edc1170",
    "--expected-audit-eval-token-ids-hash",
    "5b725289d3308a0b8f64ea0e2a49195e9aa95b8dfaf5b9359644250426c00b41",
    "--expected-audit-eval-source-indices-hash",
    "1b1051e80f559ec6a517fbcfd38e0d39c2ba0b4b880edfac48ddae8ef9963dba",
    "--expected-train-pool-source-indices-hash",
    "aa07dd4e29918460d6706797c592980ee7b7c663944eed5714214320d380bbd9",
    "--expected-train-source-indices-hash",
    "e870a71b6f7fe7b1be245651a6b46880c3040575f330523aca35627ce920327c",
    "--require-frozen-eval",
    "--parent-manifest",
    f"{PARENT}/p0a-phase-map-manifest.json",
    "--expected-parent-manifest-hash",
    "02b1f99537d2611e3462ebe1b4ccedd11fdc07588b7c01d3abdeabbdb5b9d8f8",
    "--parent-replay-report",
    f"{PARENT}/p0a-replay-report.json",
    "--expected-parent-replay-report-hash",
    "4c1616de16708590d6a30aaf3af805adc4bad47b827087b49251d678e200c276",
    "--provider-evidence",
    f"{SCIENCE_ROOT}/provider-evidence.json",
    "--h",
    "16",
    "--mu",
    "0,.5,.9",
    "--eta",
    ".0875",
    "--seed",
    "337",
    "--training-seed",
    "337337",
    "--order-seed",
    "20260714",
    "--eval-split-seed",
    "331",
    "--token-budget",
    "65536",
    "--seq-len",
    "128",
    "--micro-batch-size",
    "1",
    "--inner-lr",
    "0.001",
    "--train-rows",
    "5000",
    "--eval-rows",
    "1024",
    "--confirmation-audit-rows",
    "1024",
    "--device",
    "cuda",
    "--gpu-slots",
    "4",
    "--syncer-checkpoint-every",
    "1",
    "--arm-timeout-min",
    "240",
    "--resource-class",
    "a2-highgpu-4g",
    "--capture-every-step",
    "--require-distinct-learner-gpu-uuids",
]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def main() -> None:
    if PHASE_MAP.exists():
        if any(PHASE_MAP.iterdir()):
            raise SystemExit(f"refusing nonempty materialization path: {PHASE_MAP}")
        PHASE_MAP.rmdir()
    PARENT.mkdir(parents=True, exist_ok=True)
    parent_packet = PACKET.parent / "p0a-parent"
    shutil.copyfile(
        parent_packet / "phase-map-manifest.json",
        PARENT / "p0a-phase-map-manifest.json",
    )
    shutil.copyfile(
        parent_packet / "p0a-replay-report.json",
        PARENT / "p0a-replay-report.json",
    )

    materialize = [
        REMOTE_PYTHON,
        f"{REMOTE_REPO}/scripts/run_phase_map.py",
        "--phase",
        "materialize",
        *COMMON,
    ]
    write_json(PACKET / "materialize-argv.json", materialize)
    head = subprocess.run(
        ["git", "-C", str(SOURCE_TREE), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        [
            "git",
            "-C",
            str(SOURCE_TREE),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if head != SOURCE_COMMIT or dirty:
        raise SystemExit("materialization source checkout is not exact and clean")

    spec = importlib.util.spec_from_file_location(
        "packet_run_phase_map", SOURCE_TREE / "scripts" / "run_phase_map.py"
    )
    if spec is None or spec.loader is None:
        raise SystemExit("cannot load exact run_phase_map source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.build_parser().parse_args(materialize[2:])
    plan = module.build_plan(args)
    template = module.verify_authoritative_prereg(args)
    bound = module.build_bound_manifest(
        args,
        plan,
        model_hash="43f9494fad3335a9237f7a3093ae1401b7c4d3164c7486542070e2cc04837132",
        data_hash="970f88b3f2fa6758f3b5f94052f4e91b872541a2ba530223b44a779168c51409",
        train_rows_hash="55f92573ceae6ca77ed58b58afd6554ae4fbde8e0d47ed83dea35a4862f97cfd",
        development_eval_rows_hash="533838a0564b13519956a044d23ed8db6705ddc7ae5f0ddb96538f49460bcebc",
        development_eval_packed_hash="865be921dfa890e26f5ddd2b268fdecfb6e90bca4048d0760012b7cb60b25cb4",
        development_eval_example_ids_hash="4378783ade97d6529729082302d2794de6a559fc618d7acfbe8d3c3a6968baf2",
        development_eval_token_ids_hash="fe488ba475717621b7559e9406924bb2543a5307317dc6b224a99045e1eee642",
        development_eval_source_indices_hash="58108d24b4976523f033a8628384e962195a6f2a68b2bad2b32c77c0c7603740",
        audit_eval_rows_hash="d71b90040a57731f25c78a2d191017ce90a12c1bb79f55a1cd2f3d085a706d7b",
        audit_eval_packed_hash="c08d196e15a0b1ee88e64da11521564de2c42d56857ef899b3ba91478ba47f7f",
        audit_eval_example_ids_hash="c7be2d71515850da85a3b9d9fa0bf27b56310a25f4b5d009d46ebe887edc1170",
        audit_eval_token_ids_hash="5b725289d3308a0b8f64ea0e2a49195e9aa95b8dfaf5b9359644250426c00b41",
        audit_eval_source_indices_hash="1b1051e80f559ec6a517fbcfd38e0d39c2ba0b4b880edfac48ddae8ef9963dba",
        audit_access_policy_hash=module.sha256_bytes(
            module.canonical_json(template["confirmation_policy"])
        ),
        train_pool_source_indices_hash="aa07dd4e29918460d6706797c592980ee7b7c663944eed5714214320d380bbd9",
        train_source_indices_hash="e870a71b6f7fe7b1be245651a6b46880c3040575f330523aca35627ce920327c",
    )
    PHASE_MAP.mkdir(parents=True, exist_ok=True)
    module.write_json(PHASE_MAP / "randomization-plan.json", plan)
    module.write_json(PHASE_MAP / "bound-manifest.json", bound)
    result = {
        "randomization_plan_hash": plan["randomization_plan_hash"],
        "bound_manifest_hash": module.sha256_bytes(module.canonical_json(bound)),
        "campaign_command_hash": bound["frozen"]["command_hash"],
        "cell_count": len(plan["cells"]),
    }
    module.write_json(PHASE_MAP / "materialization.json", result)
    execute = [
        REMOTE_PYTHON,
        f"{REMOTE_REPO}/scripts/run_phase_map.py",
        "--phase",
        "execute",
        *COMMON,
        "--expected-randomization-plan-hash",
        result["randomization_plan_hash"],
        "--expected-bound-manifest-hash",
        result["bound_manifest_hash"],
    ]
    write_json(PACKET / "execute-argv.json", execute)

    materialized = PACKET / "materialized"
    materialized.mkdir(parents=True, exist_ok=True)
    for name in (
        "randomization-plan.json",
        "bound-manifest.json",
        "materialization.json",
    ):
        shutil.copyfile(PHASE_MAP / name, materialized / name)


if __name__ == "__main__":
    main()

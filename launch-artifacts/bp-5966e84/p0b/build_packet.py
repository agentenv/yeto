#!/usr/bin/env python3
"""Render the reviewed, non-launching P0b bootstrap and harness specification."""

from __future__ import annotations

import base64
import hashlib
import json
import shlex
from pathlib import Path


PACKET = Path(__file__).resolve().parent
RUN_ID = "bp-p0b-5966e84-20260715a"
SOURCE_COMMIT = "8d58208cacafef12cb95f2642b4fa700531151b4"
PREREG_COMMIT = "16d27bc60deb6d8910bf0111c7fb57c9d0eb5b80"
PREREG_PATH = "experiment-specs/best-paper-phase-map-p0-p1-prereg.json"
PREREG_SHA256 = "7cba3c62328b4bfe15fffbc523979274e834e8e720e16f70d79621eaf6ebdb7b"
AMENDMENT_PATH = "docs/AMENDMENT-parallel-cells.md"
AMENDMENT_SHA256 = "e2c87fd6c2ec0e4b91f488b5771334e0befd175560a3e2ccfcf349be1ee8b3dd"
REMOTE_REPO = "/tmp/yeto-best-paper"
REMOTE_RUN = f"/tmp/runs/{RUN_ID}"
ARTIFACT_URI = f"gs://yeto-exp2-52-model-training-497007/{RUN_ID}"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def shell_quote_base64(value: bytes) -> str:
    return shlex.quote(base64.b64encode(value).decode("ascii"))


def build_bootstrap() -> str:
    execute_argv = json.loads((PACKET / "execute-argv.json").read_text())
    parent = PACKET.parent / "p0a-parent"
    parent_manifest = parent / "phase-map-manifest.json"
    parent_replay = parent / "p0a-replay-report.json"
    manifest_payload = shell_quote_base64(parent_manifest.read_bytes())
    replay_payload = shell_quote_base64(parent_replay.read_bytes())
    execute = shlex.join(execute_argv)
    return f"""#!/bin/bash
set -euo pipefail

run={REMOTE_RUN}
repo={REMOTE_REPO}
phase_map="$run/phase-map"
inputs="$run/inputs"
model_archive="$inputs/model-93efa2f.tar"
model_dir="$inputs/model"
data_path="$inputs/train.parquet"
parent="$run/parent"

test "$(git -C "$repo" rev-parse HEAD)" = '{SOURCE_COMMIT}'
test "$(git -C "$repo" rev-parse --is-shallow-repository)" = false
git -C "$repo" cat-file -e '{PREREG_COMMIT}^{{commit}}'
git -C "$repo" cat-file -e '5966e8432e0c350d8968000289656cce2a22fc9d^{{commit}}'
git -C "$repo" cat-file -e '{SOURCE_COMMIT}^{{commit}}'
git -C "$repo" merge-base --is-ancestor \
  {PREREG_COMMIT} {SOURCE_COMMIT}
git -C "$repo" merge-base --is-ancestor \
  5966e8432e0c350d8968000289656cce2a22fc9d {SOURCE_COMMIT}
test "$(git -C "$repo" show '{SOURCE_COMMIT}:{AMENDMENT_PATH}' | sha256sum | awk '{{print $1}}')" = \
  '{AMENDMENT_SHA256}'
test -z "$(git -C "$repo" status --porcelain=v1 --untracked-files=all)"

test ! -e "$inputs" || {{
  echo 'input staging directory already exists' >&2
  exit 31
}}
test ! -e "$parent" || {{
  echo 'parent staging directory already exists' >&2
  exit 32
}}
mkdir -p "$model_dir" "$parent" "$phase_map"

printf %s {manifest_payload} | base64 -d > "$parent/p0a-phase-map-manifest.json"
printf %s {replay_payload} | base64 -d > "$parent/p0a-replay-report.json"
printf '%s  %s\n' \
  '{sha256_file(parent_manifest)}' \
  "$parent/p0a-phase-map-manifest.json" | sha256sum -c -
printf '%s  %s\n' \
  '{sha256_file(parent_replay)}' \
  "$parent/p0a-replay-report.json" | sha256sum -c -

test -f "$run/provider-evidence.json"
test ! -L "$run/provider-evidence.json"
test ! -e "$phase_map/provider-evidence.json"
cp -- "$run/provider-evidence.json" "$phase_map/provider-evidence.json"
cmp -- "$run/provider-evidence.json" "$phase_map/provider-evidence.json"

gcloud storage cp \
  'gs://yeto-exp2-52-model-training-497007/prelaunch/bp-p0a-0af7f4a-20260714a/model-93efa2f.tar#1784089423172165' \
  "$model_archive"
printf '%s  %s\n' \
  '53d15a96a333e33c6a7a9224dbe6392a2480420bd40a327588797d03b625e4c3' \
  "$model_archive" | sha256sum -c -
tar -xf "$model_archive" -C "$model_dir"
(cd "$model_dir" && sha256sum -c model-files.sha256)
test "$(cat "$model_dir/model-id.txt")" = 'HuggingFaceTB/SmolLM2-135M'
test "$(cat "$model_dir/model-revision.txt")" = \
  '93efa2f097d58c2a74874c7e644dbc9b0cee75a2'

gcloud storage cp \
  'gs://yeto-exp2-52-model-training-497007/prelaunch/bp-p0a-0af7f4a-20260714a/train.parquet#1784090284099303' \
  "$data_path"
printf '%s  %s\n' \
  '970f88b3f2fa6758f3b5f94052f4e91b872541a2ba530223b44a779168c51409' \
  "$data_path" | sha256sum -c -

PYTHONPATH="$repo" {execute}
"""


def build_spec(bootstrap: bytes) -> dict[str, object]:
    encoded = base64.b64encode(bootstrap).decode("ascii")
    phase_map = f"{REMOTE_RUN}/phase-map"
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "repo_url": "https://github.com/agentenv/yeto.git",
        "repo_commit": SOURCE_COMMIT,
        "cloud": {
            "provider": "gcp",
            "project": "model-training-497007",
            "zone": "us-central1-c",
            "instance_name": RUN_ID,
            "machine_type": "a2-highgpu-4g",
            "accelerator_count": 4,
            "max_total_accelerators": 16,
            "provisioning_model": "SPOT",
            "termination_action": "DELETE",
            "max_run_duration_seconds": 14400,
            "boot_disk_size_gb": 250,
            "boot_disk_type": "pd-ssd",
            "image": (
                "projects/model-training-497007/global/images/"
                "yeto-optimizer-a100-20260714"
            ),
            "expected_source_image_id": "7290368630472593484",
            "scopes": ["storage-rw"],
            "adopt_only": False,
            "labels": {
                "managed-by": "yeto-optimizer-harness",
                "run-id": RUN_ID,
                "stage": "p0b",
                "campaign": "best-paper-phase-map",
                "draft": "false",
            },
        },
        "execution": {
            "source_mode": "checkout",
            "source_authority": {
                "ref": "refs/heads/experiment/best-paper-phase-map",
                "ancestor_commit": PREREG_COMMIT,
                "ancestor_path": PREREG_PATH,
                "ancestor_sha256": PREREG_SHA256,
            },
            "remote_repo_dir": REMOTE_REPO,
            "remote_run_dir": REMOTE_RUN,
            "command": [
                "/bin/bash",
                "-lc",
                f"printf %s {shlex.quote(encoded)} | base64 -d | /bin/bash",
            ],
            "env": {
                "PATH": (
                    "/home/shou/venv/bin:/snap/bin:/usr/local/sbin:"
                    "/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
                ),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONUNBUFFERED": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "PYTHONPATH": REMOTE_REPO,
            },
            "required_paths": [
                "/bin/bash",
                "/home/shou/venv/bin/python",
                "/usr/bin/nvidia-smi",
                f"{REMOTE_REPO}/scripts/run_phase_map.py",
                f"{REMOTE_REPO}/scripts/compare_diloco.py",
            ],
            "required_executables": [
                "/bin/bash",
                "/home/shou/venv/bin/python",
                "/usr/bin/nvidia-smi",
            ],
            "completion_paths": [
                f"{REMOTE_RUN}/provider-evidence.json",
                f"{REMOTE_RUN}/parent/p0a-phase-map-manifest.json",
                f"{REMOTE_RUN}/parent/p0a-replay-report.json",
                f"{phase_map}/provider-evidence.json",
                f"{phase_map}/phase-map-manifest.json",
                f"{phase_map}/phase-map-acquisition-manifest.json",
                f"{phase_map}/acquisition-seal.json",
                f"{phase_map}/acquisition.sha256",
                f"{phase_map}/phase-map.sha256",
                f"{phase_map}/randomization-plan.json",
                f"{phase_map}/expected-manifest.json",
            ],
            "checksum_manifests": [
                f"{phase_map}/acquisition.sha256",
                f"{phase_map}/phase-map.sha256",
            ],
            "input_checksum_manifests": [],
            "input_provenance_paths": [],
            "legacy_completion": False,
        },
        "artifacts": {"uri": ARTIFACT_URI, "sync_interval_seconds": 60},
        "checks": {"expected_arms": []},
        "analysis": [],
    }


def main() -> None:
    bootstrap = build_bootstrap().encode("utf-8")
    bootstrap_path = PACKET / "bootstrap.sh"
    bootstrap_path.write_bytes(bootstrap)
    spec = build_spec(bootstrap)
    spec_path = PACKET / "optimizer-harness-p0b.json"
    write_json(spec_path, spec)
    decoded = base64.b64decode(spec["execution"]["command"][2].split()[2])
    if decoded != bootstrap:
        raise SystemExit("embedded bootstrap differs from bootstrap.sh")
    write_json(
        PACKET / "build-summary.json",
        {
            "run_id": RUN_ID,
            "operator_label_commit": "5966e84",
            "source_commit": SOURCE_COMMIT,
            "artifact_uri": ARTIFACT_URI,
            "bootstrap_sha256": sha256_bytes(bootstrap),
            "spec_sha256": sha256_file(spec_path),
            "embedded_bootstrap_byte_identical": True,
        },
    )


if __name__ == "__main__":
    main()

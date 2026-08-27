#!/usr/bin/env python3
"""Start the role-isolated actor/critic streaming-DiLoCo syncers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from yeto.syncer_profile import SyncerSemanticProfile

ACTOR_PORT = 29400
CRITIC_PORT = 29401
ROLES = (("actor", ACTOR_PORT), ("critic", CRITIC_PORT))


class SyncerLaunchError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest(path: Path) -> tuple[dict[str, Any], SyncerSemanticProfile]:
    if not path.is_file() or path.is_symlink():
        raise SyncerLaunchError("streaming contract manifest is not a regular file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SyncerLaunchError("streaming contract manifest is malformed") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "yeto.tbench21-sao-streaming-contracts.v1"
        or payload.get("phase") != "final"
        or payload.get("expected_fragments") is None
    ):
        raise SyncerLaunchError("streaming contract manifest is not final")
    profile_record = payload.get("syncer_profile")
    if not isinstance(profile_record, dict):
        raise SyncerLaunchError("streaming contract manifest has no syncer profile")
    profile = SyncerSemanticProfile.from_mapping(profile_record.get("profile"))
    profile.validate_sao()
    if (
        profile.sha256 != profile_record.get("semantic_sha256")
        or profile.total_steps != payload["expected_fragments"]
        or profile.learners != 8
        or profile.quorum != 8
    ):
        raise SyncerLaunchError("syncer profile differs from the one-sweep run")
    return payload, profile


def _port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind(("0.0.0.0", port))
        except OSError:
            return False
    return True


def _binary(path: Path) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise SyncerLaunchError("syncer binary must be an absolute regular file")
    path = path.resolve()
    if not os.access(path, os.X_OK):
        raise SyncerLaunchError("syncer binary is not executable")
    try:
        version = subprocess.run(
            [str(path), "--version"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise SyncerLaunchError("syncer binary cannot execute") from error
    if version != "yeto-syncer 0.1.0":
        raise SyncerLaunchError(f"syncer binary version drifted: {version!r}")
    return path


def _argv(
    binary: Path,
    profile: SyncerSemanticProfile,
    *,
    role: str,
    port: int,
    role_dir: Path,
) -> list[str]:
    values = [
        str(binary),
        "--port",
        str(port),
        "--learners",
        str(profile.learners),
        "--quorum",
        str(profile.quorum),
        "--grace-ms",
        str(profile.grace_ms),
        "--grace-gamma",
        str(profile.grace_gamma),
        "--grace-tau",
        str(profile.grace_tau),
        "--pipeline",
        str(profile.pipeline),
        "--min-round-interval-ms",
        str(profile.min_round_interval_ms),
        "--sync-interval-steps",
        str(profile.sync_interval_steps),
        "--delta-correction",
        profile.delta_correction,
        "--quorum-timeout-s",
        str(profile.quorum_timeout_s),
        "--final-ack-timeout-s",
        str(profile.final_ack_timeout_s),
        "--total-steps",
        str(profile.total_steps),
        "--outer-lr",
        str(profile.outer_lr),
        "--outer-momentum",
        str(profile.outer_momentum),
        "--checkpoint-path",
        str(role_dir / "checkpoint.bin"),
        "--checkpoint-every",
        str(profile.checkpoint_every),
        "--final-state",
        str(role_dir / "final-state.f32"),
        "--event-tape",
        str(role_dir / "events.jsonl"),
        "--learner-weight",
        profile.learner_weight,
        "--max-base-lag",
        str(profile.max_base_lag),
    ]
    if profile.resume:
        values.append("--resume")
    if profile.mark_final_checkpoint:
        values.append("--mark-final-checkpoint")
    if profile.require_profile_binding:
        values.append("--require-profile-binding")
    if profile.policy_sweep_fragments is not None:
        raise SyncerLaunchError(
            f"{role} SAO syncer unexpectedly requested legacy sweeps"
        )
    if profile.learner_budget_steps is not None:
        raise SyncerLaunchError(
            f"{role} SAO syncer unexpectedly requested a budget cutoff"
        )
    return values


def _ready(process: subprocess.Popen[bytes], port: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise SyncerLaunchError(
                f"syncer on port {port} exited with status {process.returncode}"
            )
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            client.settimeout(0.2)
            if client.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise SyncerLaunchError(f"syncer on port {port} did not become reachable")


def launch(manifest_path: Path, binary_path: Path, run_dir: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    binary = _binary(binary_path)
    contract, profile = _load_manifest(manifest_path)
    if any(not _port_available(port) for _, port in ROLES):
        raise SyncerLaunchError("actor or critic syncer port is already occupied")
    run_dir = run_dir.resolve()
    if run_dir.exists() or run_dir.is_symlink():
        raise SyncerLaunchError("syncer run directory must be fresh")
    run_dir.mkdir(mode=0o700, parents=True)

    started: list[tuple[str, int, subprocess.Popen[bytes], Any]] = []
    records = []
    try:
        for role, port in ROLES:
            role_dir = run_dir / role
            role_dir.mkdir(mode=0o700)
            log_path = role_dir / "syncer.log"
            descriptor = os.open(
                log_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            log = os.fdopen(descriptor, "wb", closefd=True)
            argv = _argv(
                binary,
                profile,
                role=role,
                port=port,
                role_dir=role_dir,
            )
            process = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            started.append((role, port, process, log))
            _ready(process, port)
            pid_path = role_dir / "pid"
            pid_path.write_text(f"{process.pid}\n", encoding="ascii")
            pid_path.chmod(0o600)
            records.append(
                {
                    "role": role,
                    "port": port,
                    "pid": process.pid,
                    "argv": argv,
                    "log": str(log_path),
                }
            )
    except BaseException:
        for _, _, process, log in reversed(started):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
            log.close()
        raise
    for _, _, _, log in started:
        log.close()

    result = {
        "schema": "yeto.tbench21-sao-syncer-launch.v1",
        "contracts": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
            "training_contract_sha256": contract["training_contract"]["sha256"],
        },
        "binary": {"path": str(binary), "sha256": _sha256(binary)},
        "semantic_profile_sha256": profile.sha256,
        "expected_fragments": contract["expected_fragments"],
        "processes": records,
    }
    output = run_dir / "launch-manifest.json"
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contracts", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = launch(args.contracts, args.binary, args.run_dir)
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        subprocess.SubprocessError,
        SyncerLaunchError,
    ) as error:
        print(
            f"SAO syncer launch failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

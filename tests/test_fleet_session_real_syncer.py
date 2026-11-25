"""The real Rust syncer, started with the command a per-island fleet's syncer launch
plans, admits the HELLO of every island launch (the run config's two, an island added
at run time, one moved from Modal to AWS) with the profile and syncer_epoch that
island's own launch compiles, and refuses a profile one bit off (no cloud; needs cargo)."""

from __future__ import annotations

import contextlib
import os
import re
import secrets
import shlex
import socket
import subprocess
import time

import pytest

from test_fleet_island_changes import prepare
from test_fleet_session_agreement import RECORD, _secret_cfg
from test_rl_full_parameter_async import needs_cargo
from test_rl_full_parameter_async import syncer_binary  # noqa: F401 - fixture
from yeto import protocol as P
from yeto.fragments import MERGE_AVG, Fragment, FragmentLayout
from yeto.rl import fleet, fleet_islands, fleet_session, island_launches


@pytest.fixture(autouse=True)
def _island_env(monkeypatch):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "fleet-t5b-test-key")
    monkeypatch.setenv("YETO_RL_STORE_S3_ACCESS_KEY_ID", "test-key-id")
    monkeypatch.setenv("YETO_RL_STORE_S3_SECRET_ACCESS_KEY", "test-secret")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _syncer_command(argv: list[str]) -> str:
    from yeto.cli import parse_args
    from yeto.launcher import dry_run_plan

    args = parse_args(argv)
    args.dry_run = True
    prepare(args)
    return dry_run_plan(args)["role_syncers"][0]["command"]


@needs_cargo
def test_the_real_syncer_admits_every_island_launch_s_hello(syncer_binary, tmp_path):  # noqa: F811
    cfg = _secret_cfg(tmp_path)
    plan = fleet.plan_of(cfg)
    members = island_launches.trainer_members(cfg, plan)
    entry = dict(cfg["islands"][-1]["samplers"][0], name="sglang-north")
    added, _ = fleet_islands.compile_island(cfg, plan, k=2, name="north", gpu="modal:1x2xh100", usd_per_h=8.0,
                                            samplers=[entry])
    moved = {**members[0], "gpu": "aws:1xh100@us-west-2"}
    islands = [fleet_session.launch_session(island_launches.launch_argv(m, syncer=RECORD), m["name"])
               for m in (moved, members[1], added)]
    syncer = fleet_session.launch_session(cfg["trainer"]["argv"], "syncer launch")
    port = _free_port()
    out = tmp_path / "syncer-out"
    out.mkdir()
    command = re.sub(r"\$\(test -f [^)]*\)", "", _syncer_command([str(a) for a in cfg["trainer"]["argv"]]))
    command = command.replace("~/yeto-output", str(out)).replace("~/yeto-syncer", str(syncer_binary))
    command = re.sub(r"--port \d+", f"--port {port}", command)
    key = secrets.token_hex(32)
    log = (tmp_path / "syncer.log").open("w")
    proc = subprocess.Popen(shlex.split(command), stdout=log, stderr=subprocess.STDOUT, cwd=tmp_path,
                            env={**os.environ, P.ISLAND_HMAC_KEY_ENV: key})
    try:
        for _ in range(200):
            with contextlib.suppress(OSError):
                socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                break
            assert proc.poll() is None, (tmp_path / "syncer.log").read_text()[-2000:]
            time.sleep(0.1)
        layout = FragmentLayout([Fragment(MERGE_AVG, [(f"w{j}", 1024)]) for j in range(int(plan_fragments(cfg)))])

        def hello(learner: int, sha: str, epoch: int) -> str | None:
            """The syncer's refusal, None when admitted."""
            s = socket.create_connection(("127.0.0.1", port), timeout=5)
            try:
                body = P.encode_hello(learner, P.DTYPE_F32, layout, 1, secrets.randbits(63) | 1, None,
                                      bytes.fromhex(sha), None, syncer_epoch=epoch)
                P.write_frame(s, P.MSG_HELLO, P.seal_hello(body, key.encode()))
                s.settimeout(3)
                try:
                    kind, payload = P.read_frame(s)
                except socket.timeout:
                    return None
                except ConnectionError as error:
                    return f"closed: {error}"
                return bytes(payload).decode("utf-8", "replace") if kind == P.MSG_ERROR else None
            finally:
                s.close()

        flipped = bytes.fromhex(syncer.profile_sha256)
        flipped = (bytes([flipped[0] ^ 1]) + flipped[1:]).hex()
        assert "semantic profile" in (hello(0, flipped, syncer.syncer_epoch) or "")
        for island in islands:
            assert island.profile_sha256 == syncer.profile_sha256, island.name
            assert hello(island.learner_id, island.profile_sha256, island.syncer_epoch) is None, island.name
    finally:
        proc.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(10)
        if proc.poll() is None:
            proc.kill()
        log.close()


def plan_fragments(cfg) -> str:
    argv = cfg["trainer"]["argv"]
    return argv[argv.index("--fragments") + 1]

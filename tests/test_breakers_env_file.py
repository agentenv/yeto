"""A pool's ``agentic_env_file`` reads the same in the breaker views as in the sampler.

The file is a ``docker run --env-file`` (the agentic sampler's docker route; the process
route makes it safe to source so it reads the same): values are literal, so a quoted
value keeps its quotes and the harness refuses ``"6"``. The views read it that way too,
``yeto run up`` and the pool dry run refuse a quoted value a breaker reads and an
unreadable file (the sampler could not start), and ``show`` / ``check`` warn instead of
showing defaults without a word.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from yeto.breakers import cli as bcli
from yeto.breakers.rules_facts import env_file_problems, pool_sources
from yeto.launch import agentic_setup
from yeto.launch.rollout.pools import PoolRef
from yeto.rl import up

ROOT = Path(__file__).resolve().parents[1]
K31 = ROOT / "examples" / "k31"
# the customer kit's agentic.env (oneclick-config/templates/agentic.env, placeholders filled) plus
# the setup retries a customer adds
CUSTOMER_ENV = """YETO_SANDBOX_RUN_LABEL=k31-run1
YETO_RL_TAPE_UPLOAD_S=30
YETO_RL_STORE_PART_CHECKSUM=0
YETO_SAMPLE_BUFFER_PAYLOAD_CODEC=zstd
YETO_AGENTENV_COLD_START=1
YETO_AGENTENV_REQUEST_TIMEOUT_S=600
YETO_AGENTENV_MAX_ACTIVE=32
YETO_AGENTENV_STARTS_PER_MINUTE=20
HARBOR_SETUP_RETRIES=6
"""


def _env(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "agentic.env"
    path.write_text(text)
    return path


def _kit(tmp_path: Path, env_text: str | None) -> Path:
    spec = tmp_path / "sampler-spec.json"
    shutil.copy(K31 / "sampler-spec.nc-mi355x-vllm-agentic-262k.json", spec)
    pool = json.loads((K31 / "pool.nc-mi355x-agentic-262k.json").read_text())
    env = _env(tmp_path, env_text) if env_text is not None else tmp_path / "missing.env"
    pool.update(spec=str(spec), run_prefix="k31/run1", agentic_env_file=str(env))
    (tmp_path / "pool.json").write_text(json.dumps(pool))
    run = json.loads((K31 / "run.agentic-262k.json").read_text())
    run["pool"] = {"config": str(tmp_path / "pool.json")}
    run["trainer"]["argv"] = ["100" if a == "<total-steps>" else a for a in run["trainer"]["argv"]]
    path = tmp_path / "run.json"
    path.write_text(json.dumps(run))
    return path


def _cli(argv):
    p = argparse.ArgumentParser()
    bcli.add_parser(p.add_subparsers(dest="command"))
    return bcli.main(p.parse_args(bcli.normalize_argv(argv)))


def test_the_reader_reads_like_docker(tmp_path):
    path = _env(tmp_path, "﻿# a comment\n\n  A=1\nB=\"2\"\nC='3'\nD=4 \nE\nF=x=y\n")
    assert agentic_setup.read_docker_env_file(path) == {"A": "1", "B": '"2"', "C": "'3'", "D": "4 ", "F": "x=y"}
    with pytest.raises(ValueError, match="contains whitespace"):
        agentic_setup.read_docker_env_file(_env(tmp_path, "export A=1\n"))
    with pytest.raises(ValueError, match="no variable name"):
        agentic_setup.read_docker_env_file(_env(tmp_path, "=1\n"))


def test_the_harness_refuses_a_quoted_value_the_views_now_see_too(tmp_path):
    """What the sampler does with HARBOR_SETUP_RETRIES="6" on the docker route: the harness reads
    the literal '"6"' and refuses it. The views read the same literal value."""
    from yeto.rl.agentic import HarnessConfigError
    from yeto.rl.harness.harbor import breaker_settings

    with pytest.raises(HarnessConfigError, match="HARBOR_SETUP_RETRIES"):
        breaker_settings.setting("harness.setup_retries.max_n", {"HARBOR_SETUP_RETRIES": '"6"'})
    env = _env(tmp_path, 'HARBOR_SETUP_RETRIES="6"\n')
    assert pool_sources({"agentic_env_file": str(env)}).env == {"HARBOR_SETUP_RETRIES": '"6"'}


def test_the_customer_env_file_has_no_problem(tmp_path):
    env = _env(tmp_path, CUSTOMER_ENV)
    assert env_file_problems({"agentic_env_file": str(env)}) == []
    assert pool_sources({"agentic_env_file": str(env)}).env["HARBOR_SETUP_RETRIES"] == "6"
    assert env_file_problems({}) == []


def test_a_quoted_breaker_value_is_a_problem(tmp_path):
    env = _env(tmp_path, CUSTOMER_ENV.replace("HARBOR_SETUP_RETRIES=6", 'HARBOR_SETUP_RETRIES="6"'))
    (problem,) = env_file_problems({"agentic_env_file": str(env)})
    assert 'HARBOR_SETUP_RETRIES="6" is quoted' in problem and "write HARBOR_SETUP_RETRIES=6" in problem


def test_a_quoted_value_no_breaker_reads_is_left_alone(tmp_path):
    env = _env(tmp_path, 'YETO_SANDBOX_RUN_LABEL="k31"\n')
    assert env_file_problems({"agentic_env_file": str(env)}) == []


def test_an_unreadable_or_docker_refused_file_is_a_problem(tmp_path):
    (missing,) = env_file_problems({"agentic_env_file": str(tmp_path / "nope.env")})
    assert "unreadable" in missing and "defaults" in missing
    env = _env(tmp_path, "export HARBOR_SETUP_RETRIES=6\n")
    (export,) = env_file_problems({"agentic_env_file": str(env)})
    assert "contains whitespace" in export


def test_run_up_preflight_refuses_the_env_file_problems(tmp_path):
    def run(env_text):
        sub = tmp_path / str(abs(hash(env_text)))
        sub.mkdir()
        cfg = json.loads(_kit(sub, env_text).read_text())
        refs = [PoolRef("pool", cfg["pool"]["config"])]
        return up.Run.env_file_problem(SimpleNamespace(pools=refs))

    assert run(CUSTOMER_ENV) is None
    assert "unreadable" in run(None)
    assert "is quoted" in run('YETO_AGENTENV_MAX_ACTIVE="32"\n')


def test_the_pool_dry_run_refuses_the_env_file_problems(tmp_path, capsys, monkeypatch):
    import subprocess

    from test_nc_pool_run import _cfg

    monkeypatch.setenv("YETO_NC_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("YETO_NC_ACCOUNTS", "1,4")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail(f"no subprocess here: {a}"))

    from yeto.launch.nc import cli as nc_cli

    _doc, path = _cfg(tmp_path, target=2, agentic_env_file=str(tmp_path / "missing.env"))
    assert nc_cli.main(["pool", "--config", path, "--dry-run"]) == 1
    out = capsys.readouterr().out
    assert "problem: agentic_env_file" in out and "unreadable" in out
    _env(tmp_path, CUSTOMER_ENV)
    _doc, path = _cfg(tmp_path, target=2, agentic_env_file=str(tmp_path / "agentic.env"))
    nc_cli.main(["pool", "--config", path, "--dry-run"])
    assert "problem: agentic_env_file" not in capsys.readouterr().out


@pytest.mark.parametrize("command", ["show", "check"])
def test_show_and_check_warn_about_an_unreadable_env_file(tmp_path, capsys, command):
    path = _kit(tmp_path, None)
    _cli(["breaker", command, "--config", str(path)])
    out = capsys.readouterr().out
    assert "WARNING: agentic_env_file" in out and "unreadable" in out and "defaults" in out
    if command == "show":  # ahead of the rows it is about
        assert out.index("WARNING: agentic_env_file") < out.index("harness.setup_retries")


def test_show_is_quiet_for_the_customer_env_file(tmp_path, capsys):
    path = _kit(tmp_path, CUSTOMER_ENV)
    assert _cli(["breaker", "show", "--config", str(path)]) == 0
    out = capsys.readouterr().out
    assert "WARNING: agentic_env_file" not in out

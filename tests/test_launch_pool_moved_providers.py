"""The provider modules moved from yeto.launch.nc to yeto.launch.pool (unified-sampler-pool
7.3): the old names forward to the same module objects, old ``provider.factory`` strings
keep working, and the pool commands run under both entries."""

from __future__ import annotations

import ast
import importlib
import json
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MOVED = ("sky_provider", "modal_provider", "daytona_provider")


def _run(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ, **(env or {}),
               PYTHONPATH=os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")]))
    return subprocess.run([sys.executable, *args], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize("name", MOVED)
def test_the_old_name_is_the_moved_module(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = importlib.import_module(f"yeto.launch.nc.{name}")
    new = importlib.import_module(f"yeto.launch.pool.{name}")
    assert old is new and new.__name__ == f"yeto.launch.pool.{name}"
    nc = importlib.import_module("yeto.launch.nc")
    assert getattr(nc, name) is new  # attribute access through the old package too


def test_importing_an_old_name_warns_once_and_forwards():
    probe = "\n".join([
        "import importlib, warnings",
        f"for m in {MOVED!r}:",
        "    with warnings.catch_warnings(record=True) as seen:",
        "        warnings.simplefilter('always')",
        "        old = importlib.import_module('yeto.launch.nc.' + m)",
        "    moved = [w for w in seen if issubclass(w.category, DeprecationWarning) and 'moved to' in str(w.message)]",
        "    assert len(moved) == 1, (m, [str(w.message) for w in seen])",
        "    assert str(moved[0].message).startswith(f'yeto.launch.nc.{m} moved to yeto.launch.pool.{m}')",
        "    assert old is importlib.import_module('yeto.launch.pool.' + m)",
        "print('ok')",
    ])
    out = _run("-c", probe)
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr[-2000:]


def test_an_old_factory_string_still_reads_as_the_same_provider():
    from yeto.launch.nc.provider import provider_kind
    from yeto.launch.pool.provider import resolve

    for prefix in ("yeto.launch.nc", "yeto.launch.pool"):
        cfg = {"provider": {"factory": f"{prefix}.sky_provider:from_config"}}
        assert provider_kind(cfg) == "sky"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = resolve("yeto.launch.nc.sky_provider:from_config")
    assert old is resolve("yeto.launch.pool.sky_provider:from_config")
    assert provider_kind({"provider": {"factory": "my_pkg.provider:make"}}) == "factory"


def _help(entry: str, *args: str) -> str:
    out = _run("-m", entry, *args, "--help")
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout


def _commands(text: str) -> tuple[set[str], set[str]]:
    """The subcommands and the flags a help text offers (argparse wraps by the prog's width)."""
    choices = set().union(*(set(m.split(",")) for m in re.findall(r"\{([a-z0-9,-]+)\}", text)))
    return choices, set(re.findall(r"(?<![\w-])--?[a-z][a-z0-9-]*", text))


@pytest.mark.parametrize("sub", [(), ("pool",), ("pool-status",), ("pool-stop",), ("pool-scale",)])
def test_the_neutral_and_the_old_entry_offer_the_same_commands(sub):
    new, old = _help("yeto.launch.pool", *sub), _help("yeto.launch.nc", *sub)
    assert new.startswith("usage: python -m yeto.launch.pool") and "yeto.launch.nc" not in new
    assert old.startswith("usage: python -m yeto.launch.nc")
    assert _commands(new) == _commands(old) and _commands(new) != (set(), set())


def test_the_neutral_and_the_old_entry_dry_run_a_pool_alike(tmp_path):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({
        "event_tape": "/work/sampler/events.jsonl", "prompts": {"groups_per_round": 4}, "num_rollouts": 8,
        "max_backlog_groups": 64, "max_prompt_tokens": 512, "sampling": {"max_tokens": 64, "temperature": 1.0},
        "engine": {"factory": "yeto.rl.adapters.sampler_engines.from_spec:build", "adapter": "fake", "kwargs": {}},
        "reward": "yeto.rl.numeric_answer_reward:score", "reward_kind": "miles"}))
    store = tmp_path / "store.env"
    store.write_text("R2_BUCKET=<bucket>\nAWS_ACCESS_KEY_ID=x\nAWS_SECRET_ACCESS_KEY=y\n")
    store.chmod(0o600)
    cfg = tmp_path / "pool.json"
    cfg.write_text(json.dumps({"target": 2, "accounts": [1, 4], "shard_base": "nc", "run": "run-a",
                               "spec": str(spec), "run_prefix": "run-a", "r2_env_file": str(store)}))
    env = {"YETO_NC_STATE_DIR": str(tmp_path / "state"), "YETO_NC_ACCOUNTS": "1,4",
           "NC_MANAGE": "true", "NC_QUERY": "true"}
    outs = {}
    for entry in ("yeto.launch.pool", "yeto.launch.nc"):
        out = _run("-m", entry, "pool", "--config", str(cfg), "--dry-run", env=env)
        assert out.returncode == 0, out.stderr[-2000:]
        outs[entry] = re.sub(r"\d\d:\d\d:\d\dZ", "<TS>", out.stdout)  # the printed hints name the old entry
    assert "account 1: 1 x mi355x node, shard nc-0" in outs["yeto.launch.nc"]
    assert outs["yeto.launch.pool"] == outs["yeto.launch.nc"]


def _package(rel: Path) -> str:
    parts = list(rel.with_suffix("").parts)
    return ".".join(parts if rel.name == "__init__.py" else parts[:-1])


def test_no_code_uses_the_old_names():
    """New code imports yeto.launch.pool.<provider>; the old names are for configs and
    callers outside this repository."""
    shims = {Path("yeto/launch/nc") / f"{m}.py" for m in MOVED}
    dotted = re.compile(r"\byeto[./]launch[./]nc[./](%s)\b" % "|".join(MOVED))
    found = []
    for path in sorted([*ROOT.glob("yeto/**/*.py"), *ROOT.glob("tests/**/*.py")]):
        rel = path.relative_to(ROOT)
        if rel in shims or rel.name == Path(__file__).name:
            continue
        src = path.read_text()
        for i, line in enumerate(src.splitlines(), 1):
            if dotted.search(line) and "SKY_FACTORY_PREFIXES" not in line:
                found.append(f"{rel}:{i}: {line.strip()}")
        if rel.parts[0] != "yeto":
            continue
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.ImportFrom) and node.level:
                pkg = _package(rel).split(".")
                base = ".".join(pkg[: len(pkg) - (node.level - 1)] + (node.module.split(".") if node.module else []))
                if base == "yeto.launch.nc" and any(a.name in MOVED for a in node.names) \
                        or any(base == f"yeto.launch.nc.{m}" for m in MOVED):
                    found.append(f"{rel}:{node.lineno}: relative import of an old name")
    assert not found, "use yeto.launch.pool.<provider>:\n" + "\n".join(found)

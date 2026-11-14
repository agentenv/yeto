"""Every configuration example of the Kimi K3.1 customer guide and of examples/k31 goes
through the code that reads it: a run config through ``load_config`` (and its pools through
``pool_config``), a pool config through ``pool_config``, a command line through its parser
(nothing is ordered: the parser only parses), an algorithm spec through ``AlgorithmSpec``.
A guide example the code refuses fails here, not on the customer's control host."""

from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path
from typing import Any

import pytest

from yeto.launch.nc import cli as nc_cli
from yeto.launch.nc.provider import provider_kind
from yeto.launch.rollout.pools import resolve
from yeto.rl import up

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "KIMI_K31_ASYNC_RL_GUIDE.md"
K31 = ROOT / "examples" / "k31"

# Fenced blocks that are not configuration: (first line of the block, why it is skipped).
# Anything else without a JSON / command line to check fails ``test_every_guide_block_is_checked_or_listed``.
NOT_CONFIG = {
    "+-----------------------------------------------+": "architecture diagram",
    "git clone <yeto-repo> && cd yeto": "installation shell steps, no yeto config",
    "hf download <org>/<name> --revision <40-hex sha> --local-dir <dir>/<name>": "third-party CLI (Hugging Face)",
    "aws s3 sync <dir>/<name> s3://<bucket>/models/<name> --exclude \".cache/*\" \\": "third-party CLI (AWS)",
    "#!/bin/sh": "sandbox-host forced command script, not a yeto config",
    "ssh -N -L <port>:127.0.0.1:<port> <control-host>": "plain ssh port forward",
    "docker:ghcr.io/agentenv/yeto-miles-k31@sha256:": "an image reference, not a config",
    "docker:docker.io/vllm/vllm-openai-rocm@sha256:": "an image reference, not a config",
    "MILES_SRC=<miles-fork-clone> SGLANG_SRC=<sglang-fork-clone> \\": "image build script, needs the fork clones",
    "# /etc/systemd/system/yeto-run-<run>.service": "systemd unit (its ExecStart yeto line is parsed below)",
}

# Placeholder values for the guide's angle-bracket placeholders that must be numbers / ids.
NUMBERS = {"price_usd_h": 3.5}
CMD_VALUES = {"<N>": "2", "<ticket>": "t-1", "<id>": "1", "<name>": "nc"}


def _blocks() -> list[tuple[int, str, str]]:
    text = GUIDE.read_text()
    out = []
    for m in re.finditer(r"^```(\w*)\n(.*?)^```", text, re.S | re.M):
        out.append((text[:m.start()].count("\n") + 1, m.group(1), m.group(2)))
    return out


def _json(body: str) -> dict[str, Any]:
    """A JSON block: a whole object, or a fragment of keys (``"key": ...``) of one."""
    try:
        return json.loads(body)
    except ValueError:
        return json.loads("{" + body + "}")


def _filled(doc: Any, tmp: Path, key: str = "") -> Any:
    if isinstance(doc, dict):
        return {k: _filled(v, tmp, k) for k, v in doc.items()}
    if isinstance(doc, list):
        if key == "accounts" and all(isinstance(v, str) and v.startswith("<") for v in doc):
            return list(range(1, len(doc) + 1))
        return [_filled(v, tmp, key) for v in doc]
    if isinstance(doc, str) and "<" in doc:
        if key in NUMBERS:
            return NUMBERS[key]
        if key == "YETO_NC_ACCOUNTS":
            return ",".join(str(i) for i in range(1, doc.count(",") + 2))
        return str(tmp / re.sub(r"[<>/ :]", "_", doc))
    return doc


def _json_blocks():
    return [(ln, _json(body)) for ln, lang, body in _blocks() if lang == "json"]


def _example(kind: str) -> tuple[int, dict[str, Any]]:
    """The guide's full run config (``run`` + ``trainer``) or pool config (``shard_base`` + ``accounts``)."""
    for ln, doc in _json_blocks():
        if kind == "run" and {"run", "store", "trainer"} <= set(doc):
            return ln, doc
        if kind == "pool" and {"shard_base", "accounts"} <= set(doc):
            return ln, doc
    raise AssertionError(f"the guide has no full {kind} config example")


def _check_run(cfg: dict[str, Any], tmp: Path) -> None:
    """``yeto run up``'s own reading of a run config: load_config, then every pool through pool_config."""
    _, pool = _example("pool")
    pool_file = tmp / "pool.json"
    pool_file.write_text(json.dumps(_filled(pool, tmp)))
    cfg = _filled(cfg, tmp)
    if "pool" in cfg:
        cfg["pool"]["config"] = str(pool_file)
    for entry in cfg.get("samplers") or ():
        if "config" in entry:
            entry["config"] = str(pool_file)
    run_file = tmp / "run.json"
    run_file.write_text(json.dumps(cfg))
    loaded = up.load_config(str(run_file))
    for ref in resolve(loaded, tmp / "state"):
        nc_cli.pool_config(ref.load())
        _check_provider(ref.load())


def _check_provider(cfg: dict[str, Any]) -> None:
    """A provider section through its provider's own reading (a stand-in SDK: nothing is called)."""
    kind = provider_kind(cfg)
    run = str(cfg.get("run") or cfg["shard_base"])
    if kind == "daytona":
        from yeto.launch.nc import daytona_provider  # noqa: PLC0415

        daytona_provider.from_config(cfg, run, sdk=object(), client=object())
    elif kind == "sky":
        from yeto.launch.nc import sky_provider  # noqa: PLC0415

        sky_provider.from_config(cfg, run, sky=object())
    elif kind == "modal":
        from yeto.launch.nc import modal_provider  # noqa: PLC0415

        modal_provider.from_config(cfg, run, client=object())


def _check_pool(cfg: dict[str, Any], tmp: Path) -> None:
    cfg = _filled(cfg, tmp)
    nc_cli.pool_config(cfg)
    _check_provider(cfg)


# A provider fragment may carry the pool keys that go with it (slot count, target, shard base).
POOL_PROVIDER_KEYS = {"provider", "target", "slots", "shard_base"}
RUN_KEYS = {"starvation", "controller", "rollout", "samplers", "budget", "first_group", "dashboard"}


def _cases():
    run_ln, run = _example("run")
    pool_ln, pool = _example("pool")
    cases = []
    for ln, doc in _json_blocks():
        if ln == run_ln:
            cases.append(pytest.param("run", run, id=f"guide:{ln}:run-config"))
        elif ln == pool_ln:
            cases.append(pytest.param("pool", pool, id=f"guide:{ln}:pool-config"))
        elif set(doc) <= RUN_KEYS:  # a run config fragment: in the full example
            merged = {k: v for k, v in run.items() if not ("samplers" in doc and k == "pool")}
            cases.append(pytest.param("run", {**merged, **doc}, id=f"guide:{ln}:run-fragment"))
        elif "provider" in doc and set(doc) <= POOL_PROVIDER_KEYS:  # a provider section: in the full pool example
            merged = {k: v for k, v in pool.items() if k != "accounts"}
            cases.append(pytest.param("pool", {**merged, "slots": [1, 2, 3], **doc},
                                      id=f"guide:{ln}:pool-provider"))
        else:
            cases.append(pytest.param("unknown", doc, id=f"guide:{ln}:unclassified"))
    return cases


@pytest.mark.parametrize(("kind", "doc"), _cases())
def test_guide_json_example_is_accepted(kind, doc, tmp_path, monkeypatch):
    monkeypatch.setattr(nc_cli.tunnel, "state_dir", lambda: tmp_path / "nc-state", raising=False)
    assert kind != "unknown", f"guide JSON block of unknown shape {sorted(doc)}: teach this test where it goes"
    (_check_run if kind == "run" else _check_pool)(doc, tmp_path)


def _command_lines() -> list[tuple[int, str]]:
    out = []
    for ln, lang, body in _blocks():
        for i, line in enumerate(body.splitlines()):
            line = line.split("  #")[0].strip()
            line = re.sub(r"^ExecStart=\S*/yeto ", "yeto ", line)  # the systemd unit's command
            if re.match(r"(yeto|python -m yeto\.launch\.nc) ", line):
                out.append((ln + 1 + i, line))
    return out


class _Parsed(Exception):
    pass


def _argv(line: str, tmp: Path) -> list[str]:
    line = re.sub(r"\[(--[\w-]+)\]", r"\1", line)  # an optional flag in brackets: given
    for key, value in CMD_VALUES.items():
        line = line.replace(key, value)
    line = re.sub(r"<([^>]+)>", lambda m: str(tmp / re.sub(r"[/ ]", "_", m.group(1))), line)
    return shlex.split(line)


@pytest.mark.parametrize(("ln", "line"), _command_lines(), ids=lambda v: str(v) if isinstance(v, int) else None)
def test_guide_command_line_parses(ln, line, tmp_path, monkeypatch):
    argv = _argv(line, tmp_path)
    if argv[0] == "yeto":
        from yeto.cli import build_parser  # noqa: PLC0415

        parser = build_parser()
        parse = parser.parse_args
        argv = argv[1:]
    else:  # python -m yeto.launch.nc: its parser lives in main(); stop right after parsing
        argv = argv[3:]
        original = argparse.ArgumentParser.parse_args

        def stop(self, args=None, namespace=None):
            original(self, args, namespace)
            raise _Parsed

        monkeypatch.setattr(argparse.ArgumentParser, "parse_args", stop)

        def parse(args):
            with pytest.raises(_Parsed):
                nc_cli.main(args)
    try:
        parse(argv)
    except SystemExit as error:
        pytest.fail(f"guide line {ln}: {line!r} is refused by the parser (exit {error.code})")


def test_every_guide_block_is_checked_or_listed():
    listed = set(NOT_CONFIG)
    for ln, lang, body in _blocks():
        if lang == "json":
            continue
        first = body.splitlines()[0].strip() if body.strip() else ""
        if any(re.match(r"\s*(yeto|python -m yeto\.launch\.nc) ", line) for line in body.splitlines()):
            continue
        assert any(first.startswith(k) for k in listed), f"guide line {ln}: unchecked block {first!r}"


def test_the_guide_has_the_examples_this_test_reads():
    assert _example("run") and _example("pool")
    assert len(_command_lines()) >= 15


@pytest.mark.parametrize("path", sorted(K31.glob("algorithm-spec.*.json")), ids=lambda p: p.name)
def test_k31_algorithm_spec_example_loads(path):
    from yeto.rl.engine.algorithm import AlgorithmSpec  # noqa: PLC0415

    assert AlgorithmSpec.from_json_file(path) is not None


def test_every_k31_example_json_is_covered():
    """Sampler / convertible specs build in test_example_sampler_specs_validate; algorithm specs here."""
    for path in K31.glob("*.json"):
        doc = json.loads(path.read_text())
        assert path.name.startswith("algorithm-spec.") or isinstance(doc.get("engine"), dict), path.name

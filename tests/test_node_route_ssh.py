"""The ssh pool on the node runtime (yeto.node.route), its default: the node runs ``python -m
yeto.node run``, which starts the sampler; the route keeps its own preflight, code upload, env
file and container flags. ``node_runtime: false`` keeps the sampler start of before."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from tests.test_ssh_sampler_pool import FakeRun, conf, pool_cfg
from yeto.launch import ssh_sampler_pool as sp
from yeto.launch.nc import cli
from yeto.launch.node_inventory import parse_node
from yeto.node import records
from yeto.rl.engine.object_store import DirectoryObjectStore
from yeto.rl.engine.sampler_resume import MEMBERS_DIR

REPO = Path(__file__).resolve().parents[1]


class _NoArchive:
    stdout = None

    def __init__(self, *a, **k):
        pass

    def wait(self):
        return 0


class NodeRun(FakeRun):
    """FakeRun with a node store that holds nothing yet (``cat`` of an absent file)."""

    def __call__(self, argv, **kw):
        out = super().__call__(argv, **kw)
        if argv[0] == "ssh" and "YETO_NODE_ABSENT_" in argv[-1]:
            return SimpleNamespace(returncode=0, stdout="YETO_NODE_ABSENT_1\n", stderr="")
        return out


def _start(tmp_path, monkeypatch, **section):
    monkeypatch.setattr(sp.subprocess, "Popen", _NoArchive)
    monkeypatch.setattr(cli.tasks, "sampler_spec", lambda spec, sampler_id, shards: dict(spec, sampler_id=sampler_id))
    monkeypatch.setattr(cli.tasks, "apply_sampler_overrides", lambda spec, **kw: spec)
    run = NodeRun()
    ops = sp.SshSamplerPoolOps(pool_cfg(tmp_path), conf=conf(**section), run=run, state_dir=tmp_path)
    t = ops.order(0)
    m = SimpleNamespace(account=0, ticket=t, shard="f09-0", data_slot=None, replaces_shard=None,
                        predecessor_lost=False)
    ops.start(m, ["f09-0"])
    return run, t


def test_the_container_runs_the_node_runtime_with_the_routes_own_flags(tmp_path, monkeypatch):
    new, t = _start(tmp_path, monkeypatch)  # the default
    (tmp_path / "old").mkdir()
    old, _ = _start(tmp_path / "old", monkeypatch, node_runtime=False)
    run_new = next(c[0][-1] for c in new.calls if c[0][0] == "ssh" and "run -d" in c[0][-1])
    run_old = next(c[0][-1] for c in old.calls if c[0][0] == "ssh" and "run -d" in c[0][-1])
    head = lambda cmd: cmd.split("--entrypoint bash")[0]  # noqa: E731 - every flag before the image
    assert head(run_new) == head(run_old)
    assert "-m yeto.node run --assignment" in run_new and "yeto.rl.engine.engine_sampler" not in run_new
    assert f"--assignment {t} --store /srv/yeto/run0/store" in run_new and "--log -" in run_new
    token_put = next(c for c in new.calls if c[0][0] == "ssh" and "f09-0.token" in c[0][-1])
    token = token_put[1]["input"]
    assert all(token not in str(c[0]) for c in new.calls)  # the token only ever on stdin
    written = next(c for c in new.calls if c[0][0] == "ssh" and f"assignments/{t}.json" in c[0][-1]
                   and c[1].get("input"))
    doc = json.loads(written[1]["input"])
    assert doc["source"] == "ssh" and doc["spec_paths"] == "host" and doc["cards"] is None
    assert doc["spec"]["store"] == "/srv/yeto/run0/store" and doc["token_sha256"] == records.token_digest(token)
    profile = json.loads(next(c for c in new.calls if c[0][0] == "ssh" and "profile.json" in c[0][-1])[1]["input"])
    assert profile["container"] == "process" and profile["paths"]["work"] == "/srv/yeto/run0"


def test_node_runtime_is_the_default_and_false_keeps_the_sampler_start(tmp_path, monkeypatch):
    assert conf().node_runtime is True and conf(node_runtime=True).node_runtime is True
    assert conf(node_runtime=False).node_runtime is False
    from tests.test_ssh_sampler_pool import npu_node, provider_section

    top = {"provider": provider_section(), "node_runtime": False}  # the pool config's own key
    assert sp.parse_config(top, inventory=[npu_node()]).node_runtime is False
    default, _ = _start(tmp_path, monkeypatch)
    cmds = [c[0][-1] for c in default.calls if c[0][0] == "ssh"]
    assert any("-m yeto.node run --assignment" in c for c in cmds)
    assert not any("yeto.rl.engine.engine_sampler --spec" in c for c in cmds)
    (tmp_path / "off").mkdir()
    old, _ = _start(tmp_path / "off", monkeypatch, node_runtime=False)
    cmds = [c[0][-1] for c in old.calls if c[0][0] == "ssh"]
    assert any("yeto.rl.engine.engine_sampler --spec" in c for c in cmds)
    assert not any("yeto.node" in c or "assignments/" in c for c in cmds)


class LocalNode:
    """The ssh pool's ``run`` on this host: an ssh command runs in a local shell (stdin
    included); the detached venv launch runs the start script in the foreground."""

    def __init__(self):
        self.calls = []

    def __call__(self, argv, **kw):
        self.calls.append(argv)
        if argv[0] != "ssh":
            return subprocess.run(argv, **kw)
        cmd = argv[-1]
        if "setsid bash -c" in cmd:  # venv_launch_command: /proc checks are Linux only; run it here
            script = cmd.split("bash ", 2)[2].split(" >>")[0].strip("'")
            log = Path(script).parent.parent / "logs" / (Path(script).name.split(".")[0] + ".log")
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("ab") as out:
                code = subprocess.run(["bash", script], stdout=out, stderr=subprocess.STDOUT, timeout=600).returncode
            return SimpleNamespace(returncode=0 if code == 0 else 1, stdout=f"YETO_SSH_STARTED exit={code}\n",
                                   stderr=log.read_text()[-3000:])
        kw = dict(kw)
        kw.pop("check", None)
        if "stdin" in kw:
            return subprocess.run(["bash", "-c", cmd], stdin=kw["stdin"], stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, timeout=kw.get("timeout"))
        return subprocess.run(["bash", "-c", cmd], input=kw.get("input"), capture_output=True,
                              text=kw.get("text", True), timeout=kw.get("timeout"))


def test_a_venv_node_runs_the_node_runtime_and_the_sampler_end_to_end(tmp_path, monkeypatch):
    work, store = tmp_path / "work", tmp_path / "store"
    (tmp_path / "prompts.jsonl").write_text("".join(json.dumps({"q": f"question {i}", "a": str(i)}) + "\n"
                                                    for i in range(6)))
    spec = {"sampler_id": "x", "engine": {"factory": "yeto.rl.adapters.sampler_engines.from_spec:build",
                                          "adapter": "fake", "kwargs": {"max_model_len": 64}},
            "sampling": {"max_tokens": 6, "n": 2}, "num_rollouts": 2, "max_backlog_groups": 100,
            "max_prompt_tokens": 32, "store": "/s", "reward": "yeto.rl.numeric_answer_reward:score",
            "reward_kind": "miles", "checks": {"require_fixed_answers": False},
            "prompts": {"path": str(tmp_path / "prompts.jsonl"), "input_key": "q", "label_key": "a",
                        "groups_per_round": 2}}
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    cfg = {"target": 1, "slots": [0], "shard_base": "e2e", "spec": str(tmp_path / "spec.json"), "run_prefix": "e2e",
           "provider": {"kind": "ssh", "workdir": str(work), "runtime": "venv", "venv": {"python": sys.executable},
                        "store": str(store)}}  # the node runtime: the default
    node = parse_node({"name": "local", "host": "localhost", "device_family": "cuda", "cpu_arch": "x86_64",
                       "card": "L4", "cards_per_node": 1, "hbm_gib": 24, "host_memory_gib": 64,
                       "runtime": "rootfs"}, index=0)
    run = LocalNode()
    ops = sp.SshSamplerPoolOps(cfg, conf=sp.parse_config(cfg, inventory=[node]), run=run, state_dir=tmp_path)
    # the preflight's side effect only (it makes the workdir's sampler/ and logs/): no card tool here
    monkeypatch.setattr(ops, "preflight", lambda node: [(work / d).mkdir(parents=True, exist_ok=True)
                                                        for d in ("sampler", "logs")] and {})
    monkeypatch.setattr(sp.ssh_transport, "mux_options", lambda route: [])
    t = ops.order(0)
    m = SimpleNamespace(account=0, ticket=t, shard="e2e-0", data_slot=None, replaces_shard=None,
                        predecessor_lost=False)
    ops.start(m, ["e2e-0"])
    status = records.node_status(DirectoryObjectStore(store), t)
    assert status.state == "exited" and status.reason == "finished", (work / "logs" / "e2e-0.log").read_text()[-3000:]
    assert status.ready["shard"] == "e2e-0"
    [member] = [json.loads(p.read_text()) for p in (store / MEMBERS_DIR / "e2e-0").glob("*.json")]
    assert member["node"]["ticket"] == t and member["node"]["source"] == "ssh"
    assert member["leave_reason"] == "FINISHED" and member["rounds"] == 2


# -- the sampler's final environment, the route's own start against the node runtime ---------

IMAGE_ENV = {"PYTHONPATH": "/usr/local/Ascend/cann/python/site-packages", "ASCEND_HOME_PATH": "/usr/local/Ascend",
             "LD_LIBRARY_PATH": "/usr/local/Ascend/lib64"}
# the shell's own, and what a Python interpreter adds to its environment at start (PEP 538 locale
# coercion; macOS) - the sampler, a Python process too, adds the same at its own start
SHELL_VARS = {"PWD", "OLDPWD", "SHLVL", "_", "LC_CTYPE", "__CF_USER_TEXT_ENCODING"}
NODE_ONLY = {"YETO_NODE_TICKET", "YETO_NODE_GENERATION", "YETO_NODE_PROFILE", "YETO_NODE_SUPERVISED",
             "YETO_DEVICE_FAMILY", "YETO_RL_RELEASE_SPREAD_KEY", "YETO_NODE_HEALTH_FILE"}


class EmulatedNode(LocalNode):
    """LocalNode plus a container engine: ``<engine> run -d`` runs its ``bash -c`` body here,
    under the image's environment, its ``--env-file`` and its ``-e`` values (nothing else)."""

    def __call__(self, argv, **kw):
        if argv[0] == "ssh" and " run -d " in argv[-1]:
            import shlex  # noqa: PLC0415

            tokens = shlex.split(argv[-1])
            start = next(i for i in range(len(tokens) - 1) if tokens[i] == "run" and tokens[i + 1] == "-d")
            run = tokens[start:tokens.index("&&", start)]
            env = {"PATH": "/usr/bin:/bin", "HOME": str(Path.home()), **IMAGE_ENV}
            for i, a in enumerate(run):
                if a == "--env-file":
                    env.update(dict(line.split("=", 1) for line in Path(run[i + 1]).read_text().splitlines() if line))
                elif a == "-e":
                    env.update([run[i + 1].split("=", 1)])
            out = subprocess.run(["bash", "-c", run[-1]], env=env, capture_output=True, text=True, timeout=600)
            return SimpleNamespace(returncode=0 if out.returncode == 0 else 1,
                                   stdout="YETO_SSH_STARTED container=x\n", stderr=out.stdout + out.stderr)
        return super().__call__(argv, **kw)


def _final_env(tmp_path, monkeypatch, *, runtime: str, family: str, node_runtime: bool) -> dict:
    work, store, dump = tmp_path / "work", tmp_path / "store", tmp_path / "sampler-env"
    stub = tmp_path / "python"  # the node runtime runs for real; the sampler only records its environment
    stub.write_text(f'#!/bin/bash\nif [ "$2" = yeto.node ]; then exec {sys.executable} "$@"; fi\n'
                    f'env -0 > {dump}\n')
    stub.chmod(0o755)
    source = tmp_path / "set_env.sh"
    source.write_text('export CANN_TOOLKIT=/usr/local/Ascend/toolkit\nexport PYTHONPATH=/cann/acl"${PYTHONPATH:+:$PYTHONPATH}"\n')
    (tmp_path / "spec.json").write_text(json.dumps({"sampler_id": "x", "prompts": {"groups_per_round": 2},
                                                    "buffer_shards": ["e-0"]}))
    provider = {"kind": "ssh", "workdir": str(work), "runtime": runtime, "store": str(store), "devices": [0],
                "node_runtime": node_runtime, "env": {"PYTHONUNBUFFERED": "0", "OPERATOR_KNOB": "7"}}
    if runtime == "venv":
        provider["venv"] = {"python": str(stub), "pythonpath": ["/opt/extra"], "source": [str(source)]}
    else:
        provider["container"] = {"engine": "podman", "image": "img", "python": str(stub)}
    cfg = {"target": 1, "slots": [0], "shard_base": "e", "spec": str(tmp_path / "spec.json"), "run_prefix": "e",
           "provider": provider}
    node = parse_node({"name": "local", "host": "localhost", "device_family": family,
                       "cpu_arch": "x86_64", "card": "910B4" if family == "npu" else "L4", "cards_per_node": 1,
                       "hbm_gib": 64 if family == "npu" else 24, "host_memory_gib": 64,
                       "runtime": "docker" if runtime == "container" else "rootfs"}, index=0)
    ops = sp.SshSamplerPoolOps(cfg, conf=sp.parse_config(cfg, inventory=[node]), run=EmulatedNode(),
                               state_dir=tmp_path)
    monkeypatch.setattr(ops, "preflight", lambda node: [(work / d).mkdir(parents=True, exist_ok=True)
                                                        for d in ("sampler", "logs")] and {})
    monkeypatch.setattr(sp.ssh_transport, "mux_options", lambda route: [])
    monkeypatch.setattr(sp, "container_pull_command", lambda conf, image: "true")
    monkeypatch.setattr(cli.tasks, "sampler_spec", lambda spec, sampler_id, shards: dict(spec, sampler_id=sampler_id))
    monkeypatch.setattr(cli.tasks, "apply_sampler_overrides", lambda spec, **kw: spec)
    m = SimpleNamespace(account=0, ticket=ops.order(0), shard="e-0", data_slot=None, replaces_shard=None,
                        predecessor_lost=False)
    ops.start(m, ["e-0"])
    assert dump.exists(), (work / "logs" / "e-0.log").read_text()[-2000:] if (work / "logs" / "e-0.log").exists() else ""
    env = dict(item.split("=", 1) for item in dump.read_text().split("\0") if "=" in item)
    return {k: v for k, v in env.items() if k not in SHELL_VARS and not k.startswith("YETO_SSH_START")}


import pytest  # noqa: E402


@pytest.mark.parametrize("runtime,family", [("venv", "npu"), ("venv", "cuda"), ("container", "npu"), ("container", "cuda")])
def test_the_samplers_final_environment_is_the_routes_own(tmp_path, monkeypatch, runtime, family):
    (tmp_path / "old").mkdir()
    (tmp_path / "new").mkdir()
    old = _final_env(tmp_path / "old", monkeypatch, runtime=runtime, family=family, node_runtime=False)
    new = _final_env(tmp_path / "new", monkeypatch, runtime=runtime, family=family, node_runtime=True)
    rel = lambda env, root: {k: v.replace(str(tmp_path / root), "<T>") for k, v in env.items()}  # noqa: E731
    old, new = rel(old, "old"), rel(new, "new")
    assert {k: v for k, v in new.items() if k not in NODE_ONLY or k in old} == old
    assert new["YETO_NODE_TICKET"] and new["PYTHONUNBUFFERED"] == "0" and new["OPERATOR_KNOB"] == "7"
    # the node runtime's health check (yeto.hw.node_health): a family with a checker only
    assert ("YETO_NODE_HEALTH_FILE" in new) == (family == "cuda")
    if family == "npu":
        assert new["ASCEND_RT_VISIBLE_DEVICES"] == "0"
    if runtime == "container":  # the image's own PYTHONPATH (CANN) stays behind the yeto tree
        assert new["PYTHONPATH"].endswith(":" + IMAGE_ENV["PYTHONPATH"]) and new["ASCEND_HOME_PATH"]
    else:  # the venv's pythonpath and the source script's entry stay
        assert ":/opt/extra:/cann/acl" in new["PYTHONPATH"] and new["CANN_TOOLKIT"]


def test_shell_objects_raise_on_an_ssh_failure_instead_of_reading_absent():
    from yeto.node.route import ShellObjects  # noqa: PLC0415

    answers = {"absent": SimpleNamespace(returncode=0, stdout="YETO_NODE_ABSENT_1\n"),
               "down": SimpleNamespace(returncode=255, stdout="")}
    assert ShellObjects(lambda cmd, text: answers["absent"], "/s").read("assignments/t.json") is None
    with pytest.raises(RuntimeError, match="exit 255"):
        ShellObjects(lambda cmd, text: answers["down"], "/s").read("assignments/t.json")

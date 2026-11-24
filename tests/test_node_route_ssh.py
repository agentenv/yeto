"""The ssh pool on the node runtime (yeto.node.route): the node runs ``python -m yeto.node
run``, which starts the sampler; the route keeps its own preflight, code upload, env file
and container flags. ``node_runtime: false`` keeps the sampler start of before."""

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


def _start(tmp_path, monkeypatch, **section):
    monkeypatch.setattr(sp.subprocess, "Popen", _NoArchive)
    monkeypatch.setattr(cli.tasks, "sampler_spec", lambda spec, sampler_id, shards: dict(spec, sampler_id=sampler_id))
    monkeypatch.setattr(cli.tasks, "apply_sampler_overrides", lambda spec, **kw: spec)
    run = FakeRun()
    ops = sp.SshSamplerPoolOps(pool_cfg(tmp_path), conf=conf(**section), run=run, state_dir=tmp_path)
    t = ops.order(0)
    m = SimpleNamespace(account=0, ticket=t, shard="f09-0", data_slot=None, replaces_shard=None,
                        predecessor_lost=False)
    ops.start(m, ["f09-0"])
    return run, t


def test_the_container_runs_the_node_runtime_with_the_routes_own_flags(tmp_path, monkeypatch):
    new, t = _start(tmp_path, monkeypatch)
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
    assert doc["source"] == "ssh" and doc["spec_paths"] == "host" and doc["cards"] == [0]
    assert doc["spec"]["store"] == "/srv/yeto/run0/store" and doc["token_sha256"] == records.token_digest(token)
    profile = json.loads(next(c for c in new.calls if c[0][0] == "ssh" and "profile.json" in c[0][-1])[1]["input"])
    assert profile["container"] == "process" and profile["paths"]["work"] == "/srv/yeto/run0"


def test_node_runtime_false_keeps_the_sampler_start(tmp_path, monkeypatch):
    old, _ = _start(tmp_path, monkeypatch, node_runtime=False)
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
                        "store": str(store)}}
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

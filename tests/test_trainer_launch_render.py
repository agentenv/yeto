"""The first launch of a trainer island renders the same requests on every source.

``launcher.run`` starts each ``--gpu`` island through its source (sky cluster, Modal
function, SSH inventory nodes). This test runs the launcher up to the point where
every island is provisioned, with fake sky / Modal / SSH backends, and records what
each backend was asked to do: the sky task and launch options of every cluster, the
Modal island config of every function and the order of the spawns, and the start
script of every SSH node. The golden (``tests/golden/trainer_launch.json``) was
recorded on main before the trainer launch moved behind the provider adapters
(``yeto.launch.trainer_provision``); the record must stay byte for byte.

Cases: a sky fleet (one single-node and one two-node island, a syncer), a Modal fleet
(one single-container and one clustered island), a mixed Modal + sky fleet, a resident
SSH island on two inventory nodes, and the customer K3.1 trainers (the Modal trainer of
``examples/k31/run.agentic-262k.json`` and the bare-metal trainer of
``run.baremetal-b300-262k.json``, argv filled as in ``test_k31_customer_plan_golden``).

Regenerate only for an intended change:
``YETO_UPDATE_TRAINER_LAUNCH_GOLDEN=1 pytest tests/test_trainer_launch_render.py``.
"""

from __future__ import annotations

import json
import os
import re
import sys
import types
from pathlib import Path

import pytest

import yeto.launcher as launcher
import yeto.modal_runner as modal_runner
from test_rl_algorithm_provenance import _fake_sky, _launcher_args  # noqa: F401  (fixture)

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden" / "trainer_launch.json"
IMAGE = "docker:ghcr.io/x/y@sha256:" + "a" * 64


class _Provisioned(Exception):
    """Raised where the fleet controller would start: every island is provisioned."""


def _render(value, depth=0):
    """A JSON form of a fake sky task / resources / any plain object (sorted, no ids)."""
    if depth > 8:
        return "<deep>"
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _render(v, depth + 1) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_render(v, depth + 1) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(str(v) for v in value)
    secret = getattr(value, "get_secret_value", None)
    if callable(secret):
        return {"secret": secret()}
    if hasattr(value, "__dict__"):
        return {"type": type(value).__name__,
                **{k: _render(v, depth + 1) for k, v in sorted(vars(value).items()) if not k.startswith("_")}}
    return repr(value)


def _setup(monkeypatch, tmp_path, record):
    from yeto import runs

    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(launcher, "prepare_launch_args", launcher._prepare_rl_args)
    monkeypatch.setattr(launcher, "prepare_run_code_pack", lambda args: None)
    monkeypatch.setattr(launcher, "warn_if_model_wont_fit", lambda *a, **k: None)
    monkeypatch.setattr(launcher, "make_syncer_task", lambda *a, **k: types.SimpleNamespace(name="syncer"))
    monkeypatch.setattr(launcher, "_tail", lambda *a, **k: None)
    monkeypatch.setattr(launcher, "_tail_modal", lambda *a, **k: None)
    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr("yeto.launch.teardown.terminate_and_verify", lambda sky, name, **k: True)
    monkeypatch.setattr(modal_runner, "resolve_syncer_for_modal", lambda addr, public: addr)
    monkeypatch.setattr(modal_runner, "modal_available", lambda: True)
    monkeypatch.setattr("yeto.launch.modal_islands._list_modal_apps", lambda: [])
    monkeypatch.setattr("yeto.launch.modal_islands.MODAL_STOP_VERIFY_DELAY_S", 0.0)
    monkeypatch.setattr("yeto.launch.syncer.tcp_probe", lambda host, port, **kw: (True, f"{host}:{port} ok"))

    class FakeModalOps:
        def __init__(self, app_name):
            self.app_name = app_name
            record.append({"modal_app": app_name})

        def define(self, cfg):
            record.append({"modal_define": json.loads(cfg.to_json())})

        def deploy(self):
            record.append({"modal_deploy": self.app_name})

        def spawn(self, cfg):
            record.append({"modal_spawn": cfg.function_name})
            return f"fc-{cfg.learner_id}"

        def status(self, call_id):
            return "RUNNING"

        def cancel(self, call_id):
            pass

        def app_status(self):
            return None

        def stop_app(self):
            pass

        def tail_logs(self, call_id, entries=100):
            return []

    class Stop:  # the fleet controller: every island is provisioned by now
        def __init__(self, *a, **k):
            raise _Provisioned

    monkeypatch.setattr(modal_runner, "ModalOps", FakeModalOps)
    monkeypatch.setattr(launcher, "FleetController", Stop)
    sky = sys.modules["sky"]

    def launch(task, **kw):
        record.append({"sky_launch": kw.get("cluster_name"),
                       "options": {k: _render(v) for k, v in sorted(kw.items()) if k != "cluster_name"},
                       "task": _render(task)})
        return f"rid-{kw.get('cluster_name')}"

    sky.launch = launch
    sky.stream_and_get = lambda rid: (1, types.SimpleNamespace(head_ip="10.0.0.1"))


def _ssh(monkeypatch, tmp_path, names):
    from test_rl_ssh_resident import FakeCluster
    from yeto.hw import card_tools

    class Cluster(FakeCluster):
        def run(self, node, script, *, timeout=300.0):  # preflight probes the fake does not know: pass
            try:
                return super().run(node, script, timeout=timeout)
            except (AttributeError, AssertionError):
                import subprocess

                return subprocess.CompletedProcess([], 0, stdout="ok\n", stderr="")

    cluster = Cluster(names)
    monkeypatch.setattr("yeto.launch.ssh_resident.SshOps", lambda: cluster)
    monkeypatch.setattr(card_tools, "check_cards", lambda family, text, **kw: [])
    # the node preflight (arch, cards, scratch, pull, weights) is no launch rendering
    monkeypatch.setattr("yeto.launch.ssh_trainer_island.SshTrainerIslands.preflight", lambda self: None)
    monkeypatch.setenv("YETO_RESOURCE_LEDGER", str(tmp_path / "ledger.jsonl"))
    return cluster


def _inventory(tmp_path, n=2):
    path = tmp_path / "inventory.yaml"
    path.write_text("nodes:\n" + "".join(
        f"  - {{name: h100-{i}, host: ops@192.0.2.{10 + i}, address: 198.51.100.{10 + i}, interface: bond0, "
        "device_family: cuda, cpu_arch: x86_64, card: H100, cards_per_node: 8, hbm_gib: 80, "
        "host_memory_gib: 1024, fabric: ib, runtime: docker, docker_args: [--gpus, all, --device, "
        "/dev/infiniband, --ulimit, 'memlock=-1:-1']}\n" for i in range(n)))
    return path


def _generic(gpu, *extra):
    def build(tmp_path, monkeypatch):
        args = _launcher_args("ports", ("--controller", "local", "--rl-image", IMAGE, *extra), gpu=gpu)
        args.keep, args.recover_timeout, args.controller_poll = False, 0.0, 30.0
        return args
    return build


def _ssh_case(tmp_path, monkeypatch):
    inv = _inventory(tmp_path)
    args = _launcher_args("ports", ("--controller", "local", "--rl-single-island-no-sync", "--rl-image", IMAGE,
                                    "--rl-node-inventory", str(inv), "--rl-ssh-trainer-workdir", "/opt/yeto",
                                    "--tensor-parallel", "8", "--pipeline-parallel", "2"),
                          gpu="ssh:2x8xh100@bm")
    args.keep, args.recover_timeout, args.controller_poll = False, 0.0, 30.0
    return args


def _k31(run_name):
    def build(tmp_path, monkeypatch):
        import yeto.cli as cli
        import yeto.provenance as provenance
        from test_k31_customer_plan_golden import K31, _argv
        from yeto.cloud import s3_store

        for key, value in (("YETO_RL_STORE_S3_ACCESS_KEY_ID", "test"), ("YETO_RL_STORE_S3_SECRET_ACCESS_KEY", "test"),
                           ("HF_HUB_OFFLINE", "1"), ("HF_HUB_CACHE", str(tmp_path / "hf")),
                           ("HF_HOME", str(tmp_path / "hf-home"))):
            monkeypatch.setenv(key, value)
        for key in (s3_store.ENDPOINT_ENV, s3_store.REGION_ENV, s3_store.PROFILE_ENV, "AWS_ENDPOINT_URL",
                    "AWS_PROFILE"):
            monkeypatch.setenv(key, "")
            monkeypatch.delenv(key)
        monkeypatch.setattr(provenance, "verify_source_tree_sha256", lambda expected=None: "5" * 64)
        args = cli.parse_args([*_argv(K31 / run_name, tmp_path), "--preflight-model", "off"])
        args.source_sha256, args.reward_sha256 = "5" * 64, "6" * 64
        args.keep, args.recover_timeout, args.controller_poll = False, 0.0, 30.0
        return args
    return build


CASES = {
    "sky-fleet": _generic("aws:1xl40s@us-east-1,aws:2x1xl40s@us-east-1"),
    "modal-fleet": _generic("modal:1xh100,modal:2x2xh100"),
    "mixed-fleet": _generic("modal:1xa100,aws:1xa100@us-east-1"),
    "ssh-resident": _ssh_case,
    "k31-modal-trainer": _k31("run.agentic-262k.json"),
    "k31-baremetal-trainer": _k31("run.baremetal-b300-262k.json"),
}
SSH_NODES = {"ssh-resident": ["h100-0", "h100-1"],
             "k31-baremetal-trainer": [f"b300-{i}" for i in range(6)]}


def _normalize(text: str, tmp_path: Path) -> str:
    for real, label in ((str(tmp_path / "home"), "<HOME>"), (str(tmp_path), "<TMP>"), (str(ROOT), "<REPO>"),
                        (str(Path.home()), "<USER-HOME>")):
        text = text.replace(real, label)
    text = re.sub(r"\d{8}T\d{6}Z", "<STAMP>", text)
    text = re.sub(r"\b1[7-9]\d{8}(\.\d+)?\b", "<EPOCH>", text)
    # a fresh random key per launch (island_hmac_secret)
    text = re.sub(r"(YETO_ISLAND_HMAC_KEY[^0-9a-f]{1,8})[0-9a-f]{64}", r"\1<HMAC>", text)
    return text


def _digest_long(value):
    """Strings longer than LONG bytes kept as their head and sha256 (the comparison stays
    byte for byte; the golden stays small)."""
    import hashlib

    if isinstance(value, str) and len(value) > LONG:
        return f"{value[:200]}...<{len(value)} bytes, sha256 {hashlib.sha256(value.encode()).hexdigest()}>"
    if isinstance(value, dict):
        return {k: _digest_long(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_digest_long(v) for v in value]
    return value


LONG = 400


def _record(case, tmp_path, monkeypatch):
    from test_k31_customer_plan_golden import PLACEHOLDER_TOKEN, _hermetic_env

    record: list = []
    _hermetic_env(tmp_path, monkeypatch)  # a placeholder HF token, no forwarded operator variables
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
    # _hermetic_env clears YETO_* too: no model lookup on the Hub, whatever the network
    monkeypatch.setenv("YETO_PREFLIGHT_MODEL", "off")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("YETO_SKY_SERVER_CHECK", "off")  # no local sky API server is asked
    # and no pre-teardown diagnostics: the launcher's teardown after _Provisioned ran the
    # real `sky logs <cluster>` per island, and that sky CLI started a sky API server
    monkeypatch.setenv("YETO_TEARDOWN_DIAG", "0")
    monkeypatch.setattr("yeto.launch.teardown.collect_teardown_diagnostics", lambda cluster, **kw: {})
    _setup(monkeypatch, tmp_path, record)
    cluster = _ssh(monkeypatch, tmp_path, SSH_NODES[case]) if case in SSH_NODES else None
    args = CASES[case](tmp_path, monkeypatch)
    try:
        launcher.run(args)
    except _Provisioned:
        pass
    if cluster is not None:
        for name in SSH_NODES[case]:
            for k, run in sorted(cluster.runs[name].items()):
                record.append({"ssh_start": name, "incarnation": k, "script": run["script"]})
    text = json.dumps(record, sort_keys=True)
    assert "hf_" not in text.replace(PLACEHOLDER_TOKEN, ""), "a Hugging Face token reached the record"
    return _digest_long(json.loads(_normalize(text, tmp_path)))


@pytest.mark.parametrize("case", sorted(CASES))
def test_trainer_first_launch_renders_unchanged(case, tmp_path, monkeypatch):
    update = os.environ.get("YETO_UPDATE_TRAINER_LAUNCH_GOLDEN") == "1"
    got = _record(case, tmp_path, monkeypatch)
    assert got, "the launcher asked no backend for anything"
    if update:
        golden = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}
        golden[case] = got
        GOLDEN.write_text(json.dumps(golden, indent=1, sort_keys=True) + "\n")
        return
    assert got == json.loads(GOLDEN.read_text())[case]

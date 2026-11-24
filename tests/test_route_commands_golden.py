"""Every command the National Compute docker start, the Daytona process start and the ssh
pool start (container and venv) render for one sampler start, byte for byte against
tests/golden/route_commands.json (rendered by the code before the route helpers were
shared). The customer's NC route must not change by a byte; set
YETO_REGEN_ROUTE_GOLDEN=1 to rewrite the file after a deliberate change."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

GOLDEN = Path(__file__).resolve().parent / "golden" / "route_commands.json"
SHA = "abcdef123456"


class _Done:
    def __init__(self, stdout="", rc=0):
        self.returncode, self.stdout, self.stderr = rc, stdout, ""


class _FakeArchive:
    stdout = None

    def wait(self):
        return 0


def _recorder(dump, name, prompts: Path):
    calls = dump.setdefault(name, [])

    def run(argv, **kw):
        if isinstance(argv, list) and argv[:2] == ["git", "-C"]:
            if "rev-parse" in argv:
                return _Done(SHA + "\n")
            return SimpleNamespace(returncode=0, stdout=b"TAR", stderr=b"")
        rec = {"argv": argv if isinstance(argv, list) else str(argv)}
        if kw.get("input") is not None:
            data = kw["input"]
            rec["input"] = data if isinstance(data, str) else f"<{len(data)} bytes>"
        calls.append(rec)
        text = argv[-1] if isinstance(argv, list) else str(argv)
        if "YETO_PROMPTS" in text:
            data = prompts.read_bytes()
            return _Done(f"YETO_PROMPTS {len(data)} {hashlib.sha256(data).hexdigest()}\n")
        if "YETO_SSH_STARTED" in text or "run -d" in text:
            return _Done("YETO_SSH_STARTED container=x\n")
        if "YETO_SSH_CARDS_BEGIN" in text:
            from tests.test_ssh_sampler_pool import IDLE  # noqa: PLC0415

            return _Done("YETO_SSH_ARCH aarch64\nYETO_SSH_FREE_GIB 500\nYETO_SSH_CARDS_BEGIN\n" + IDLE
                         + "\nYETO_SSH_CARDS_END rc=0\n")
        return _Done("")
    return run


def render(tmp: Path, monkeypatch) -> dict:
    from tests.test_daytona_sampler import FakeBox  # noqa: PLC0415
    from yeto.launch import daytona_sampler as ds  # noqa: PLC0415
    from yeto.launch import ssh_sampler_pool as sp  # noqa: PLC0415
    from yeto.launch.nc import cli, ssh_sampler  # noqa: PLC0415
    from yeto.launch.nc.client import NodeLease  # noqa: PLC0415
    from yeto.launch.node_inventory import parse_node  # noqa: PLC0415

    dump: dict = {}
    prompts = tmp / "prompts.jsonl"
    prompts.write_text('{"q": "x"}\n')
    (tmp / "r2.env").write_text("R2_BUCKET=b\nAWS_ENDPOINT_URL=https://e\n")
    class Popen(subprocess.Popen):  # git archive is faked; everything else is real (and still a class)
        def __new__(cls, argv, *a, **k):
            if isinstance(argv, list) and argv[:1] == ["git"] and "archive" in argv:
                return _FakeArchive()
            return super().__new__(cls)

    monkeypatch.setattr(subprocess, "Popen", Popen)

    # National Compute docker route (the customer's)
    lease = NodeLease(1, "tk1", "192.0.2.9", "ubuntu", "", None, None, 8, None)
    spec = {"sampler_id": "nc-0", "prompts": {"path": "/work/prompts/p.jsonl"}, "engine": {"adapter": "vllm-server"}}
    ssh_sampler.start_sampler(lease, spec=spec, shard="nc-0", run_prefix="runs1", r2_env_file=str(tmp / "r2.env"),
                              state_dir=tmp, run=_recorder(dump, "nc_start", prompts), prompts_file=str(prompts))
    dump["nc_spec_file"] = (tmp / "spec-nc-0.json").read_text()
    dump["nc_pull"] = ssh_sampler.pull_model_command("models/k31", "/mnt/nvme/models/Kimi-K3.1-Pro")
    dump["nc_sampler_command"] = ssh_sampler.sampler_command("/work/sampler/nc-0.json")
    dump["nc_env"] = ssh_sampler.sampler_env("runs1")

    # the customer's agentic NC start: agentic.setup (the node preparation itself is
    # agentic_setup's, not rendered here), the operator's env file second, the K3.1 ROCm image the
    # customer's pool runs (examples/k31 names none: the route's default)
    from tests.test_agentic_setup import _spec as agentic_spec  # noqa: PLC0415
    from yeto.launch import agentic_setup  # noqa: PLC0415

    monkeypatch.setattr(agentic_setup.Provisioner, "run", lambda self: {})
    agentic = dict(agentic_spec(tmp), sampler_id="nc-1", prompts={"path": "/work/prompts/p.jsonl"},
                   engine={"adapter": "vllm-server"})
    (tmp / "op.env").write_text("HARBOR_ENV=remote\n")
    ssh_sampler.start_sampler(lease, spec=agentic, shard="nc-1", run_prefix="runs1", r2_env_file=str(tmp / "r2.env"),
                              state_dir=tmp, agentic_env_file=str(tmp / "op.env"),
                              run=_recorder(dump, "nc_agentic_start", prompts), prompts_file=str(prompts))
    dump["nc_agentic_spec_file"] = (tmp / "spec-nc-1.json").read_text()
    dump["nc_agentic_setup_env"] = (tmp / "nc-1.agentic-setup.env").read_text()

    # Daytona process route
    class Box(FakeBox):
        def exec(self, command, *, timeout=600):
            if "YETO_PROMPTS" in command:
                self.commands.append(command)
                data = prompts.read_bytes()
                return ds.ExecResult(0, f"YETO_PROMPTS {len(data)} {hashlib.sha256(data).hexdigest()}\n")
            return super().exec(command, timeout=timeout)

    box = Box()
    dspec = {"sampler_id": "d-0", "engine": {"adapter": "vllm-server", "server": {"in_place_reload": True}},
             "event_tape": "/work/sampler/events-d-0.jsonl", "prompts": {"path": "/work/prompts/p.jsonl"}}
    ds.start_sampler(box, spec=dspec, shard="d-0", run_prefix="runs1", r2_env_file=str(tmp / "r2.env"),
                     code=(SHA, b"TAR"), prompts_file=str(prompts))
    dump["daytona_commands"] = box.commands
    dump["daytona_uploads"] = {k: (v.decode() if not k.endswith(".tar") and k != ds.STORE_ENV + ".up" else len(v))
                               for k, v in sorted(box.uploads.items())}
    dump["daytona_pull"] = ds.pull_command("models/k31")

    # ssh inventory pool route, on its own sampler start
    monkeypatch.setattr(sp.ssh_transport, "mux_options", lambda route: [])
    monkeypatch.setattr(cli.tasks, "sampler_spec", lambda spec, sampler_id, shards: dict(spec, sampler_id=sampler_id))
    monkeypatch.setattr(cli.tasks, "apply_sampler_overrides", lambda spec, **kw: spec)
    node = parse_node({"name": "a0", "host": "u@a0", "device_family": "npu", "cpu_arch": "aarch64", "card": "910B4",
                       "cards_per_node": 8, "hbm_gib": 64, "host_memory_gib": 1024, "runtime": "docker"}, index=0)
    (tmp / "spec.json").write_text(json.dumps({"sampler_id": "x", "prompts": {"groups_per_round": 2},
                                               "buffer_shards": ["s-0"]}))
    for runtime in ("container", "venv"):
        provider = {"kind": "ssh", "workdir": "/srv/yeto/run0", "runtime": runtime, "devices": [0],
                    "preflight": {"require_idle": False}}
        provider.update({"container": {"engine": "podman", "image": "img"}} if runtime == "container"
                        else {"venv": {"python": "/opt/v/bin/python"}})
        if "node_runtime" in getattr(sp, "CONFIG_KEYS", ()):
            provider["node_runtime"] = False
        cfg = {"run": "r0", "shard_base": "s", "accounts": [0], "target": 1, "spec": str(tmp / "spec.json"),
               "run_prefix": "runs1", "r2_env_file": str(tmp / "r2.env"), "provider": provider}
        ops = sp.SshSamplerPoolOps(cfg, conf=sp.parse_config(cfg, inventory=[node]),
                                   run=_recorder(dump, f"ssh_pool_{runtime}", prompts), state_dir=tmp)
        ticket = ops.order(0)
        m = SimpleNamespace(account=0, ticket=ticket, shard="s-0", data_slot=None, replaces_shard=None,
                            predecessor_lost=False)
        ops.start(m, ["s-0"])
        for call in dump[f"ssh_pool_{runtime}"]:  # the ticket has a time stamp in it
            call["argv"] = [a.replace(ticket, "<T>") if isinstance(a, str) else a for a in call["argv"]]
            if "input" in call:
                call["input"] = call["input"].replace(ticket, "<T>")
    return json.loads(json.dumps(dump, sort_keys=True, default=str).replace(str(tmp), "<TMP>"))


def test_route_commands_are_byte_identical_to_the_golden(tmp_path, monkeypatch):
    monkeypatch.setenv("YETO_NC_STATE_DIR", str(tmp_path / "state"))
    rendered = render(tmp_path, monkeypatch)
    if os.environ.get("YETO_REGEN_ROUTE_GOLDEN") == "1":
        GOLDEN.write_text(json.dumps(rendered, indent=1, sort_keys=True) + "\n")
    golden = json.loads(GOLDEN.read_text())
    assert sorted(rendered) == sorted(golden)
    for key in sorted(golden):
        assert rendered[key] == golden[key], key

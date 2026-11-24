"""Shadow comparison (yeto.node.shadow): the National Compute and Daytona routes keep their
own sampler starts; the node runtime's plan on the same node is compared with them, field
by field, and a route may switch only on an empty difference list. The differences of
today are pinned here: a change on either side shows up as a failing test."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from yeto.launch import daytona_sampler as ds
from yeto.launch.nc import ssh_sampler
from yeto.launch.nc.client import NodeLease
from yeto.node import shadow

SHA = "abcdef123456"
REPO = Path(__file__).resolve().parents[1]


class _NoArchive:
    stdout = None

    def __init__(self, *a, **k):
        pass

    def wait(self):
        return 0


def customer_spec(shard: str) -> dict:
    """The customer K3.1 NC sampler spec's launch-relevant part (examples/k31)."""
    return {"sampler_id": shard, "store": "/s", "event_tape": f"/work/sampler/events-{shard}.jsonl",
            "engine": {"adapter": "vllm-server", "options": {"k31_preset": "mi355x"},
                       "server": {"in_place_reload": True, "model": "/models/kimi-k3.1-pro", "port": 8000}},
            "prompts": {"path": "/work/prompts/p.jsonl", "groups_per_round": 4}}


def old_nc(tmp_path, monkeypatch, spec, shard):
    calls = []

    def run(argv, **kw):
        if argv[:2] == ["git", "-C"]:
            return SimpleNamespace(returncode=0, stdout=SHA + "\n", stderr="")
        calls.append(argv)
        if "YETO_PROMPTS" in argv[-1]:
            data = (tmp_path / "p.jsonl").read_bytes()
            return SimpleNamespace(returncode=0, stdout=f"YETO_PROMPTS {len(data)} {hashlib.sha256(data).hexdigest()}",
                                   stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(ssh_sampler.subprocess, "Popen", _NoArchive)
    (tmp_path / "r2.env").write_text("R2_BUCKET=b\n")
    (tmp_path / "p.jsonl").write_text('{"q": "x"}\n')
    lease = NodeLease(1, "tk1", "192.0.2.9", "ubuntu", "", None, None, 8, None)
    ssh_sampler.start_sampler(lease, spec=spec, shard=shard, run_prefix="runs1", r2_env_file=str(tmp_path / "r2.env"),
                              state_dir=tmp_path, run=run, prompts_file=str(tmp_path / "p.jsonl"))
    return lease, next(c[-1] for c in calls if "docker run" in c[-1])


def test_the_nc_docker_command_parses_into_its_launch(tmp_path, monkeypatch):
    _, command = old_nc(tmp_path, monkeypatch, customer_spec("nc-0"), "nc-0")
    old = shadow.docker_launch(command)
    assert old.devices[:2] == ("--network", "host") and "/dev/kfd" in old.devices
    assert old.mounts["/work"] == "/mnt/nvme/work" and old.mounts["/models/kimi-k3.1-pro"].endswith(":ro")
    assert old.image == ssh_sampler.IMAGE and old.module == "yeto.rl.engine.engine_sampler"
    assert old.spec == "/work/sampler/nc-0.json" and old.python == ssh_sampler.PYTHON
    assert old.env["YETO_RL_RELEASE_SPREAD_KEY"] == "runs1/nc-0" and old.env_files == ("/home/ubuntu/.yeto-nc-r2.env",)
    assert shadow.compare(old, old) == [] and shadow.switchable([])


def test_nc_shadow_of_the_customer_spec_pins_what_is_left(tmp_path, monkeypatch):
    spec = customer_spec("nc-0")
    lease, command = old_nc(tmp_path, monkeypatch, spec, "nc-0")
    old = shadow.docker_launch(command)
    profile = shadow.nc_profile(lease)
    plan = shadow.node_plan(profile, spec={"start_on_latest": True, **spec}, shard="nc-0", run_prefix="runs1",
                            code_sha=SHA, store_uri="s3://b/runs/runs1", env_file="/home/ubuntu/.yeto-nc-r2.env",
                            image=ssh_sampler.IMAGE, python=ssh_sampler.PYTHON, source="nc",
                            env={"YETO_NC_RUN_PREFIX": "runs1"})
    new = shadow.plan_launch(plan)
    # what already matches: device flags, image, interpreter, sampler, spec, env file, store, marker, spread key
    for name in ("devices", "image", "python", "module", "spec", "env_files"):
        assert getattr(new, name) == getattr(old, name), name
    for key in ("HIP_VISIBLE_DEVICES", "YETO_SPOT_RECLAIM_MARKER", "YETO_RL_RELEASE_SPREAD_KEY", "YETO_RL_STORE_IMPL",
                "YETO_RL_STORE_S3_STAGE", "YETO_RL_STORE_S3_CREDENTIALS", "YETO_RL_STORE_S3_REGION",
                "PYTHONUNBUFFERED"):
        assert new.env[key] == old.env[key], key
    differences = shadow.compare(old, new)
    assert not shadow.switchable(differences)  # NC stays on its route
    # what is left: the code tree (the route mounts it at /yeto and sets -e PYTHONPATH=/yeto, which
    # drops an image's own entries; the runtime prepends its tree inside the container) and the
    # store URI (the route builds it on the node from the env file's bucket)
    assert differences == [
        "mount /yeto: /mnt/nvme/yeto/abcdef123456:ro -> (none)",
        "env PYTHONPATH: /yeto -> (unset)",
        "env YETO_RL_STORE_S3_URI: (unset) -> s3://b/runs/runs1",
    ]


class _Box(__import__("tests.test_daytona_sampler", fromlist=["FakeBox"]).FakeBox):
    def __init__(self, prompts: bytes):
        super().__init__()
        self.prompts = prompts

    def exec(self, command, *, timeout=600):
        if "YETO_PROMPTS" in command:
            self.commands.append(command)
            return ds.ExecResult(0, f"YETO_PROMPTS {len(self.prompts)} {hashlib.sha256(self.prompts).hexdigest()}\n")
        return super().exec(command, timeout=timeout)


def test_daytona_shadow_pins_what_is_left(tmp_path):
    (tmp_path / "r2.env").write_text("R2_BUCKET=b\nAWS_ENDPOINT_URL=https://e\n")
    (tmp_path / "p.jsonl").write_text('{"q": "x"}\n')
    spec = customer_spec("d-0")
    box = _Box((tmp_path / "p.jsonl").read_bytes())
    ds.start_sampler(box, spec=spec, shard="d-0", run_prefix="runs1", r2_env_file=str(tmp_path / "r2.env"),
                     code=(SHA, b"TAR"), prompts_file=str(tmp_path / "p.jsonl"))
    old = shadow.script_launch(box.uploads["/work/sampler/d-0.start.sh.up"].decode())
    assert old.module == "yeto.rl.engine.engine_sampler" and old.env_files == (ds.STORE_ENV,)
    plan = shadow.node_plan(shadow.daytona_profile("box"), spec={"start_on_latest": True, **spec}, shard="d-0",
                            run_prefix="runs1", code_sha=SHA, store_uri="s3://b/runs/runs1", python=ds.PYTHON,
                            source="daytona", env={"YETO_NC_RUN_PREFIX": "runs1"})
    new = shadow.plan_launch(plan)
    assert (new.module, new.spec, new.python) == (old.module, old.spec, old.python)
    for key in ("HIP_VISIBLE_DEVICES", "PYTHONPATH", "YETO_SPOT_RECLAIM_MARKER", "YETO_RL_RELEASE_SPREAD_KEY",
                "YETO_RL_STORE_S3_STAGE"):
        assert new.env[key] == old.env[key], key
    differences = shadow.compare(old, new)
    assert not shadow.switchable(differences)  # Daytona stays on its route
    assert differences == [  # the store env file is read by the runtime, the URI built from it there
        "env_files: ('/work/sampler/store.env',) -> ()",
        "env YETO_RL_STORE_S3_ENDPOINT_URL: $AWS_ENDPOINT_URL -> (unset)",
        "env YETO_RL_STORE_S3_URI: s3://${R2_BUCKET}/runs/${YETO_NC_RUN_PREFIX} -> s3://b/runs/runs1",
    ]


def test_plan_launch_of_a_process_plan():
    record = {"argv": ["python3", "-m", "yeto.rl.engine.engine_sampler", "--spec", "/work/sampler/s.json"],
              "env": {"A": "1"}}
    launch = shadow.plan_launch(record)
    assert (launch.python, launch.module, launch.spec, dict(launch.env)) == (
        "python3", "yeto.rl.engine.engine_sampler", "/work/sampler/s.json", {"A": "1"})
    assert json.dumps(launch.record())


@pytest.mark.parametrize("line", ["sudo docker rm -f x >/dev/null 2>&1; sudo docker run -d --name x --gpus all "
                                  "-v /a:/work -e K=V --entrypoint bash img -c 'exec python3 -m m --spec /work/s.json'"])
def test_docker_launch_of_a_shell_line(line):
    launch = shadow.docker_launch(line)
    assert launch.devices == ("--gpus", "all") and launch.mounts == {"/work": "/a"} and launch.env == {"K": "V"}
    assert (launch.image, launch.python, launch.module, launch.spec) == ("img", "python3", "m", "/work/s.json")

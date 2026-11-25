"""The syncer of a fleet launched island by island runs on its own VM: its tape reaches
the run store through the uploader sidecar on that VM (``tapes/syncer/<incarnation>/``,
what the dashboard reads with ``--store``), with the store credential as a task secret.
Without an object store, or for any launch that is not the fleet's syncer, the syncer
task is unchanged (fake sky; nothing is ordered)."""

from __future__ import annotations

import json
import platform
import shlex

import pytest

from test_rl_island_pool_launcher import POOL, _args, _fake_sky
from yeto import launcher
from yeto.launch.syncer import syncer_shard_count
from yeto.rl.engine import tape_upload

STORE = "s3://fleet-bucket/runs/r1"
KEY_ID, SECRET = "AKIDTEST0123", "s3cr3t-value-0123"


@pytest.fixture
def syncer_vm(monkeypatch, tmp_path):
    _fake_sky(monkeypatch)
    monkeypatch.setattr("yeto.launch.syncer.build_syncer_binary", lambda: tmp_path / "yeto-syncer")
    monkeypatch.setenv("YETO_RL_STORE_S3_ACCESS_KEY_ID", KEY_ID)
    monkeypatch.setenv("YETO_RL_STORE_S3_SECRET_ACCESS_KEY", SECRET)
    monkeypatch.delenv("YETO_RL_STORE_S3_PROFILE", raising=False)
    monkeypatch.delenv(tape_upload.INTERVAL_ENV, raising=False)


def _syncer(store: str | None = STORE, role: str | None = "syncer"):
    args = _args(POOL)
    args.rl_checkpoint_store = store
    args.rl_fleet_role = role
    return args


def _follow(task) -> list[str]:
    lines = [line for line in task.run.splitlines() if "yeto.rl.engine.tape_upload follow" in line]
    assert len(lines) == 1, task.run
    inner = lines[0].split("nohup ", 1)[1].split(" >", 1)[0]
    return shlex.split(inner)


@pytest.mark.parametrize("system,machine", [("Linux", "x86_64"), ("Darwin", "arm64")])  # fast path, remote build
def test_the_fleet_syncer_vm_uploads_its_tape_to_the_run_store(syncer_vm, monkeypatch, system, machine):
    monkeypatch.setattr(platform, "system", lambda: system)
    monkeypatch.setattr(platform, "machine", lambda: machine)
    task = launcher.make_syncer_task(_syncer(), 2)
    argv = _follow(task)
    # the tape the syncer writes, every shard's, into the run store's tapes/syncer[-shard<i>]/
    assert argv[argv.index("--tape") + 1] == launcher.syncer_event_tape(_syncer())
    assert argv[argv.index("--store") + 1] == STORE
    assert argv[argv.index("--shards") + 1] == str(syncer_shard_count(_syncer()))
    assert "--from-end" in argv and float(argv[argv.index("--interval") + 1]) == tape_upload.DEFAULT_INTERVAL_S
    # beside the syncer, never in front of it: backgrounded, and the syncer stays the job's command
    run = task.run
    assert run.index("tape_upload follow") < run.index("yeto-syncer") and "&) || true" in run
    assert "pip install -q boto3" in task.setup and "|| echo" in task.setup
    assert task.workdir  # the yeto package rides along for the uploader
    # the VM reads the store with the launcher's store credential: the secret as a task secret only
    assert task.secrets["YETO_RL_STORE_S3_SECRET_ACCESS_KEY"] == SECRET
    assert task.envs["YETO_RL_STORE_S3_ACCESS_KEY_ID"] == KEY_ID and task.envs["YETO_RL_STORE_S3_URI"] == STORE
    assert SECRET not in task.run + (task.setup or "") + json.dumps(task.envs)


def test_the_upload_lands_where_the_dashboard_reads_the_syncer(tmp_path):
    from yeto.rl.engine.object_store import DirectoryObjectStore

    tape = tmp_path / "yeto-tape.jsonl"
    tape.write_text("")
    objects = DirectoryObjectStore(tmp_path / "store")
    ups = tape_upload.syncer_uploaders(objects, tape, incarnation="vm-1", interval_s=3600)
    with tape.open("a") as fh:
        fh.write(json.dumps({"event": "merge", "outer_step": 1}) + "\n")
    assert all(up.stop() for up in ups)
    keys = sorted(objects.list("tapes/"))
    assert keys and all(k.startswith("tapes/syncer/vm-1/") and k.endswith(".jsonl") for k in keys)


@pytest.mark.parametrize("store,role", [(None, "syncer"), ("/shared/runs/r1", "syncer"),
                                        ("modal://vol/r1", "syncer"), (STORE, None), (STORE, "island")])
def test_no_uploader_without_an_object_store_or_off_the_fleet_syncer(syncer_vm, monkeypatch, store, role):
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    task = launcher.make_syncer_task(_syncer(store, role), 2)
    assert "tape_upload" not in task.run and "boto3" not in (task.setup or "")
    assert task.workdir is None and "YETO_RL_STORE_S3_URI" not in (task.envs or {})
    assert set(task.secrets) == {"YETO_ISLAND_HMAC_KEY"}  # the task as before


def test_an_upload_turned_off_or_without_a_store_credential_leaves_the_task_alone(syncer_vm, monkeypatch, capsys):
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setenv(tape_upload.INTERVAL_ENV, "0")
    assert "tape_upload" not in launcher.make_syncer_task(_syncer(), 2).run
    monkeypatch.delenv(tape_upload.INTERVAL_ENV)
    monkeypatch.delenv("YETO_RL_STORE_S3_SECRET_ACCESS_KEY")
    task = launcher.make_syncer_task(_syncer(), 2)
    assert "tape_upload" not in task.run and "WARN: the syncer tape is not uploaded" in capsys.readouterr().out

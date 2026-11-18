"""Node-local storage, tape and resource ledger of an SSH trainer island
(``yeto.launch.ssh_node_storage``, ``yeto.launch.ssh_trainer_ledger``)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_rl_ssh_trainer_island import FakeOps, b300, b300_spec, node, spec
from yeto.cloud import oss_store, s3_store
from yeto.cost import billing
from yeto.cost import cleanup as cost_cleanup
from yeto.cost.ledger import LEDGER_ENV, ResourceLedger
from yeto.hw import card_tools
from yeto.launch import ssh_node_storage as storage
from yeto.launch import ssh_trainer_ledger
from yeto.launch.node_inventory import InventoryError, parse_node
from yeto.launch.ssh_trainer_island import (
    CONTAINER_STATE_DIR,
    SshIslandError,
    SshTrainerIslands,
    remote_paths,
    start_script,
)

RAW = {"name": "b300-0", "host": "ops@10.0.0.10", "device_family": "cuda", "cpu_arch": "x86_64", "card": "B300",
       "cards_per_node": 8, "hbm_gib": 288, "host_memory_gib": 2048, "runtime": "rootfs"}
STAGED = {s3_store.STAGE_ENV: s3_store.DEFAULT_STAGE, s3_store.URI_ENV: "s3://bkt/run"}


@pytest.fixture
def checked_cards(monkeypatch):
    monkeypatch.setattr(card_tools, "check_cards", lambda family, text, **kw: [])


# ---------------------------------------------------------------- inventory
def test_inventory_scratch_is_an_absolute_plain_path_and_min_free_needs_it():
    parsed = parse_node({**RAW, "scratch": "/mnt/nvme/yeto/", "scratch_min_free_gib": 1500}, index=0)
    assert (parsed.scratch, parsed.scratch_min_free_gib) == ("/mnt/nvme/yeto", 1500.0)
    assert parse_node(RAW, index=0).scratch is None
    for bad in ("relative/dir", "/", "~/x", "/mnt/../etc", "/mnt/a b", "/mnt/$HOME", "/mnt/a:b", "/mnt/a,b", ""):
        with pytest.raises(InventoryError, match="scratch"):
            parse_node({**RAW, "scratch": bad}, index=0)
    with pytest.raises(InventoryError, match="scratch_min_free_gib needs scratch"):
        parse_node({**RAW, "scratch_min_free_gib": 10}, index=0)
    with pytest.raises(InventoryError, match="scratch_min_free_gib"):
        parse_node({**RAW, "scratch": "/x", "scratch_min_free_gib": -1}, index=0)


# ---------------------------------------------------------------- environment and mounts
def test_scratch_env_moves_the_stage_staging_and_tmp_and_leaves_the_inventory_env_alone():
    docker = b300(scratch="/mnt/nvme/yeto")
    env = storage.scratch_env(docker, "k31", STAGED)
    assert env == {s3_store.STAGE_ENV: "/yeto-scratch/store-stage",
                   storage.RELEASE_STAGING_ENV: "/yeto-scratch/release-staging", "TMPDIR": "/yeto-scratch/tmp"}
    # an unstaged store stays unstaged; an oss stage moves too
    assert s3_store.STAGE_ENV not in storage.scratch_env(docker, "k31", {s3_store.URI_ENV: "s3://b/r"})
    assert storage.scratch_env(docker, "k31", {oss_store.STAGE_ENV: "~/x"})[oss_store.STAGE_ENV] == \
        "/yeto-scratch/store-stage"
    assert storage.scratch_env(docker, "k31", {oss_store.URI_ENV: "oss://b/r"})[oss_store.STAGE_ENV] == \
        "/yeto-scratch/store-stage"  # an oss store always stages (default stage when unset)
    rootfs = node(scratch="/data/scratch")
    assert storage.scratch_env(rootfs, "run/1", STAGED)[s3_store.STAGE_ENV] == "/data/scratch/run-1/store-stage"
    pinned = b300(scratch="/s", env={"TMPDIR": "/fast/tmp"})
    assert "TMPDIR" not in storage.scratch_env(pinned, "k31", STAGED)
    assert storage.scratch_env(b300(), "k31", STAGED) == {}
    assert storage.docker_options(docker, "k31") == ["-v", "/mnt/nvme/yeto/k31:/yeto-scratch"]
    assert storage.docker_options(rootfs, "k31") == [] and storage.docker_options(b300(), "k31") == []


def test_release_staging_name_is_the_one_the_miles_trainer_reads():
    from yeto.rl.adapters.miles import sampler_island

    assert storage.RELEASE_STAGING_ENV == sampler_island.STAGING_ENV


def test_docker_start_script_mounts_one_scratch_and_exports_its_paths():
    nodes = [b300(i, scratch="/mnt/nvme/yeto") for i in range(2)]
    paths = remote_paths("k31", "t", 2)
    script = start_script(b300_spec(nodes, env=dict(STAGED)), 1, {}, paths, container="c1")
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    assert "-v /mnt/nvme/yeto/k31:/yeto-scratch" in script
    assert "mkdir -p /mnt/nvme/yeto/k31\n" in script  # created by the ssh user before docker run
    assert f"export {s3_store.STAGE_ENV}=/yeto-scratch/store-stage" in script
    assert f"export {s3_store.STAGE_ENV}={s3_store.DEFAULT_STAGE}" not in script
    assert "export TMPDIR=/yeto-scratch/tmp" in script and f"-e {storage.RELEASE_STAGING_ENV}" in script
    assert 'ln -s /yeto-scratch/init-release "$HOME/yeto-init-release"' in script
    assert f'ln -s {CONTAINER_STATE_DIR}/yeto-output "$HOME/yeto-output"' in script
    assert "(umask 077 && mkdir -p" in script  # the state dir (logs, tape) is the ssh user's alone


def test_docker_start_script_without_scratch_still_keeps_the_tape_in_the_state_dir():
    script = start_script(b300_spec(), 0, {}, remote_paths("k31", "t", 1), container="c0")
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    assert "/yeto-scratch" not in script and "TMPDIR" not in script
    assert f'ln -s {CONTAINER_STATE_DIR}/yeto-output "$HOME/yeto-output"' in script


def test_rootfs_start_script_exports_host_paths_and_links_nothing():
    script = start_script(spec([node(scratch="/data/s"), node("ppu1", "u@10.0.0.2", scratch="/data/s")],
                               env=dict(STAGED)), 0, {}, remote_paths("run1", "t", 1))
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    assert f"export {s3_store.STAGE_ENV}=/data/s/run1/store-stage" in script
    assert "ln -s" not in script and "docker" not in script


def test_the_job_prelude_links_home_into_scratch_and_a_release_hard_links_into_the_stage(tmp_path, monkeypatch):
    """Run the docker prelude with the container paths moved under tmp: the links land
    in $HOME, an existing entry is kept, and staging -> stage is a hard link (one mount)."""
    root, state, home = tmp_path / "scratch", tmp_path / "state", tmp_path / "home"
    home.mkdir()
    (home / "miles-ray").mkdir()  # already there: left alone
    monkeypatch.setattr(storage, "CONTAINER_SCRATCH", str(root))
    prelude = storage.job_prelude(b300(scratch="/mnt/nvme/yeto"), "k31", str(state))
    done = subprocess.run(["bash", "-ec", prelude], env={**os.environ, "HOME": str(home)}, capture_output=True,
                          text=True, check=False)
    assert done.returncode == 0, done.stderr
    assert os.readlink(home / "yeto-init-release") == str(root / "init-release")
    assert os.readlink(home / "yeto-output") == str(state / "yeto-output")
    assert (home / "miles-ray").is_dir() and not (home / "miles-ray").is_symlink()
    assert (root / "tmp").is_dir() and (root / "release-staging").is_dir()
    weights = root / "release-staging" / "model.safetensors"
    weights.write_bytes(b"w" * 16)
    (root / "store-stage").mkdir()
    os.link(weights, root / "store-stage" / "model.safetensors")
    assert (root / "store-stage" / "model.safetensors").stat().st_nlink == 2
    # run twice (a second incarnation in a fresh container sees the same scratch)
    assert subprocess.run(["bash", "-ec", prelude], env={**os.environ, "HOME": str(home)},
                          check=False).returncode == 0


# ---------------------------------------------------------------- preflight
class ScratchOps(FakeOps):
    def __init__(self, free_kb=2 * 1024 ** 3, rc=0):
        super().__init__()
        self.free_kb, self.rc = free_kb, rc

    def run(self, n, script, *, timeout=300.0):
        if "df -Pk" in script:
            self.runs.append((n.name, script))
            return subprocess.CompletedProcess([], self.rc, stdout=f"{self.free_kb}\n",
                                               stderr="mkdir: Permission denied" if self.rc else "")
        if script.startswith("docker pull"):
            self.runs.append((n.name, script))
            return subprocess.CompletedProcess([], 0, stdout="", stderr="")
        return super().run(n, script, timeout=timeout)


def test_preflight_checks_every_scratch_and_refuses_too_little_free_space(tmp_path, checked_cards, capsys):
    nodes = [b300(i, scratch="/mnt/nvme/yeto", scratch_min_free_gib=1500) for i in range(2)]
    ops = ScratchOps(free_kb=2 * 1024 ** 3)  # 2 TiB
    SshTrainerIslands(b300_spec(nodes), name="t", jobs_path=tmp_path / "j.json", ops=ops).preflight()
    assert [n for n, s in ops.runs if "df -Pk" in s] == ["b300-0", "b300-1"]
    assert "scratch /mnt/nvme/yeto has 2048 GiB free" in capsys.readouterr().out
    small = SshTrainerIslands(b300_spec(nodes), name="t", jobs_path=tmp_path / "j.json",
                              ops=ScratchOps(free_kb=100 * 1024 ** 2))
    with pytest.raises(SshIslandError, match="b300-0: scratch /mnt/nvme/yeto has 100 GiB free.*1500"):
        small.preflight()
    broken = SshTrainerIslands(b300_spec(nodes), name="t", jobs_path=tmp_path / "j.json", ops=ScratchOps(rc=1))
    with pytest.raises(SshIslandError, match="not a writable directory.*Permission denied"):
        broken.launch(1, None)


def test_preflight_warns_when_docker_nodes_stage_a_store_without_scratch(tmp_path, checked_cards, capsys):
    SshTrainerIslands(b300_spec(env=dict(STAGED)), name="t", jobs_path=tmp_path / "j.json",
                      ops=ScratchOps()).preflight()
    assert "declare no inventory scratch" in capsys.readouterr().out
    SshTrainerIslands(b300_spec(), name="t", jobs_path=tmp_path / "j.json", ops=ScratchOps()).preflight()
    assert "declare no inventory scratch" not in capsys.readouterr().out  # nothing staged


def test_preflight_script_reports_free_space_on_a_real_directory(tmp_path):
    n = node(scratch=str(tmp_path / "nvme"))
    done = subprocess.run(["bash", "-c", storage.preflight_script(n, "run1")], capture_output=True, text=True,
                          check=False)
    assert storage.check_preflight(n, done.returncode, done.stdout, done.stderr) > 0
    assert (tmp_path / "nvme" / "run1").is_dir() and not list((tmp_path / "nvme" / "run1").iterdir())
    with pytest.raises(storage.ScratchError, match="no free size"):
        storage.check_preflight(n, 0, "Filesystem\n")
    assert storage.preflight_script(node(), "run1") is None


# ---------------------------------------------------------------- resource ledger
def test_incarnations_are_ledger_rows_closed_by_a_confirmed_stop(tmp_path, checked_cards, monkeypatch):
    path = tmp_path / "resources.jsonl"
    monkeypatch.setenv(LEDGER_ENV, str(path))
    nodes = [b300(i, price_per_card_hour=2.5) for i in range(2)]
    ops = ScratchOps()
    islands = SshTrainerIslands(b300_spec(nodes), name="trainer", jobs_path=tmp_path / "j.json", ops=ops,
                                kill_grace_s=7)
    job = islands.launch(1, None)
    (row,) = ResourceLedger(path).open_resources()
    assert (row.kind, row.id, row.name) == ("ssh_trainer", "k31/trainer-i1", "trainer-i1")
    assert row.attrs["nodes"] == [{"name": "b300-0", "host": "ops@10.0.0.10", "proxy_jump": None},
                                  {"name": "b300-1", "host": "ops@10.0.0.11", "proxy_jump": None}]
    assert (row.attrs["docker"], row.attrs["gpus"], row.attrs["usd_per_hour"], row.attrs["kill_grace_s"]) == \
        (True, 16, 40.0, 7)
    assert islands.release(job)
    assert ResourceLedger(path).open_resources() == []
    islands.launch(2, None)

    class FailingKill(ScratchOps):
        def run(self, n, script, *, timeout=300.0):
            if "kill -TERM" in script:
                return subprocess.CompletedProcess([], 3, stdout="", stderr="no pid")
            return super().run(n, script, timeout=timeout)

    islands.ops = FailingKill()
    assert not islands.release_incarnation(2)
    assert [r.id for r in ResourceLedger(path).open_resources()] == ["k31/trainer-i2"]  # left to a cleanup


def test_cleanup_stops_an_open_incarnation_on_every_node_by_the_ledger_alone(tmp_path):
    ledger = ResourceLedger(tmp_path / "resources.jsonl")
    attrs = ssh_trainer_ledger.created_attrs(run_id="k31", island="trainer", incarnation=3,
                                             nodes=[b300(i, proxy_jump="ops@bastion") for i in range(2)],
                                             docker=True, kill_grace_s=9)
    ledger.created("ssh_trainer", "k31/trainer-i3", "trainer-i3", **attrs)
    sent = []

    class Ssh:
        def run(self, n, script, *, timeout):
            sent.append((n.host, n.ssh_options(), script, timeout))
            return subprocess.CompletedProcess([], 0, stdout="", stderr="")

    ops = SimpleNamespace(ssh_ops=Ssh())
    (dry,) = cost_cleanup.cleanup(ledger, ops, dry_run=True)
    assert dry.ok and "would stop" in dry.action and not sent
    (action,) = cost_cleanup.cleanup(ledger, ops)
    assert action.ok and ledger.open_resources() == []
    assert [h for h, *_ in sent] == ["ops@10.0.0.10", "ops@10.0.0.11"]
    assert all(o == ["-o", "ProxyJump=ops@bastion"] and t == 69.0 for _, o, _, t in sent)
    assert "docker rm -f yeto-k31-trainer-i3-n1" in sent[1][2] and "trainer-i3/pgid" in sent[1][2]
    # a node that does not confirm keeps the row open
    ledger.created("ssh_trainer", "k31/trainer-i4", "trainer-i4", **{**attrs, "incarnation": 4})

    class Down:
        def run(self, n, script, *, timeout):
            raise subprocess.TimeoutExpired("ssh", timeout)

    (failed,) = cost_cleanup.cleanup(ledger, SimpleNamespace(ssh_ops=Down()))
    assert not failed.ok and "not confirmed" in failed.action
    assert [r.id for r in ledger.open_resources()] == ["k31/trainer-i4"]


def test_cost_prices_an_incarnation_by_the_declared_price_or_leaves_it_unpriced(tmp_path):
    ledger = ResourceLedger(tmp_path / "resources.jsonl", clock=lambda: 1000.0)
    nodes = [b300(i, price_per_card_hour=1.0) for i in range(6)]
    for i, priced in ((1, nodes), (2, [b300(0)])):
        ledger.created("ssh_trainer", f"k31/t-i{i}", f"t-i{i}", **ssh_trainer_ledger.created_attrs(
            run_id="k31", island="t", incarnation=i, nodes=priced, docker=True, kill_grace_s=30))
    summary = billing.resource_costs(ledger.resources(), now=1000.0 + 1800)
    by_id = {item.key: item for item in summary.items}
    assert by_id["ssh_trainer:k31/t-i1"].estimated_usd == pytest.approx(24.0)  # 48 cards, $1, 30 min
    assert not by_id["ssh_trainer:k31/t-i2"].priced
    assert "price_per_card_hour" in by_id["ssh_trainer:k31/t-i2"].note


# ---------------------------------------------------------------- multi-node staged releases
def test_a_multi_node_trainer_writes_staged_releases_on_rank_0():
    from yeto.rl.adapters.miles.sampler_island import (
        STAGED_EXPORT_ENV,
        staged_export_by_all_ranks,
    )

    one, six = SimpleNamespace(actor_num_nodes=1), SimpleNamespace(actor_num_nodes=6)
    assert staged_export_by_all_ranks(one, {}, staged=True)
    assert not staged_export_by_all_ranks(six, {}, staged=True)  # node 0 uploads only its own disk
    assert not staged_export_by_all_ranks(one, {STAGED_EXPORT_ENV: "miles"}, staged=True)
    assert not staged_export_by_all_ranks(one, {}, staged=False)
    assert staged_export_by_all_ranks(SimpleNamespace(), {}, staged=True)  # no layout: one node


def test_wired_multi_node_trainer_exports_with_export_hf(tmp_path):
    from yeto.rl.adapters.miles import buffer_rollout as br
    from yeto.rl.adapters.miles import sampler_island as si
    from yeto.rl.engine import sample_roles

    exported = []
    driver = SimpleNamespace(publisher=SimpleNamespace(publish=lambda s: None), emit=lambda *a, **k: None,
                             rollout=SimpleNamespace(_in_executor=lambda fn, *a: fn(None, *a)))
    args = SimpleNamespace(actor_num_nodes=6, yeto_rl_sample_buffer=sample_roles.SampleBufferConfig(
        role="trainer", store=str(tmp_path / "store")))
    wired = si.wire_policy_releases(
        driver, miles_args=args, runner=SimpleNamespace(run=lambda value: value),
        actor_model=SimpleNamespace(export_hf=lambda rollout_id, path: exported.append(rollout_id),
                                    run_plugin=lambda *a: pytest.fail("every rank wrote a node-local share")),
        environ={si.STAGING_ENV: str(tmp_path / "staging")})
    br._ACTIVE.pop(br.TRAINER, None)
    wired.export(5, Path(tmp_path / "staging"))
    assert exported == [5]


# ---------------------------------------------------------------- store credential on the node
def test_inventory_store_env_file_reaches_docker_by_env_file_and_rootfs_by_source():
    assert parse_node({**RAW, "store_env_file": "/etc/yeto/store.env"}, index=0).store_env_file == \
        "/etc/yeto/store.env"
    with pytest.raises(InventoryError, match="store_env_file"):
        parse_node({**RAW, "store_env_file": "store.env"}, index=0)
    nodes = [b300(i, store_env_file="/etc/yeto/store.env") for i in range(2)]
    script = start_script(b300_spec(nodes), 0, {}, remote_paths("k31", "t", 1), container="c0")
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    assert "--env-file /etc/yeto/store.env" in script and ". /etc/yeto/store.env" not in script
    rootfs = start_script(spec([node(store_env_file="/etc/yeto/store.env"),
                                node("ppu1", "u@10.0.0.2", store_env_file="/etc/yeto/store.env")]),
                          0, {}, remote_paths("run1", "t", 1))
    assert "set -a; . /etc/yeto/store.env; set +a" in rootfs and "--env-file" not in rootfs


def test_nodes_with_a_store_env_file_take_no_launcher_credential(monkeypatch):
    from yeto.launch.storage import store_object_envs

    for name in s3_store.CREDENTIAL_ENVS:
        monkeypatch.delenv(name, raising=False)
    args = SimpleNamespace(rl_checkpoint_store="s3://bkt/run", rl_sample_role="trainer",
                           rl_node_inventory_nodes=[b300(i, store_env_file="/etc/yeto/store.env")
                                                    for i in range(6)])
    envs = store_object_envs(args, "ssh", num_nodes=6)
    assert envs[s3_store.CREDENTIALS_ENV] == s3_store.CREDENTIALS_AWS_ENV
    assert not any(name in envs for name in s3_store.CREDENTIAL_ENVS)
    assert envs[s3_store.STAGE_ENV] == s3_store.DEFAULT_STAGE
    mixed = SimpleNamespace(**{**vars(args), "rl_node_inventory_nodes": [b300(0, store_env_file="/e"), b300(1)]})
    with pytest.raises(ValueError, match="every trainer node"):
        store_object_envs(mixed, "ssh", num_nodes=2)
    without = SimpleNamespace(**{**vars(args), "rl_node_inventory_nodes": [b300(0), b300(1)]})
    with pytest.raises(ValueError, match="store credentials"):  # no file, no launcher key
        store_object_envs(without, "ssh", num_nodes=2)


class EnvFileOps(ScratchOps):
    def __init__(self, mode="600", rc=0):
        super().__init__()
        self.mode, self.env_rc = mode, rc

    def run(self, n, script, *, timeout=300.0):
        if "stat -c %a" in script:
            self.runs.append((n.name, script))
            return subprocess.CompletedProcess([], self.env_rc, stdout=f"{self.mode}\n", stderr="")
        return super().run(n, script, timeout=timeout)


@pytest.mark.parametrize("mode, rc, match", [("640", 0, "has mode 640.*chmod 600"),
                                             ("600", 1, "cannot read|not a file")])
def test_preflight_refuses_a_store_env_file_others_may_read(tmp_path, checked_cards, mode, rc, match):
    nodes = [b300(i, store_env_file="/etc/yeto/store.env") for i in range(2)]
    ok = EnvFileOps("400")
    SshTrainerIslands(b300_spec(nodes), name="t", jobs_path=tmp_path / "j.json", ops=ok).preflight()
    assert [n for n, s in ok.runs if "stat -c %a /etc/yeto/store.env" in s] == ["b300-0", "b300-1"]
    bad = SshTrainerIslands(b300_spec(nodes), name="t", jobs_path=tmp_path / "j.json", ops=EnvFileOps(mode, rc))
    with pytest.raises(SshIslandError, match=match):
        bad.preflight()

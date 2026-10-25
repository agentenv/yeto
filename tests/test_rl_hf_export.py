"""--rl-export-hf-rounds: the model of chosen rounds as HF directories (yeto.rl.engine.hf_export).

CPU only: the backend writer is a double, no Ray, no Miles."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from yeto.rl.engine.hf_export import (
    COMPLETE_MARKER,
    FINAL_POINTER,
    HF_EXPORT_EVENT,
    ExportRounds,
    HfRoundExports,
    export_store_path,
)
from yeto.rl.engine.resume import CheckpointStore


def _write_shards(rollout_id: int, directory: Path) -> None:
    """The Miles writer's result: the files, then its completion marker."""
    directory.mkdir(parents=True)
    (directory / "config.json").write_text("{}")
    (directory / "model-00001.safetensors").write_bytes(b"w" * (rollout_id + 10))
    (directory / COMPLETE_MARKER).touch()


class CommitViewStore(CheckpointStore):
    """A store whose readers see only what the last commit covered (a Modal
    Volume seen from another container): ``views`` holds every commit's files."""

    def __init__(self, root):
        super().__init__(root)
        self.views: list[set[str]] = []

    def commit(self) -> float:
        self.views.append({p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file()})
        return 0.0


# ---------------------------------------------------------------- the rules
def test_rounds_parse_to_ascending_numbers_and_final():
    assert ExportRounds.parse("48,96") == ExportRounds((48, 96))
    assert ExportRounds.parse(" 96, final,48,48,") == ExportRounds((48, 96), final=True)
    assert str(ExportRounds.parse("final,56")) == "56,final"
    assert not ExportRounds.parse(None) and not ExportRounds.parse("")
    for bad in ("0", "-1", "a", "4.5", "48;96", "FINAL"):
        with pytest.raises(ValueError, match="--rl-export-hf-rounds"):
            ExportRounds.parse(bad)


def test_final_is_the_round_whose_boundary_stops_the_island():
    rounds = ExportRounds.parse("2,final")
    assert [rounds.due(r, final=False) for r in range(4)] == [False, True, False, False]
    assert rounds.due(6, final=True) and not ExportRounds((2,)).due(6, final=True)


def test_each_island_exports_under_its_own_directory():
    assert export_store_path("/s/hf/", learner_id=1) == "/s/hf/island1"


def test_the_marker_name_is_the_miles_one():
    config = pytest.importorskip("miles.utils.hf_utils.config")
    assert COMPLETE_MARKER == config.HF_EXPORT_COMPLETE_MARKER


# ---------------------------------------------------------------- one export
def test_export_puts_the_round_into_the_store_and_times_it(tmp_path):
    store = CheckpointStore(tmp_path / "store" / "hf")
    ticks = iter([0.0, 2.0, 5.0])
    exports = HfRoundExports(ExportRounds((2, 4)), store, _write_shards, staging=tmp_path / "staging",
                             clock=lambda: next(ticks))
    record = exports.export(1, policy_version=2, final=False)
    assert record == {"round": 2, "final": False, "policy_version": 2, "path": str(store.root / "round-2"),
                      "store_kind": "local", "ok": True, "files": 2, "bytes": 13, "write_s": 2.0,
                      "upload_s": 3.0, "seconds": 5.0, "upload_attempts": 1}
    assert sorted(p.name for p in (store.root / "round-2").iterdir()) == [
        COMPLETE_MARKER, "config.json", "model-00001.safetensors"]
    assert json.loads((store.root / "round-2" / COMPLETE_MARKER).read_text()) == {
        "round": 2, "final": False, "policy_version": 2, "files": 2, "bytes": 13}
    assert not (store.root / FINAL_POINTER).exists()
    assert not (tmp_path / "staging" / "round-2").exists()  # the staging copy is gone


def test_a_directory_with_the_marker_is_already_committed(tmp_path):
    store = CommitViewStore(tmp_path / "hf" / "island0")
    (store.root / "round-3").mkdir(parents=True)
    (store.root / "round-3" / "stale.safetensors").write_bytes(b"old")  # an older export of round 3
    exports = HfRoundExports(ExportRounds((3,), final=True), store, _write_shards, staging=tmp_path / "staging")
    record = exports.export(2, policy_version=3, final=True)
    assert record["ok"], record
    files = {"round-3/config.json", "round-3/model-00001.safetensors"}
    marker = f"round-3/{COMPLETE_MARKER}"
    assert store.views[-1] == files | {marker, FINAL_POINTER}  # the stale file is gone
    for view in store.views:  # whenever a reader sees the marker, it sees every file of the round
        assert marker not in view or files <= view
    assert any(files <= view and marker not in view for view in store.views)  # files first, marker after
    assert json.loads((store.root / FINAL_POINTER).read_text()) == {
        "round": 3, "final": True, "policy_version": 3, "files": 2, "bytes": 14, "path": "round-3"}


def test_a_write_without_the_marker_is_a_failed_export(tmp_path):
    def unfinished(rollout_id, directory):
        directory.mkdir(parents=True)
        (directory / "model-00001.safetensors").write_bytes(b"part")

    store = CommitViewStore(tmp_path / "hf")
    record = HfRoundExports(ExportRounds((1,)), store, unfinished, staging=tmp_path / "staging").export(
        0, policy_version=1, final=False)
    assert not record["ok"] and record["error"].startswith("write: RuntimeError: the writer left no .complete")
    assert store.views == [] and not (store.root / "round-1").exists()


def test_a_failed_export_is_taped_and_the_island_goes_on(tmp_path):
    class BrokenStore(CheckpointStore):
        def put_directory(self, local, relative):
            raise OSError("volume full")

    waits = []
    exports = HfRoundExports(ExportRounds((1,)), BrokenStore(tmp_path / "store"), _write_shards,
                             staging=tmp_path / "staging", sleep=waits.append)
    record = exports.export(0, policy_version=1, final=True)
    assert (record["ok"], record["error"], record["bytes"]) == (False, "upload: OSError: volume full", 12)
    assert (record["upload_attempts"], record["upload_errors"], waits) == (
        3, ["OSError: volume full"] * 2, [10.0, 30.0])
    # the model stays on local disk, its marker put back: one upload restores the export
    kept = tmp_path / "staging" / "round-1"
    assert record["staging_kept"] == str(kept)
    assert sorted(p.name for p in kept.iterdir()) == [COMPLETE_MARKER, "config.json", "model-00001.safetensors"]
    assert json.loads((kept / COMPLETE_MARKER).read_text()) == {
        "round": 1, "final": True, "policy_version": 1, "files": 2, "bytes": 12}

    def broken_write(rollout_id, directory):
        directory.mkdir(parents=True)
        raise RuntimeError("rank 3 died")

    failing = HfRoundExports(ExportRounds((1,)), CheckpointStore(tmp_path / "store"), broken_write,
                             staging=tmp_path / "staging2", sleep=waits.append)
    record = failing.export(0, policy_version=1, final=False)
    assert (record["ok"], record["error"], record["bytes"], record["upload_s"], record["upload_attempts"]) == (
        False, "write: RuntimeError: rank 3 died", 0, 0.0, 0)
    assert "staging_kept" not in record and not (tmp_path / "staging2" / "round-1").exists()


def test_an_upload_that_fails_once_is_retried(tmp_path):
    class FlakyStore(CommitViewStore):
        failures = 1

        def put_directory(self, local, relative):
            if self.failures:
                self.failures -= 1
                raise OSError("transient")
            super().put_directory(local, relative)

    waits = []
    store = FlakyStore(tmp_path / "hf")
    record = HfRoundExports(ExportRounds((1,)), store, _write_shards, staging=tmp_path / "staging",
                            sleep=waits.append).export(0, policy_version=1, final=False)
    assert (record["ok"], record["upload_attempts"], record["upload_errors"], waits) == (
        True, 2, ["OSError: transient"], [10.0])
    assert (store.root / "round-1" / COMPLETE_MARKER).is_file()
    assert "staging_kept" not in record and not (tmp_path / "staging" / "round-1").exists()


def test_at_most_keep_failed_staging_copies_stay(tmp_path):
    class BrokenStore(CheckpointStore):
        def put_directory(self, local, relative):
            raise OSError("volume full")

    exports = HfRoundExports(ExportRounds((1, 2, 3)), BrokenStore(tmp_path / "store"), _write_shards,
                             staging=tmp_path / "staging", sleep=lambda s: None, keep_failed=2)
    records = [exports.export(r, policy_version=r + 1, final=False) for r in range(3)]
    assert [r["staging_kept"].rsplit("/", 1)[1] for r in records] == ["round-1", "round-2", "round-3"]
    assert records[2]["staging_dropped"] == [str(tmp_path / "staging" / "round-1")]
    assert sorted(p.name for p in (tmp_path / "staging").iterdir()) == ["round-2", "round-3"]


# ---------------------------------------------------------------- Modal Volume store
class _MissingPathVolume:
    """A Modal Volume fake that reports a missing path as Modal's server does for
    ``remove_file``: ``InvalidError("No such file or directory")``. ``backend`` is
    the Volume; ``listdir`` lists an existing directory; ``refuse`` paths fail to
    be removed although they exist."""

    class InvalidError(Exception):
        pass

    def __init__(self, backend, refuse=()):
        self.backend = backend
        backend.mkdir(parents=True, exist_ok=True)  # the Volume root always exists
        self.refuse = set(refuse)
        self.removed = []

    def _path(self, remote):
        return self.backend / remote.lstrip("/")

    def listdir(self, path, recursive=False):
        target = self._path(path)
        if not target.is_dir():
            raise self.InvalidError(f"No such file or directory: {path}")
        return [SimpleNamespace(path=str(child.relative_to(self.backend))) for child in target.iterdir()]

    def remove_file(self, path, recursive=False):
        import shutil

        if not self._path(path).exists() or path in self.refuse:
            raise self.InvalidError(f"No such file or directory: {path}")
        self.removed.append(path)
        shutil.rmtree(self._path(path))

    def batch_upload(self, force=False):
        import contextlib
        import shutil

        volume = self

        @contextlib.contextmanager
        def batch():
            ops = []
            yield SimpleNamespace(put_directory=lambda local, remote: ops.append(("dir", local, remote)),
                                  put_file=lambda local, remote: ops.append(("file", local, remote)))
            for kind, local, remote in ops:
                target = volume._path(remote)
                target.parent.mkdir(parents=True, exist_ok=True)
                if kind == "dir":
                    shutil.copytree(local, target, dirs_exist_ok=force)
                else:
                    shutil.copyfile(local, target)

        return batch()


def _modal_store(tmp_path, volume):
    from yeto.cloud.modal_ckpt_store import ModalVolumeStore

    mount = tmp_path / "mount"
    return ModalVolumeStore(mount / "run-a" / "hf" / "island0", "vol", volume_factory=lambda name: volume,
                            mount=mount)


def test_the_first_export_to_a_modal_volume_succeeds(tmp_path):
    # S27 s27-bb-b1 round 56: removing the not yet existing round directory raised
    # InvalidError "No such file or directory", so every first export failed
    volume = _MissingPathVolume(tmp_path / "volume")
    store = _modal_store(tmp_path, volume)
    record = HfRoundExports(ExportRounds((1,)), store, _write_shards, staging=tmp_path / "staging",
                            sleep=lambda s: None).export(0, policy_version=1, final=False)
    assert (record["ok"], record["upload_attempts"]) == (True, 1), record
    exported = tmp_path / "volume" / "run-a" / "hf" / "island0" / "round-1"
    assert sorted(p.name for p in exported.iterdir()) == [COMPLETE_MARKER, "config.json", "model-00001.safetensors"]
    # a relaunch exports the round again: the older directory is removed first
    record = HfRoundExports(ExportRounds((1,)), store, _write_shards, staging=tmp_path / "staging",
                            sleep=lambda s: None).export(0, policy_version=1, final=False)
    assert record["ok"] and volume.removed == ["/run-a/hf/island0/round-1"]


def test_a_modal_removal_fails_only_when_the_path_stays(tmp_path):
    from yeto.cloud.modal_ckpt_store import _volume_has

    volume = _MissingPathVolume(tmp_path / "volume", refuse={"/run-a/hf/island0/round-1"})
    store = _modal_store(tmp_path, volume)
    store.remove_tree("round-1")  # missing, and its parents are missing too: no error
    (tmp_path / "volume" / "run-a" / "hf" / "island0" / "round-1").mkdir(parents=True)
    assert _volume_has(volume, "/run-a/hf/island0/round-1") and not _volume_has(volume, "/run-a/hf/x")
    with pytest.raises(_MissingPathVolume.InvalidError):
        store.remove_tree("round-1")  # there, but the removal failed


def test_the_runner_script_removes_the_same_way(tmp_path, monkeypatch):
    import sys

    from yeto.cloud.modal_ckpt_store import _REMOVE_SCRIPT

    volume = _MissingPathVolume(tmp_path / "volume")
    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(Volume=SimpleNamespace(from_name=lambda name: volume)))
    for remote in ("/run-a/hf/island0/round-1", "/run-a/hf/island0/round-1"):
        monkeypatch.setattr(sys, "argv", ["-c", "vol", remote])
        exec(compile(_REMOVE_SCRIPT, "<remove>", "exec"), {"__name__": "__main__"})
        (tmp_path / "volume" / "run-a" / "hf" / "island0" / "round-1").mkdir(parents=True, exist_ok=True)
    assert volume.removed == ["/run-a/hf/island0/round-1"]


# ---------------------------------------------------------------- the driver
def test_driver_exports_at_the_boundary_of_the_chosen_and_the_final_round(tmp_path):
    from yeto.rl.engine.algorithm import AlgorithmSpec
    from yeto.rl.engine.bridges import LocalOnlySync
    from yeto.rl.engine.driver import EventTape, IslandDriver
    from yeto.rl.engine.fake import FakeEngine, fake_capabilities

    engine = FakeEngine(tensors={"base_model.model.layer.lora_A.weight": torch.zeros(1, 2)}, step_delta=1.0)
    driver = IslandDriver(learner_id=0, rollout=engine.rollout, trainer=engine.trainer,
                          policy_state=engine.policy_state, publisher=engine.publisher,
                          placement=engine.placement, capabilities=fake_capabilities(),
                          algorithm=AlgorithmSpec(), sync=LocalOnlySync(3),
                          events=EventTape(tmp_path / "e.jsonl", 0))
    store = CheckpointStore(tmp_path / "store")
    published_at_write, order = [], []
    sync_published = driver.sync.published

    def published(driver, *, rollout_id, policy_hash):  # FullParameterSync ACKs FINAL here
        order.append(("published", rollout_id))
        return sync_published(driver, rollout_id=rollout_id, policy_hash=policy_hash)

    driver.sync.published = published

    def write(rollout_id, directory):
        published_at_write.append([c[1] for c in engine.calls if c[0] == "publish"])
        order.append(("export", rollout_id))
        _write_shards(rollout_id, directory)

    driver.hf_exports = HfRoundExports(ExportRounds((2,), final=True), store, write, staging=tmp_path / "staging")
    driver.run()
    events = [json.loads(line) for line in (tmp_path / "e.jsonl").read_text().splitlines()]
    exported = [e for e in events if e["event"] == HF_EXPORT_EVENT]
    assert [(e["rollout_id"], e["round"], e["final"], e["policy_version"], e["ok"], e["bytes"])
            for e in exported] == [(1, 2, False, 2, True, 13), (2, 3, True, 3, True, 14)]
    assert {p.name for p in store.root.iterdir()} == {"round-2", "round-3", FINAL_POINTER}
    assert json.loads((store.root / FINAL_POINTER).read_text())["path"] == "round-3"
    # on the boundary of each round: after its sync phase, before its policy is published
    phases = [(e["phase"], e.get("rollout_id")) for e in events if e["event"] == "rl_driver_phase"]
    for rollout_id in (1, 2):
        assert phases[phases.index(("hf_export", rollout_id)) - 1] == ("sync", rollout_id)
    assert published_at_write == [[0, 1], [0, 1, 2]]
    # the final export (rollout 2) comes before the sync hears of the last publication (the final ACK)
    assert order[-2:] == [("export", 2), ("published", 3)]


# ---------------------------------------------------------------- Miles wiring
def test_miles_island_writes_with_every_rank_on_one_node_else_rank0(tmp_path):
    from yeto.rl.adapters.miles.entry import hf_exports_for
    from yeto.rl.adapters.miles.release_plugin import WRITE_FULL_CHECKPOINT

    calls = []
    actor = SimpleNamespace(run_plugin=lambda path, kwargs: calls.append(("plugin", path, kwargs)),
                            export_hf=lambda rollout_id, path: calls.append(("export_hf", rollout_id, path)))
    runner = SimpleNamespace(run=lambda value: value)
    config = {"rounds": "2,final", "store": str(tmp_path / "hf")}
    assert hf_exports_for(SimpleNamespace(), actor_model=actor, runner=runner) is None
    exports = hf_exports_for(SimpleNamespace(yeto_rl_hf_exports=config, actor_num_nodes=1, lora_rank=0),
                             actor_model=actor, runner=runner, environ={})
    assert exports.rounds == ExportRounds((2,), final=True) and exports.store.root == tmp_path / "hf"
    exports.write(1, tmp_path / "d")
    assert calls == [("plugin", WRITE_FULL_CHECKPOINT, {"path": str(tmp_path / "d"), "rollout_id": 1})]
    calls.clear()
    across = hf_exports_for(SimpleNamespace(yeto_rl_hf_exports=config, actor_num_nodes=2, lora_rank=0),
                            actor_model=actor, runner=runner, environ={})
    across.write(1, tmp_path / "d")
    assert calls == [("export_hf", 1, str(tmp_path / "d"))]
    with pytest.raises(ValueError, match="full weights"):
        hf_exports_for(SimpleNamespace(yeto_rl_hf_exports=config, lora_rank=8), actor_model=actor, runner=runner)


def _learner(*extra, no_sync=False):
    from test_rl_engine_selection import _learner_argv

    from yeto.rl.adapters.miles import island_entry as learner

    argv = _learner_argv(extra + (("--rl-single-island-no-sync",) if no_sync else ()))
    if no_sync:
        i = argv.index("--syncer")
        del argv[i:i + 2]
    return learner, learner.parse_args(argv)


@pytest.mark.parametrize("no_sync", [True, False])
def test_island_takes_the_rounds_and_its_own_export_store(no_sync):
    learner, parsed = _learner("--rl-export-hf-rounds", "final,96,48", "--rl-export-hf-store", "/root/s/hf",
                               no_sync=no_sync)
    miles_args = SimpleNamespace()
    learner.apply_ports_infra_switches(parsed, miles_args, {})
    assert miles_args.yeto_rl_hf_exports == {"rounds": "48,96,final", "store": "/root/s/hf/island0"}
    _, plain = _learner()
    learner.apply_ports_infra_switches(plain, untouched := SimpleNamespace(), {})
    assert not hasattr(untouched, "yeto_rl_hf_exports")


@pytest.mark.parametrize("extra", [
    ("--rl-export-hf-rounds", "4"),  # no export store
    ("--rl-export-hf-store", "/s"),  # no rounds
    ("--rl-export-hf-rounds", "0", "--rl-export-hf-store", "/s"),
])
def test_island_refusals(extra):
    with pytest.raises(SystemExit):
        _learner(*extra)


# ---------------------------------------------------------------- launcher
def _cli(*extra):
    from test_rl_engine_selection import _cli as cli

    return cli(extra)


def _island_flag(flags: str, flag: str) -> str:
    tokens = shlex.split(flags)
    return tokens[tokens.index(flag) + 1]


def test_synced_launch_forwards_the_rounds_and_the_store(capsys):
    from yeto.launch import rl_ports

    args = _cli("--rl-checkpoint-store", "modal-volume://vol/run-1", "--rl-export-hf-rounds", "final,3,1")
    rl_ports._check_hf_export_launch_flags(args)
    assert rl_ports._check_resume_launch_flags(args) is False  # the exports are a use of the store
    flags = rl_ports._ports_infra_flags(args)[1]
    assert _island_flag(flags, "--rl-export-hf-rounds") == "1,3,final"
    assert _island_flag(flags, "--rl-export-hf-store") == "/root/yeto-checkpoint-store/run-1/hf"
    assert "WARNING" not in capsys.readouterr().err
    assert "--rl-export-hf-store" not in rl_ports._ports_infra_flags(_cli("--rl-checkpoint-store", "/s",
                                                                          "--rl-cut-every", "2"))[1]


def test_rounds_past_the_expected_rounds_warn(capsys):
    from yeto.launch import rl_ports

    args = _cli("--rl-checkpoint-store", "/s", "--rl-export-hf-rounds", "2,999", "--total-steps", "4")
    rl_ports._check_hf_export_launch_flags(args)
    out = capsys.readouterr()
    assert "WARNING: --rl-export-hf-rounds: rounds [999] are past the 4 rounds" in out.err
    assert not out.out  # stdout carries the dry-run plan


@pytest.mark.parametrize("extra, match", [
    (("--rl-export-hf-rounds", "2"), "needs --rl-checkpoint-store"),
    (("--rl-export-hf-rounds", "2,x", "--rl-checkpoint-store", "/s"), "neither a round number"),
    (("--rl-export-hf-rounds", "2", "--rl-checkpoint-store", "/s", "--rl-sample-role", "trainer"),
     "sample role both"),
])
def test_launch_refusals(extra, match):
    from yeto.launch import rl_ports

    with pytest.raises(ValueError, match=match):
        rl_ports._check_hf_export_launch_flags(_cli(*extra))


def test_verl_lora_and_elastic_launches_are_refused():
    from yeto.launch import rl_ports

    args = _cli("--rl-export-hf-rounds", "2", "--rl-checkpoint-store", "/s")
    # S27 G-DS: a verl launch exports only from synced full-parameter islands
    args.rl_backend = "verl"
    with pytest.raises(ValueError, match="verl: synced full-parameter islands"):
        rl_ports._check_hf_export_launch_flags(args)
    args.rl_backend, args.rl_elastic = "miles", True
    with pytest.raises(ValueError, match="--rl-elastic"):
        rl_ports._check_hf_export_launch_flags(args)

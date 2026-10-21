"""S25 C1: a sampler island on spot capacity is reclaimed and replaced.

CPU only (no Miles, no Ray): the resume point read from the buffer, the
resumable round writer (no round written twice, no prompt drawn twice), several
sampler islands writing their own shards of one shared-directory buffer that
the trainer merges, the launch composition of one trainer and two samplers, and
the Modal runner's reclaim marker for sampler islands.
"""

from __future__ import annotations

import shlex
import signal as _signal
from types import SimpleNamespace

import pytest

from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from tests.buffer_rollout_procs import sample
from yeto.rl.adapters.miles import buffer_rollout as br
from yeto.rl.engine.policy_release import PolicyFollower, PolicyReleaseStore
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer, SampleBufferError
from yeto.rl.engine.sampler_loop import SamplerLoop
from yeto.rl.engine.sampler_resume import (
    RECLAIM_MARKER_ENV,
    Member,
    SamplerResumeError,
    SamplerRound,
    reclaim_requested,
    resume_point,
    write_member,
)


def _group(g: int, version: int = 1) -> list[dict]:
    return [sample(2 * g, g, version), sample(2 * g + 1, g, version)]


def _cursor(offset: int) -> dict[str, int]:
    return {"sample_offset": offset, "epoch_id": 0, "sample_group_index": offset, "sample_index": 2 * offset}


def _round(rid: int, groups: int) -> list[tuple[str, list[dict]]]:
    return [(f"r{rid}-g{g}", _group(g)) for g in range(groups)]


def _shard_env(store, sampler=None, shards=("island0", "island1")):
    env = {br.BUFFER_ROOT_ENV: str(store), br.SHARDS_ENV: ",".join(shards)}
    if sampler is not None:
        env[br.SAMPLER_ID_ENV] = sampler
    return env


# ---------------------------------------------------------------- resume point
def test_resume_point_of_an_empty_buffer_starts_at_round_zero():
    resume = resume_point([], "island0")
    assert (resume.next_rollout_id, resume.data_cursor, resume.resumed) == (0, None, False)


def test_resume_point_continues_after_the_newest_written_round(tmp_path):
    buffer = FileSampleBuffer(CheckpointStore(tmp_path), batch_size=1)
    writer = br.SamplerRoundWriter(buffer, "island0", incarnation="a")
    assert writer.write_round(0, _round(0, 2), _cursor(2)) == 2
    assert writer.write_round(1, _round(1, 2), _cursor(4)) == 2
    other = br.SamplerRoundWriter(buffer, "island9", incarnation="z")
    other.write_round(5, _round(5, 1), _cursor(9))  # another sampler: not ours
    resume = resume_point(buffer.entries(), "island0")
    assert (resume.next_rollout_id, resume.data_cursor) == (2, _cursor(4))
    assert (resume.written_rounds, resume.partial_rounds, resume.groups) == ((0, 1), (), 4)
    assert resume.incarnations == ("a",)


def test_a_round_cut_short_by_the_reclaim_keeps_its_groups_and_is_not_written_again(tmp_path):
    """The island leaves while it writes round 1: the written group stays, the
    rest are dropped; the replacement continues at round 2 from round 1's cursor
    (round 1's prompts are not drawn again)."""
    store = CheckpointStore(tmp_path)
    leaving = {"now": False}
    old = br.SamplerRoundWriter(FileSampleBuffer(store, batch_size=1), "island0", incarnation="a",
                                reclaimed=lambda: leaving["now"])
    old.write_round(0, _round(0, 3), _cursor(3))

    class Leaving(list):
        def __iter__(self):  # the reclaim lands after the first group of round 1
            for i, item in enumerate(super().__iter__()):
                leaving["now"] = i >= 1
                yield item

    assert old.write_round(1, Leaving(_round(1, 3)), _cursor(6)) == 1
    assert old.dropped_groups == 2
    new = br.SamplerRoundWriter(FileSampleBuffer(store, batch_size=1), "island0", incarnation="b")
    assert (new.resume.next_rollout_id, new.resume.data_cursor) == (2, _cursor(6))
    assert new.resume.partial_rounds == (1,)
    for rid in (0, 1):
        with pytest.raises(SamplerResumeError, match="already in the sample buffer"):
            new.write_round(rid, _round(rid, 1), _cursor(0))
    new.write_round(2, _round(2, 3), _cursor(9))
    ids = [e.sample_id for e in FileSampleBuffer(store, batch_size=1).entries()]
    assert len(ids) == len(set(ids)) == 3 + 1 + 3
    assert [i.split("-")[1] for i in ids] == ["a"] * 4 + ["b"] * 3


def test_a_round_still_generating_at_the_reclaim_is_generated_again(tmp_path):
    store = CheckpointStore(tmp_path)
    old = br.SamplerRoundWriter(FileSampleBuffer(store, batch_size=1), "island0", incarnation="a",
                                reclaimed=lambda: True)
    assert old.write_round(0, _round(0, 2), _cursor(2)) == 0  # left before the write
    new = br.SamplerRoundWriter(FileSampleBuffer(store, batch_size=1), "island0", incarnation="b")
    assert (new.resume.next_rollout_id, new.resume.data_cursor) == (0, None)
    assert new.write_round(0, _round(0, 2), _cursor(2)) == 2


def test_resume_point_refuses_a_buffer_with_two_cursors_for_one_round():
    def entry(cursor, last):
        return SimpleNamespace(sampler_round=SamplerRound("island0", "a", 0, cursor, last).record())

    with pytest.raises(SamplerResumeError, match="two data cursors"):
        resume_point([entry(_cursor(2), False), entry(_cursor(3), True)], "island0")
    with pytest.raises(SamplerResumeError, match="two last groups"):
        resume_point([entry(_cursor(2), True), entry(_cursor(2), True)], "island0")


def test_seek_puts_the_data_source_where_its_load_would_and_only_once(tmp_path):
    class Dataset:
        def __init__(self):
            self.shuffled = []

        def shuffle(self, epoch):
            self.shuffled.append(epoch)

    source = SimpleNamespace(sample_offset=0, epoch_id=0, sample_group_index=0, sample_index=0,
                             dataset=Dataset())
    store = CheckpointStore(tmp_path)
    br.SamplerRoundWriter(FileSampleBuffer(store, batch_size=1), "island0", incarnation="a"
                          ).write_round(0, _round(0, 1), {**_cursor(5), "epoch_id": 1})
    writer = br.SamplerRoundWriter(FileSampleBuffer(store, batch_size=1), "island0", incarnation="b")
    assert writer.seek(SimpleNamespace(__self__=source), shuffle=True)["sample_offset"] == 5
    assert (source.sample_offset, source.epoch_id, source.sample_index) == (5, 1, 10)
    assert source.dataset.shuffled == [1]
    assert writer.seek(source, shuffle=True) is None  # the second round: no seek
    with pytest.raises(br.SampleBufferRolloutError, match="cannot seek"):
        br.seek_data_source(SimpleNamespace(sample_offset=0), _cursor(1), shuffle=False)


def test_a_writer_without_resume_writes_as_before(tmp_path):
    """A convertible node keeps its own rollout numbering: no round check."""
    store = CheckpointStore(tmp_path)
    writer = br.SamplerRoundWriter(FileSampleBuffer(store, batch_size=1), "n0", incarnation="a",
                                   resume=False)
    writer.write_round(0, _round(0, 1), None)
    writer.write_round(0, [("r0-again", _round(0, 1)[0][1])], None)  # payload names stay unique
    assert writer.seek(object(), shuffle=True) is None
    assert len(FileSampleBuffer(store, batch_size=1)) == 2


# ---------------------------------------------------------------- two samplers, one trainer
def test_two_samplers_write_shards_the_trainer_merges_and_a_replacement_continues(tmp_path):
    a = br.SamplerRoundWriter(br.open_buffer(_shard_env(tmp_path, "island0"), sampler=True),
                              "island0", incarnation="a1")
    b = br.SamplerRoundWriter(br.open_buffer(_shard_env(tmp_path, "island1"), sampler=True),
                              "island1", incarnation="b1")
    assert a.buffer.shard == "island0" and not (tmp_path / "sample-buffer/index.jsonl").exists()
    trainer = br.open_buffer(_shard_env(tmp_path), batch_size=2, reader=True)
    reader = br.BatchReader(trainer, max_policy_age=2, wait_s=0.0, poll_s=0.0)
    a.write_round(0, _round(0, 2), _cursor(2))
    b.write_round(0, _round(0, 2), _cursor(2))
    taken, _ = reader.take(0, 4, current_version=1)
    assert len(taken) == 4
    # island0 is reclaimed after one more round; its replacement continues at round 2
    a.write_round(1, _round(1, 2), _cursor(4))
    a2 = br.SamplerRoundWriter(br.open_buffer(_shard_env(tmp_path, "island0"), sampler=True),
                               "island0", incarnation="a2")
    assert (a2.resume.next_rollout_id, a2.resume.incarnations) == (2, ("a1",))
    a2.write_round(2, _round(2, 2), _cursor(6))
    b.write_round(1, _round(1, 2), _cursor(4))
    taken2, _ = reader.take(1, 6, current_version=2)
    ids = [e.sample_id for e in trainer.entries()]
    assert len(taken2) == 6 and len(ids) == len(set(ids)) == 10  # none repeated, none lost
    assert trainer.shard_positions(trainer.read_cursor) == {"island0": 6, "island1": 4}
    # the data ends only when every named shard has ended
    a2.buffer.mark_data_end()
    trainer.refresh()
    assert not trainer.data_ended
    b.buffer.mark_data_end()
    trainer.refresh()
    assert trainer.data_ended


def test_the_data_does_not_end_before_a_named_shard_exists(tmp_path):
    a = br.open_buffer(_shard_env(tmp_path, "island0"), sampler=True)
    a.mark_data_end()
    trainer = br.open_buffer(_shard_env(tmp_path), reader=True)
    assert not trainer.data_ended  # island1 has not written anything yet


def test_a_restarted_trainer_merges_no_record_twice(tmp_path):
    a = br.SamplerRoundWriter(br.open_buffer(_shard_env(tmp_path, "island0"), sampler=True),
                              "island0", incarnation="a")
    a.write_round(0, _round(0, 3), _cursor(3))
    assert len(br.open_buffer(_shard_env(tmp_path), reader=True)) == 3
    a.write_round(1, _round(1, 1), _cursor(4))
    again = br.open_buffer(_shard_env(tmp_path), reader=True)
    assert len(again) == 4
    assert [r.get("shard_pos") for r in map(_json, _lines(tmp_path))] == [0, 1, 2, 3]


def _lines(root):
    return (root / "sample-buffer/index.jsonl").read_text().splitlines()


def _json(line):
    import json

    return json.loads(line)


def test_the_trainer_merges_one_refresh_with_one_store_commit(tmp_path):
    class Counting(CheckpointStore):
        commits = 0

        def commit(self):
            Counting.commits += 1

    writer = br.SamplerRoundWriter(FileSampleBuffer(CheckpointStore(tmp_path), batch_size=1,
                                                    index="shards/island0.jsonl"),
                                   "island0", incarnation="a")
    writer.write_round(0, _round(0, 5), _cursor(5))
    Counting.commits = 0
    trainer = FileSampleBuffer(Counting(tmp_path), batch_size=1, merge_shards=True,
                               expected_shards=["island0"])
    assert len(trainer) == 5 and Counting.commits == 1


def test_expected_shards_need_a_merging_buffer(tmp_path):
    with pytest.raises(SampleBufferError, match="expected_shards needs merge_shards"):
        FileSampleBuffer(CheckpointStore(tmp_path), batch_size=1, expected_shards=["a"])


def test_island_env_names_the_shards_and_refuses_a_sampler_outside_them():
    env = br.island_env("sampler", store_root="/s", sampler_id="island1", shards=["island0", "island1"])
    assert env[br.SHARDS_ENV] == "island0,island1" and env[br.SAMPLER_ID_ENV] == "island1"
    with pytest.raises(ValueError, match="not one of the buffer shards"):
        br.island_env("sampler", store_root="/s", sampler_id="island2", shards=["island0"])
    assert br.SHARDS_ENV in br.ISLAND_ENVS and br.INCARNATION_ENV in br.ISLAND_ENVS


# ---------------------------------------------------------------- the sampler loop leaves
def test_the_sampler_loop_stops_on_the_reclaim_and_marks_no_data_end(tmp_path):
    store = CheckpointStore(tmp_path)
    marker = tmp_path / "reclaim"
    env = {RECLAIM_MARKER_ENV: str(marker)}
    buffer = FileSampleBuffer(store, batch_size=1)
    writer = br.SamplerRoundWriter(buffer, "island0", incarnation="a",
                                   reclaimed=lambda: reclaim_requested(env))
    events = []

    def generate(rid):
        writer.write_round(rid, _round(rid, 2), _cursor(2 * (rid + 1)))
        if rid == 1:
            marker.write_text("{}")  # the container gets SIGINT during round 1

    loop = SamplerLoop(follower=PolicyFollower(PolicyReleaseStore(store), lambda r, d: None),
                       buffer=buffer, generate=generate, num_rollouts=5, max_backlog_groups=100,
                       emit=lambda e, **f: events.append(e), poll_s=0.0,
                       should_stop=lambda: reclaim_requested(env))
    assert loop.run() == 2
    assert "rl_sampler_stopped" in events and not buffer.data_ended
    resume = resume_point(FileSampleBuffer(store, batch_size=1).entries(), "island0")
    assert resume.next_rollout_id == 2
    loop.should_stop = lambda: False
    marker.unlink()
    assert loop.run(start=resume.next_rollout_id) == 3 and buffer.data_ended


def test_member_records_are_one_file_per_incarnation(tmp_path):
    store = CheckpointStore(tmp_path)
    first = write_member(store, Member("island0", "a", 1.0, {"next_rollout_id": 0}))
    member = Member("island0", "b", 2.0, {"next_rollout_id": 3}, leave_reason="PREEMPTED")
    second = write_member(store, member)
    assert first != second and first.exists() and "PREEMPTED" in second.read_text()


# ---------------------------------------------------------------- launch composition
GPU3 = "modal:1xh100,modal:1xh100,modal:1xh100"
STORE = "modal-volume://yeto-c1/run-a"
GROUP = ("--rl-single-island-no-sync", "--tuning", "lora", "--lora-r", "16",
         "--rl-island-sample-role", "0:sampler", "--rl-island-sample-role", "1:sampler",
         "--rl-island-sample-role", "2:trainer", "--rl-max-policy-age", "2",
         "--rl-algorithm-spec", "examples/rl_algorithms/decoupled-ppo-c2.json",
         "--rl-checkpoint-store", STORE, "--cluster-prefix", "c1-spot", "--total-steps", "4")


def _args(extra=()):
    from yeto.launcher import _prepare_rl_args

    args = _launcher_args("ports", GROUP + tuple(extra), gpu=GPU3)
    args.controller = "local"
    _prepare_rl_args(args)
    return args


def _tokens(command):
    return shlex.split(command.split("-m yeto.rl.adapters.miles.island_entry", 1)[1])


def _flag(tokens, flag):
    return tokens[tokens.index(flag) + 1] if flag in tokens else None


def test_one_trainer_and_two_samplers_get_their_node_and_the_shards():
    from yeto.launcher import dry_run_plan

    plan = dry_run_plan(_args())
    roles = [i["sample_role"] for i in plan["island_requests"]]
    assert roles == ["sampler", "sampler", "trainer"] and plan["syncer"] is None
    tokens = [_tokens(i["learner_command"]) for i in plan["island_requests"]]
    assert [_flag(t, "--rl-sampler-node") for t in tokens] == ["island0", "island1", None]
    assert {_flag(t, "--rl-buffer-shards") for t in tokens} == {"island0,island1"}
    # the trainer trains --total-steps rounds; each of the two samplers generates half
    assert [_flag(t, "--global-rounds") for t in tokens] == ["2", "2", "4"]
    assert [_flag(t, "--total-fragment-steps") for t in tokens] == ["2", "2", "4"]


def test_the_samplers_split_the_rounds_evenly():
    with pytest.raises(ValueError, match="multiple of the sampler count"):
        _args(("--total-steps", "3"))


def test_the_island_resolves_its_sampler_id_and_the_shards():
    from yeto.launcher import dry_run_plan
    from yeto.rl.adapters.miles import island_entry

    plan = dry_run_plan(_args())
    sampler = island_entry.parse_args([{"$LEARNER_ID": "0"}.get(t, t) for t in
                                       _tokens(plan["island_requests"][1]["learner_command"])])
    env = {}
    island_entry.apply_sample_role(sampler, SimpleNamespace(), env)
    assert (env[br.SAMPLER_ID_ENV], env[br.SHARDS_ENV]) == ("island1", "island0,island1")


def test_sampler_and_resumable_trainer_islands_get_the_reclaim_marker():
    """S25 C1: the sampler islands; S26 E9.1: also the trainer, a resumable island on
    Modal (--rl-checkpoint-store), whose driver turns the marker into the preemption
    notice."""
    from yeto.gpu_spec import parse_gpu_spec
    from yeto.launch.modal_islands import build_modal_island_config
    from yeto.launcher import _island_task, make_miles_island_task

    args = _args()
    for m, spec in enumerate(parse_gpu_spec(args.gpu)):
        task = _island_task(make_miles_island_task, args, spec, m, 3, "none", {})
        cfg = build_modal_island_config(args, spec, m, task, "none")
        assert RECLAIM_MARKER_ENV in cfg.envs, m


def test_two_samplers_refuse_the_trainer_scheduler():
    with pytest.raises(ValueError, match="several sampler islands"):
        _args(("--rl-trainer-scheduler", "s1", "--rl-trainer-island", "2"))


def test_every_island_needs_a_role_and_one_trainer():
    from yeto.launcher import _prepare_rl_args

    two_trainers = tuple("1:trainer" if t == "1:sampler" else t for t in GROUP)
    args = _launcher_args("ports", two_trainers, gpu=GPU3)
    args.controller = "local"
    with pytest.raises(ValueError, match="exactly one trainer"):
        _prepare_rl_args(args)


def test_the_modal_runner_turns_the_signal_into_the_marker_on_a_sampler(monkeypatch):
    from yeto import modal_runner as mr

    seen = []
    monkeypatch.setattr(mr, "_run_forwarding_to_learner", lambda cmd, env, **kw: seen.append(env) or 0)

    class Sig:
        SIGINT, SIGTERM = _signal.SIGINT, _signal.SIGTERM

        @staticmethod
        def signal(num, fn):
            raise AssertionError("not expected")

    env = {RECLAIM_MARKER_ENV: "/tmp/yeto-reclaim-requested"}
    assert mr.run_island_script(["x"], env, popen=lambda c, env, **k: None, signal_mod=Sig) == 0
    assert seen == [env]


def test_the_reclaim_watcher_leaves_once_while_the_loop_is_still_in_a_round(tmp_path):
    import threading

    from yeto.rl.engine.sampler_resume import ReclaimWatcher, SamplerMembership

    store = CheckpointStore(tmp_path)
    events = []
    membership = SamplerMembership(store, "island0", "b", resume_point([], "island0"),
                                   lambda e, **f: events.append((e, f)), clock=lambda: 7.0)
    membership.join(latest_release={"seq": 3, "policy_version": 3})
    flag = threading.Event()
    watcher = ReclaimWatcher(flag.is_set, lambda: membership.leave("PREEMPTED", rounds=2,
                                                                   in_flight_rollout_id=4),
                             poll_s=0.01).start()
    flag.set()
    assert watcher.fired.wait(2.0)
    watcher.stop()
    assert membership.leave("FINISHED") is False  # the exit path after the watcher: no second LEAVE
    names = [e for e, _ in events]
    assert names == ["rl_sampler_join", "rl_sampler_leave"]
    leave = events[1][1]
    assert (leave["reason"], leave["rounds"], leave["in_flight_rollout_id"]) == ("PREEMPTED", 2, 4)
    record = (tmp_path / "sample-buffer/members/island0/b.json").read_text()
    assert '"leave_reason": "PREEMPTED"' in record and '"joined_at": 7.0' in record


def test_the_reclaim_signal_writes_the_marker_instead_of_raising(tmp_path):
    from yeto.rl.engine.sampler_resume import install_marker_signals

    handlers = {}

    class Sig:
        SIGINT, SIGTERM = _signal.SIGINT, _signal.SIGTERM

        @staticmethod
        def signal(num, fn):
            handlers[num] = fn

    env = {RECLAIM_MARKER_ENV: str(tmp_path / "marker")}
    assert install_marker_signals(env, signal_mod=Sig) == ["SIGINT", "SIGTERM"]
    assert not reclaim_requested(env)
    handlers[_signal.SIGINT](_signal.SIGINT, None)  # Modal's SIGINT reaches the island process
    assert reclaim_requested(env) and '"signal": 2' in (tmp_path / "marker").read_text()


def test_sampler_spot_bills_only_the_samplers_and_adds_no_spot_volume(monkeypatch):
    from yeto.cloud import s3_store
    from yeto.gpu_spec import parse_gpu_spec
    from yeto.launch.rl_island_task import make_miles_island_task
    from yeto.launcher import _island_task, _prepare_rl_args

    monkeypatch.setenv(s3_store.KEY_ID_ENV, "AKIDSTORE")
    monkeypatch.setenv(s3_store.SECRET_ENV, "store-secret")
    gpu = "nebius:1xh100@eu-north1,nebius:1xh100@eu-north1"
    base = ("--rl-single-island-no-sync", "--tuning", "lora", "--rl-island-sample-role", "0:sampler",
            "--rl-island-sample-role", "1:trainer", "--rl-max-policy-age", "2",
            "--rl-algorithm-spec", "examples/rl_algorithms/decoupled-ppo-c2.json",
            "--rl-checkpoint-store", "s3://yeto-ckpt/run-a", "--on-demand", "--rl-sampler-spot")
    args = _launcher_args("ports", base, gpu=gpu)
    args.controller = "local"
    _prepare_rl_args(args)
    from yeto.cloud import nebius_reclaim

    spots, volumes, hooks, markers = [], [], [], []
    for m, spec in enumerate(parse_gpu_spec(args.gpu)):
        task = _island_task(make_miles_island_task, args, spec, m, 2, "none", {})
        spots.append(task.resources.use_spot)
        volumes.append(sorted(getattr(task, "storage_mounts", None) or {}))
        hooks.append(nebius_reclaim.wants_host_hook(task.envs))
        markers.append(task.envs.get(RECLAIM_MARKER_ENV))
    # the sampler gets the store mount only, like the trainer: no legacy --spot volume
    assert spots == [True, False] and volumes[0] == volumes[1] and len(volumes[0]) == 1
    # Nebius stops the VM without signalling the container: the host unit writes the
    # sampler's marker (S25 C1b); the on-demand trainer gets no unit
    assert hooks == [True, False] and markers == [nebius_reclaim.NOTICE_FILE, None]


def test_an_on_demand_sampler_on_nebius_gets_no_host_unit():
    from yeto.launcher import _spot_sampler_cloud_env

    sampler = SimpleNamespace(rl_sample_role="sampler", rl_sampler_node="island0", spot=False)
    assert _spot_sampler_cloud_env(sampler, "nebius") == {}
    sampler.spot = True
    assert _spot_sampler_cloud_env(sampler, "modal") == {}  # Modal signals the container
    assert _spot_sampler_cloud_env(sampler, "nebius")


def test_the_reclaim_ack_is_written_only_where_a_host_hook_waits(tmp_path):
    from yeto.rl.engine.sampler_resume import RECLAIM_ACK_ENV, acknowledge_reclaim, clear_reclaim_files

    assert acknowledge_reclaim({}) is None
    env = {RECLAIM_MARKER_ENV: str(tmp_path / "notice"), RECLAIM_ACK_ENV: str(tmp_path / "ack")}
    assert acknowledge_reclaim(env) == tmp_path / "ack" and (tmp_path / "ack").exists()
    (tmp_path / "notice").write_text("1\n")
    clear_reclaim_files(env)  # a restarted VM keeps /tmp: the next incarnation starts clean
    assert not (tmp_path / "notice").exists() and not (tmp_path / "ack").exists()


def test_member_records_go_by_object_on_an_object_store(tmp_path):
    from yeto.rl.engine.object_store import DirectoryObjectStore

    class StagedStore(CheckpointStore):  # an s3/OSS store: its stage does not carry sample-buffer/
        def __init__(self, root, objects):
            super().__init__(root)
            self.objects = objects

    objects = DirectoryObjectStore(tmp_path / "bucket")
    store = StagedStore(tmp_path / "stage", objects)
    write_member(store, Member("island0", "a", 1.0, {"next_rollout_id": 0}, leave_reason="PREEMPTED"))
    assert '"leave_reason": "PREEMPTED"' in objects.read("sample-buffer/members/island0/a.json").decode()
    assert not (tmp_path / "stage" / "sample-buffer").exists()


def test_sampler_spot_needs_a_pair():
    from yeto.launcher import _prepare_rl_args

    args = _launcher_args("ports", ("--rl-single-island-no-sync", "--rl-sampler-spot"),
                          gpu="modal:1xh100")
    args.controller = "local"
    with pytest.raises(ValueError, match="--rl-sampler-spot needs --rl-island-sample-role"):
        _prepare_rl_args(args)


def test_a_failed_marker_write_in_the_signal_handler_does_not_raise(tmp_path, capsys):
    from yeto.rl.engine.sampler_resume import write_reclaim_marker

    env = {RECLAIM_MARKER_ENV: str(tmp_path / "missing-dir" / "marker")}
    write_reclaim_marker(env, _signal.SIGINT)  # T-C1 run b: the runner raced the same temp name
    assert "reclaim marker not written" in capsys.readouterr().out

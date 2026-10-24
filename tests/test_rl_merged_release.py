"""S25 C3: synced trainer islands feed sampler islands (yeto.rl.engine.merged_release).

Unit level: buffer lanes, the lane cursor log, the release guard, the publisher
wrappers, the role checks and the launch/island flags. The end-to-end case with
the real Rust syncer is in test_rl_synced_trainers_sampler.py."""

from __future__ import annotations

import json
import shlex
from types import SimpleNamespace

import pytest
from test_rl_algorithm_provenance import _fake_sky, _no_modal_listing  # noqa: F401

from yeto.rl.adapters.miles import buffer_rollout as br
from yeto.rl.adapters.miles.sampler_island import GateVersionPublisher, ReleasingPublisher
from yeto.rl.engine import sample_roles
from yeto.rl.engine.merged_release import (
    BufferLane,
    LaneCursorLog,
    MergedReleaseGuard,
    ObjectLaneCursorLog,
    gate_version,
    is_publisher,
    lane_cursor_log,
    merged_version,
    merges_shards,
)
from yeto.rl.engine.policy_release import PolicyReleaseStore
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer, SampleBufferError, open_trainer_buffer
from yeto.rl.engine.trainer_lifecycle import SampleIndexEntry


def _fill(buffer: FileSampleBuffer, n: int, version: int = 0) -> None:
    for i in range(len(buffer), len(buffer) + n):
        ref = buffer.write_payload(f"s{i}", br.encode_group([{"tokens": [i]}]))
        buffer.append(SampleIndexEntry(f"s{i}", version, ref))


# ---------------------------------------------------------------- lanes
def test_lane_parse_and_ownership():
    assert BufferLane.parse(None) is None and BufferLane.parse("0/1") == BufferLane(0, 1)
    lane = BufferLane.parse("1/3")
    assert lane == BufferLane(1, 3) and lane.env() == "1/3"
    assert [p for p in range(9) if lane.owns(p)] == [1, 4, 7]
    with pytest.raises(ValueError):
        BufferLane(2, 2)
    with pytest.raises(ValueError):
        BufferLane.parse("x/2")


def test_two_lane_readers_split_one_buffer_without_overlap(tmp_path):
    writer = FileSampleBuffer(CheckpointStore(tmp_path), batch_size=1)
    _fill(writer, 7)
    readers = [FileSampleBuffer(CheckpointStore(tmp_path), batch_size=1, lane=BufferLane(m, 2))
               for m in range(2)]
    taken = [[s.position for s in r.select("trainer", 10)] for r in readers]
    assert taken == [[0, 2, 4, 6], [1, 3, 5]]
    assert readers[0].read_cursor == readers[1].read_cursor == 7  # foreign positions are passed
    # a limited select stops at the limit and resumes after the last lane position
    again = FileSampleBuffer(CheckpointStore(tmp_path), batch_size=1, lane=BufferLane(0, 2))
    assert [s.position for s in again.select("trainer", 2)] == [0, 2]
    assert [s.position for s in again.select("trainer", 2)] == [4, 6]


def test_only_the_publishers_lane_merges_shards(tmp_path):
    assert merges_shards(None) and merges_shards(BufferLane(0, 2)) and not merges_shards(BufferLane(1, 2))
    with pytest.raises(SampleBufferError, match="only the publisher's lane merges"):
        FileSampleBuffer(CheckpointStore(tmp_path), batch_size=1, lane=BufferLane(1, 2),
                         merge_shards=True)


# ---------------------------------------------------------------- object stores (S3, OSS)
def _s3_store(tmp_path, s3):
    from yeto.cloud import s3_store
    from yeto.rl.engine.resume import STORE_IMPL_ENV, store_for

    env = {STORE_IMPL_ENV: s3_store.STORE_IMPL, s3_store.URI_ENV: "s3://bkt/runs/a"}
    return store_for(tmp_path / "mount", environ=env, client=s3)


def test_lane_readers_and_cursors_on_an_s3_store_use_the_object_api(tmp_path):
    from test_rl_object_store import FakeS3

    s3 = FakeS3()
    _fill(FileSampleBuffer(_s3_store(tmp_path, s3), batch_size=1), 6)

    def reader(m):
        buffer = open_trainer_buffer(_s3_store(tmp_path, s3), batch_size=1, lane=BufferLane(m, 2))
        return br.BatchReader(buffer, max_policy_age=2, cursor_log=lane_cursor_log(buffer))

    first = [reader(m) for m in range(2)]
    assert all(isinstance(r.cursor_log, ObjectLaneCursorLog) for r in first)
    taken = [[p for p, _ in r.take(0, 2, current_version=0)[0]] for r in first]
    assert taken == [[0, 2], [1, 3]]
    keys = sorted(k for _, k in s3.objects if "/lanes/" in k)
    assert keys == ["runs/a/sample-buffer/lanes/0-of-2/0000000000.json",
                    "runs/a/sample-buffer/lanes/1-of-2/0000000000.json"]
    assert not (tmp_path / "mount").exists()  # nothing through the mount
    # a restarted lane reader continues after its last round (at most once)
    again = reader(1)
    assert again.buffer.read_cursor == 4
    assert [p for p, _ in again.take(1, 1, current_version=0)[0]] == [5]
    assert again.cursor_log.last_cursor() == 6
    assert len([k for _, k in s3.objects if "/lanes/1-of-2/" in k]) == 2


def test_on_an_oss_store_the_publisher_merges_and_the_other_lane_reads_the_merged_index():
    from test_rl_oss_store import _entry, _objects

    from yeto.rl.engine.sample_buffer import shard_index

    objects = _objects()
    sampler = FileSampleBuffer(objects, batch_size=1, index=shard_index("s0"))
    for i in range(4):
        ref = sampler.write_payload(f"s{i}", br.encode_group([{"tokens": [i]}]))
        sampler.append(SampleIndexEntry(f"s{i}", 0, ref, appended_at=float(i)))
    other = open_trainer_buffer(objects, batch_size=1, lane=BufferLane(1, 2))
    assert other.select("trainer", 10) == []  # nothing merged yet: lane 1 does not merge
    publisher = open_trainer_buffer(objects, batch_size=1, lane=BufferLane(0, 2))
    assert [s.position for s in publisher.select("trainer", 10)] == [0, 2]
    assert [s.entry.sample_id for s in other.select("trainer", 10)] == ["s1", "s3"]
    log = lane_cursor_log(other)
    log.append(0, [1, 3], other.read_cursor)
    assert lane_cursor_log(open_trainer_buffer(objects, batch_size=1, lane=BufferLane(1, 2))).last_cursor() == 4


def test_reader_continues_after_its_lane_cursor_and_never_retrains(tmp_path):
    store = CheckpointStore(tmp_path)
    writer = FileSampleBuffer(store, batch_size=1)
    _fill(writer, 8)
    lane = BufferLane(1, 2)

    def reader():
        buffer = FileSampleBuffer(CheckpointStore(tmp_path), batch_size=1, lane=lane)
        return br.BatchReader(buffer, max_policy_age=2, cursor_log=br.open_lane_log(buffer))

    first = reader()
    groups, report = first.take(0, 2, current_version=0)
    assert [p for p, _ in groups] == [1, 3] and report.end == 4
    # restart (a re-joined island): the next round continues after position 3
    second = reader()
    assert second.buffer.read_cursor == 4
    groups, _ = second.take(1, 2, current_version=1)
    assert [p for p, _ in groups] == [5, 7]
    lines = [json.loads(line) for line in
             (tmp_path / "sample-buffer" / "lanes" / "1-of-2.jsonl").read_text().splitlines()]
    assert [(r["rollout_id"], r["positions"], r["cursor"]) for r in lines] == \
        [(0, [1, 3], 4), (1, [5, 7], 8)]


def test_lane_cursor_log_skips_a_torn_last_line(tmp_path):
    log = LaneCursorLog(tmp_path, BufferLane(0, 2))
    log.append(0, [0, 2], 3)
    with log.path.open("a") as fh:
        fh.write('{"cursor": 9')
    assert log.last_cursor() == 3


def test_gate_uses_the_merged_version_once_set(tmp_path):
    writer = FileSampleBuffer(CheckpointStore(tmp_path), batch_size=1)
    _fill(writer, 2, version=1)
    _fill(writer, 2, version=4)
    reader = br.BatchReader(FileSampleBuffer(CheckpointStore(tmp_path), batch_size=1),
                            max_policy_age=1, wait_s=0.0)
    reader.gate_version = 5  # merged version 5: version 1 is 4 behind -> passed over
    groups, report = reader.take(0, 2, current_version=1)
    assert [p for p, _ in groups] == [2, 3] and report.stale_groups == 2
    assert gate_version(3, None) == 3 and gate_version(3, 7) == 7


def test_executor_gate_version_setter_reaches_the_active_reader(tmp_path, monkeypatch):
    monkeypatch.setattr(br, "_ACTIVE", {})
    assert br.executor_set_gate_version(None, 3) is None  # no reader yet
    reader = br.BatchReader(FileSampleBuffer(CheckpointStore(tmp_path), batch_size=1),
                            max_policy_age=1)
    br.register_reader(reader)
    assert br.executor_set_gate_version(None, 3) is None
    assert br.executor_set_gate_version(None, 4) == 3 and reader.gate_version == 4


def test_island_env_carries_the_lane_only_for_a_trainer():
    env = br.island_env("trainer", store_root="/s", max_policy_age=2, lane=BufferLane(1, 2))
    assert env[br.LANE_ENV] == "1/2" and br.LANE_ENV in br.ISLAND_ENVS
    with pytest.raises(ValueError):
        br.island_env("sampler", store_root="/s", sampler_id="a", lane=BufferLane(1, 2))


# ---------------------------------------------------------------- versions and the publisher
def test_publisher_and_merged_version_rules():
    assert is_publisher(0, 2) and not is_publisher(1, 2)
    with pytest.raises(ValueError):
        is_publisher(2, 2)
    state = SimpleNamespace(policy_version=4)
    assert merged_version(SimpleNamespace(outer_version=None), state) == 4
    assert merged_version(SimpleNamespace(outer_version=6), state) == 6
    assert merged_version(SimpleNamespace(), state) == 4


def test_guard_releases_increasing_versions_and_reports_gaps():
    gaps = []
    guard = MergedReleaseGuard(lambda a, b: gaps.append((a, b)))
    assert guard.check(0, None) == "release"
    assert guard.check(1, 0) == "release"
    assert guard.check(1, 1) == "stale" and guard.check(0, 1) == "stale"
    assert guard.check(4, 1) == "release" and gaps == [(1, 4)]


class _State:
    def __init__(self, version: int, digest: str) -> None:
        self.policy_version = version
        self._digest = digest

    def policy_tensor_hash(self) -> str:
        return self._digest


class _Inner:
    def __init__(self) -> None:
        self.published: list[int] = []

    def publish(self, state):
        self.published.append(state.policy_version)
        return "ok"


def test_releasing_publisher_releases_merged_versions_in_order(tmp_path):
    events: list[tuple[str, dict]] = []
    releases = PolicyReleaseStore(CheckpointStore(tmp_path))
    driver = SimpleNamespace(outer_version=None)
    gates: list[int] = []
    inner = GateVersionPublisher(_Inner(), gates.append, lambda s: merged_version(driver, s))
    publisher = ReleasingPublisher(
        inner, releases, lambda v, d: (d / "w.bin").write_bytes(b"%d" % v), lambda: 3,
        lambda event, **f: events.append((event, f)),
        version_of=lambda s: merged_version(driver, s),
        guard=MergedReleaseGuard(lambda a, b: events.append(("gap", {"a": a, "b": b}))),
        origin=lambda s: {"kind": "merged", "learner_id": 0, "num_learners": 2,
                          "local_version": s.policy_version})
    for local, outer, digest in ((0, 0, "h0"), (1, 1, "h1"), (2, 1, "h1"), (3, 3, "h3")):
        driver.outer_version = outer
        assert publisher.publish(_State(local, digest)) == "ok"
    assert gates == [0, 1, 1, 3]
    assert [e for e, _ in events] == ["rl_policy_released", "rl_policy_released",
                                      "rl_policy_release_skipped", "gap", "rl_policy_released"]
    latest = releases.latest()
    assert (latest.seq, latest.version, latest.policy_hash) == (3, 3, "h3")
    assert latest.origin == {"kind": "merged", "learner_id": 0, "num_learners": 2,
                             "local_version": 3}
    assert latest.buffer_selected == 3


# ---------------------------------------------------------------- role checks
def _role_args(**kw):
    base = dict(rl_sample_role="trainer", rl_single_island_no_sync=False,
                rl_checkpoint_store="modal-volume://v/p", rl_max_policy_age=2, tuning="lora")
    return SimpleNamespace(**{**base, **kw})


def test_a_trainer_may_sync_a_sampler_may_not():
    sample_roles.check_launch(_role_args())
    sample_roles.check_synced_trainers(_role_args(), 2)
    with pytest.raises(ValueError, match="only trainers may sync"):
        sample_roles.check_launch(_role_args(rl_sample_role="sampler"))
    with pytest.raises(ValueError, match="placement reconfiguration"):
        sample_roles.check_launch(_role_args(rl_elastic=True))


@pytest.mark.parametrize("kw,islands,message", [
    ({}, 1, "at least two islands"),
    ({"tuning": "full"}, 2, "--tuning lora"),
])
def test_synced_trainer_refusals(kw, islands, message):
    with pytest.raises(ValueError, match=message):
        sample_roles.check_synced_trainers(_role_args(**kw), islands)


@pytest.mark.parametrize("store", ["s3://b/p", "oss://b/p", "modal-volume://v/p"])
def test_synced_trainers_take_an_object_store(store):
    sample_roles.check_synced_trainers(_role_args(rl_checkpoint_store=store), 2)


def test_buffer_lane_of_island_args():
    args = _role_args(learner_id=1, rl_sample_lanes=2)
    assert sample_roles.buffer_lane(args) == BufferLane(1, 2)
    assert sample_roles.buffer_lane(_role_args(learner_id=0, rl_single_island_no_sync=True,
                                               rl_sample_lanes=2)) is None
    resolved = sample_roles.resolve(_role_args(rl_sample_store="/root/store"), groups_per_round=4)
    assert resolved.store == "/root/store" and resolved.role == "trainer"


# ---------------------------------------------------------------- launch and island flags
SYNCED = ("--rl-sample-role", "trainer", "--rl-max-policy-age", "2",
          "--rl-checkpoint-store", "modal-volume://yeto-c3/run-a", "--rl-sample-wait-s", "900",
          "--rl-sync-quorum-theta", "0.75", "--rl-sync-max-base-lag", "2", "--cluster-prefix", "c3-synced",
          "--rl-algorithm-spec", "examples/rl_algorithms/decoupled-ppo-c2.json")


@pytest.fixture(autouse=True)
def _island_hmac_key(monkeypatch):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "c3-test-key")


def _synced_plan(extra=()):
    from test_rl_sampler_trainer_pair_launch import _args

    from yeto.launcher import dry_run_plan

    args = _args(tuple(extra), base=SYNCED)
    return args, dry_run_plan(args)


def _island_tokens(command: str, learner: str) -> list[str]:
    tail = command.split("-m yeto.rl.adapters.miles.island_entry", 1)[1]
    return [{"$LEARNER_ID": learner}.get(t, t) for t in shlex.split(tail)]


def test_synced_trainer_plan_forwards_the_sample_store_and_lanes():
    from yeto.rl.adapters.miles import island_entry

    _, plan = _synced_plan()
    assert plan["syncer"] is not None
    islands = plan["island_requests"]
    assert len(islands) == 2
    for m, island in enumerate(islands):
        tokens = _island_tokens(island["learner_command"], str(m))
        assert tokens[tokens.index("--rl-sample-role") + 1] == "trainer"
        assert tokens[tokens.index("--rl-sample-lanes") + 1] == "2"
        assert "--rl-sample-store" in tokens and "--rl-resume-store" not in tokens
        parsed = island_entry.parse_args(tokens)
        assert sample_roles.buffer_lane(parsed) == BufferLane(m, 2)


def test_synced_trainer_launch_writes_island_round_cuts():
    """S27 R-2: a cut interval gives every synced trainer, in a fixed roster or an
    island pool, its own write-only cuts."""
    from types import SimpleNamespace as NS

    from yeto.rl.adapters.miles import island_entry

    _, plan = _synced_plan(("--rl-cut-every", "2"))
    for m, island in enumerate(plan["island_requests"]):
        tokens = _island_tokens(island["learner_command"], str(m))
        root = tokens[tokens.index("--rl-island-cut-store") + 1]
        assert root.endswith("/island-cuts") and "--rl-resume-store" not in tokens
        miles_args = NS()
        island_entry.apply_ports_infra_switches(island_entry.parse_args(tokens), miles_args, {})
        assert miles_args.yeto_rl_resume["store"] == f"{root}/island{m}"
        assert miles_args.yeto_rl_resume["write_only"] is True
    with pytest.raises(ValueError, match="does not resume from round cuts"):
        _synced_plan(("--rl-cut-every", "1", "--rl-stop-after-rounds", "2"))


def test_island_refuses_a_sample_store_outside_a_synced_trainer():
    from yeto.rl.adapters.miles import island_entry

    with pytest.raises(ValueError, match="synced trainer island needs"):
        island_entry._check_sample_store(SimpleNamespace(
            rl_sample_role="trainer", rl_single_island_no_sync=False, rl_sample_store=None,
            rl_sample_lanes=None, rl_resume_store=None))
    with pytest.raises(ValueError, match="need --rl-sample-role trainer"):
        island_entry._check_sample_store(SimpleNamespace(
            rl_sample_role="sampler", rl_single_island_no_sync=True, rl_sample_store="/s",
            rl_sample_lanes=2, rl_resume_store=None))

"""A shard reused after a pool fence or an end never makes a sampler finish silently.

The chain this file replays (a Modal pool, 2026-11-24): the pool took a starting node for
lost and fenced its shard (a stop request naming ``pool-fence:<ticket>``), the operator
moved the stopped pool's state away, and the new pool gave the same shard name to a new
sampler. That sampler obeyed the stop request no one would ever take over, ended the
shard, and exited 0; every later incarnation joined and finished at once with 0 rounds,
and the pool took it for a normal finish.

Now (CPU, a fake S3 bucket and local stores, no cloud resource):

* the sampler judges a stop request at its join: a fence its own lease claim superseded,
  a successor that is its own shard or that never joined is stale - set aside, the shard
  taken over - and any other request is obeyed with ``rl_sampler_not_run`` (reason,
  successor) and exit code 3, as is a shard whose data already ended;
* the pool reads exit 3 as ``sampler_not_run`` (WARNING, event), never as the end of the
  data, and gives a new member the next shard whose data did not end in the store;
* ``rl_sampler_start.store`` is the store URI, not the node's staging directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_nc_elastic import Clock, Cloud, Objects, _names, _pool, _run  # noqa: F401
from test_rl_object_store import FakeS3
from tests.test_stage1_gaps import _sampler_spec
from yeto.cloud import s3_store
from yeto.launch.nc import elastic
from yeto.launch.nc.modal_provider import NOT_RUN, RUNNING, ModalPoolOps, ModalSdkClient
from yeto.launch.pool import status as el
from yeto.launch.pool.data_path import R2DataPath
from yeto.launch.pool.island_pool import GONE, READY
from yeto.rl.engine import engine_sampler_entry as entry
from yeto.rl.engine.release_delta import MIRROR_ENV
from yeto.rl.engine.resume import STORE_IMPL_ENV, CheckpointStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer, shard_index
from yeto.rl.engine.sampler_resume import (
    Member,
    handoff_requested,
    judge_handoff,
    read_handoff,
    request_handoff,
    set_aside_handoff,
    write_member,
)
from yeto.rl.engine.trainer_lease import TrainerLease

BUCKET, RUN = "b", "runs/fleet-r/trainer1"
URI = f"s3://{BUCKET}/{RUN}"
_BUCKETS: dict[str, FakeS3] = {}
BUCKET_ENV = "YETO_TEST_B3_BUCKET"


def fake_s3_store(path, *, environ, **_kwargs):
    """``YETO_RL_STORE_IMPL`` of these tests: the real S3 store hook over a fake bucket."""
    return s3_store.store_from_env(path, environ=environ, client=_BUCKETS[environ[BUCKET_ENV]])


@pytest.fixture
def bucket(tmp_path, monkeypatch, request):
    """One fake bucket; samplers stage it under ``tmp/stage`` like a Modal island."""
    client = _BUCKETS[request.node.name] = FakeS3()
    monkeypatch.setenv(BUCKET_ENV, request.node.name)
    monkeypatch.setenv(STORE_IMPL_ENV, f"{__name__}:fake_s3_store")
    monkeypatch.setenv(s3_store.URI_ENV, URI)
    monkeypatch.setenv(s3_store.STAGE_ENV, str(tmp_path / "stage"))
    monkeypatch.setenv(MIRROR_ENV, str(tmp_path / "mirror"))
    objects = s3_store.head_objects(URI, environ={}, client=client)
    TrainerLease.claim(objects, incarnation="trainer-1")  # the trainer reads ownership records
    yield elastic.ObjectRecords(objects)  # the run store as the pool sees it (s3_fence_store)
    _BUCKETS.pop(request.node.name, None)


def _spec(tmp_path: Path, shard: str, rounds: int = 2, **over) -> dict:
    spec = _sampler_spec(tmp_path, sampler_id=shard, num_rollouts=rounds, buffer_shards=["nc-0", "nc-1", "nc-2"],
                         store=str(tmp_path / "stage" / BUCKET / RUN), **over)
    spec["event_tape"] = str(tmp_path / f"tape-{shard}" / "events.jsonl")
    return spec


def _main(tmp_path: Path, spec: dict) -> int:
    path = tmp_path / f"spec-{spec['sampler_id']}.json"
    path.write_text(json.dumps(spec))
    return entry.main(["--spec", str(path)])


def _events(tmp_path: Path, shard: str) -> list[dict]:
    out = []
    for tape in sorted((tmp_path / f"tape-{shard}").glob("events.*.jsonl"), key=lambda p: p.stat().st_mtime):
        out += [json.loads(line) for line in tape.read_text().splitlines()]
    return out


def _of(events, name):
    return [e for e in events if e["event"] == name]


class StoreCloud(Cloud):
    """A pool over a run store the pool can read, like the SSH and Modal ops."""

    def __init__(self, clock, store, *, set_aside=True):
        super().__init__(clock)
        self.store = store
        if set_aside:
            self.clear_stale_handoff = lambda m, keep: R2DataPath(self.store).clear_stale_handoff(m.shard, keep)

    def exit_evidence(self, m):
        return R2DataPath(self.store).exit_evidence(m.shard)

    def shard_ended(self, shard):
        return R2DataPath(self.store).shard_ended(shard)


def _lose_before_join(pool, clock, cloud):
    """The pool's first node is lost while its sampler starts: the pool fences its shard."""
    _run(pool, clock, clock.t + 180)
    m = pool.members[0]
    cloud.tickets[m.ticket].update(state="ended", reason="lost")
    _run(pool, clock, clock.t + 120)
    assert m.state == GONE and m.end_reason == "lost"
    return m


# ------------------------------------------------------------------ the whole chain (B3)

@pytest.mark.parametrize("pool_sets_aside", [False, True])
def test_a_fence_left_by_a_pool_whose_state_was_moved_away_never_stops_the_new_sampler(
        tmp_path, bucket, pool_sets_aside):
    clock = Clock()
    # attempt 1: the node of nc-0 is lost while it starts; the pool fences nc-0
    first = _pool(clock, StoreCloud(clock, bucket), accounts=(1,), max_orders=1)
    lost = _lose_before_join(first, clock, first.ops)
    assert read_handoff(bucket, "nc-0")["successor"] == f"pool-fence:{lost.ticket}"
    first.stop("signal SIGTERM")
    # the operator moves the stopped state away: a new pool numbers its shards from nc-0 again
    cloud = StoreCloud(clock, bucket, set_aside=pool_sets_aside)
    second = _pool(clock, cloud, accounts=(1,))
    _run(second, clock, clock.t + 180)
    m = second.members[0]
    assert m.shard == "nc-0" and m.started is not None
    assert handoff_requested(bucket, "nc-0") is not pool_sets_aside
    # its sampler: the fence is older than its own lease claim, so it is stale; it runs
    code = _main(tmp_path, _spec(tmp_path, "nc-0"))
    events = _events(tmp_path, "nc-0")
    (leave,) = _of(events, "rl_sampler_leave")
    assert code == 0 and leave["reason"] == "FINISHED" and leave["rounds"] == 2
    assert not _of(events, "rl_sampler_not_run")
    if not pool_sets_aside:
        (stale,) = _of(events, "rl_sampler_handoff_stale")
        assert stale["successor"] == f"pool-fence:{lost.ticket}" and stale["why"] == "fence_before_this_claim"
        assert stale["set_aside"] is True
    assert not handoff_requested(bucket, "nc-0")
    shard = FileSampleBuffer(bucket.objects, batch_size=1, index=shard_index("nc-0"), create_index=False)
    assert len(shard) == 4 and shard.data_ended  # its own two rounds, then the end of its data
    assert shard.end_record.get("why") is None  # a normal end, not a handoff


def test_a_shard_an_earlier_attempt_ended_is_skipped_and_a_sampler_there_says_why(tmp_path, bucket):
    clock = Clock()
    # what the code before this fix left: a sampler obeyed the fence's stop request and ended nc-0
    writer = FileSampleBuffer(bucket.objects, batch_size=1, index=shard_index("nc-0"))
    writer.claim_owner("7d9a2adb", by="sampler")
    writer.mark_data_end()
    assert R2DataPath(bucket).shard_ended("nc-0") == {"ended_by": "7d9a2adb", "end_why": None, "successor": None}
    assert R2DataPath(bucket).shard_ended("nc-1") is None
    # a sampler started there anyway (old pool code, an operator) does not finish silently
    code = _main(tmp_path, _spec(tmp_path, "nc-0"))
    events = _events(tmp_path, "nc-0")
    (not_run,) = _of(events, "rl_sampler_not_run")
    assert code == entry.NOT_RUN_EXIT_CODE == 3
    assert not_run["reason"] == "shard_ended" and not_run["ended_by"] == "7d9a2adb"
    (leave,) = _of(events, "rl_sampler_leave")
    assert leave["rounds"] == 0 and leave["not_run"] == "shard_ended"
    # the pool after the state was moved away: nc-0 is skipped, nc-1 is ordered
    cloud = StoreCloud(clock, bucket)
    pool = _pool(clock, cloud, accounts=(1,))
    _run(pool, clock, 1800)
    (m,) = pool.members
    assert m.shard == "nc-1" and m.state == READY
    (skip,) = [e for e in pool.events.events if e["event"].endswith("shard_skipped")]
    assert skip["shard"] == "nc-0" and skip["ended_by"] == "7d9a2adb"


def test_the_pool_reports_a_sampler_that_did_not_run_and_continues_on_a_fresh_shard(tmp_path, bucket):
    clock = Clock()
    cloud = StoreCloud(clock, bucket)
    pool = _pool(clock, cloud, accounts=(1, 4))
    _run(pool, clock, 1800)
    (m,) = pool.members
    assert m.state == READY and m.shard == "nc-0"
    # a stop request for a successor that joined (an earlier attempt's replacement) is obeyed
    write_member(bucket, Member(sampler="nc-9", incarnation="old", joined_at=1.0, resume={}, left_at=2.0,
                                leave_reason="FINISHED"))
    request_handoff(bucket, "nc-0", "nc-9")
    code = _main(tmp_path, _spec(tmp_path, "nc-0"))
    assert code == entry.NOT_RUN_EXIT_CODE
    (not_run,) = _of(_events(tmp_path, "nc-0"), "rl_sampler_not_run")
    assert (not_run["reason"], not_run["successor"], not_run["why"]) == ("shard_handed_off", "nc-9",
                                                                       "successor_joined")
    end = FileSampleBuffer(bucket.objects, batch_size=1, index=shard_index("nc-0"), create_index=False).end_record
    assert end["why"]["reason"] == "shard_handed_off" and end["why"]["successor"] == "nc-9"
    cloud.tickets[m.ticket].update(sampler="exited", exit=code)
    _run(pool, clock, clock.t + 1800)
    names = _names(pool)
    assert "sampler_not_run" in names and "sampler_finished" not in names
    assert not pool.finished and not pool.stopped
    (event,) = [e for e in pool.events.events if e["event"].endswith("sampler_not_run")]
    assert (event["reason"], event["successor"], event["shard"]) == ("shard_handed_off", "nc-9", "nc-0")
    assert m.state == GONE
    nxt = [x for x in pool.members if x.state != GONE]
    assert [x.shard for x in nxt] == ["nc-1"] and nxt[0].replaces_shard == "nc-0"  # it continues nc-0's prompts


# ------------------------------------------------------------------ sampler-side rules

def _store(tmp_path):
    return CheckpointStore(tmp_path / "store")


def test_judge_handoff_follows_the_lease_and_the_member_records(tmp_path):
    store = _store(tmp_path)

    def judge(successor, superseded=()):
        return judge_handoff(store, "nc-0", {"successor": successor, "requested_at": 1.0}, superseded=superseded)

    assert (judge("nc-0").obey, judge("nc-0").why) == (False, "successor_is_this_shard")
    assert (judge("pool-fence:t1", ["pool-fence:t1"]).obey, judge("pool-fence:t1", ["x"]).obey) == (False, True)
    assert judge("pool-fence:t1", None).why == "pool_fence_no_lease" and judge("pool-fence:t1", None).obey
    assert (judge("nc-2").obey, judge("nc-2").successor_state) == (False, "never_joined")
    write_member(store, Member(sampler="nc-2", incarnation="i", joined_at=1.0, resume={}))
    assert (judge("nc-2").obey, judge("nc-2").successor_state) == (True, "running")
    write_member(store, Member(sampler="nc-2", incarnation="i", joined_at=1.0, resume={}, left_at=2.0))
    assert (judge("nc-2").obey, judge("nc-2").successor_state) == (True, "left")
    assert judge_handoff(store, "nc-0", {"successor": None, "torn": True}, superseded=[]).obey


def test_a_new_stop_request_is_never_set_aside_for_an_old_one(tmp_path):
    store = _store(tmp_path)
    request_handoff(store, "nc-0", "pool-fence:t1", clock=lambda: 1.0)
    old = read_handoff(store, "nc-0")
    request_handoff(store, "nc-0", "pool-fence:t2", clock=lambda: 2.0)  # a new fence meanwhile
    assert set_aside_handoff(store, "nc-0", only=old) is None and read_handoff(store, "nc-0")["successor"] == \
        "pool-fence:t2"
    assert set_aside_handoff(store, "nc-0", only=read_handoff(store, "nc-0"))["successor"] == "pool-fence:t2"


def test_a_fence_written_after_the_join_is_obeyed(tmp_path, bucket):
    """The pool fences a running sampler (its node taken for lost): the request is new
    (not superseded by the sampler's claim) and the ownership record fences it."""
    spec = _spec(tmp_path, "nc-0", rounds=4, reward=f"{__name__}:fencing_reward")
    FENCE["store"], FENCE["n"] = bucket, 0
    rounds = entry.run(spec, resource_sampler=lambda emit: None)
    events = _events(tmp_path, "nc-0")
    assert rounds < 4 and not _of(events, "rl_sampler_handoff_stale") and not _of(events, "rl_sampler_not_run")
    assert _of(events, "rl_sampler_leave")[-1]["reason"] == "FENCED"
    assert handoff_requested(bucket, "nc-0")  # the pool's request stands
    assert not FileSampleBuffer(bucket.objects, batch_size=1, index=shard_index("nc-0"),
                                create_index=False).data_ended  # a fenced incarnation cannot end the shard


FENCE: dict = {}


def fencing_reward(item, completion) -> float:
    """At the third scored completion the pool fences the running sampler's shard."""
    FENCE["n"] += 1
    if FENCE["n"] == 3:
        elastic.fence_shard(FENCE["store"], "nc-0", by="t-9")
    return 0.5


def test_a_running_sampler_stopped_for_its_successor_says_so_and_exits_zero(tmp_path):
    store = _store(tmp_path)
    write_member(store, Member(sampler="nc-1", incarnation="s", joined_at=1.0, resume={}))
    HANDOFF["store"], HANDOFF["n"] = store, 0
    spec = _sampler_spec(tmp_path, sampler_id="nc-0", num_rollouts=4, buffer_shards=["nc-0", "nc-1"],
                         reward=f"{__name__}:handoff_reward")
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    assert entry.main(["--spec", str(path)]) == 0
    [tape] = (tmp_path / "tape").glob("events.*.jsonl")
    events = [json.loads(line) for line in tape.read_text().splitlines()]
    (handed,) = _of(events, "rl_sampler_handed_off")
    assert handed["successor"] == "nc-1" and handed["reason"] == "shard_handed_off" and handed["rounds"] == 1
    assert not _of(events, "rl_sampler_not_run")


HANDOFF: dict = {}


def handoff_reward(item, completion) -> float:
    """At the first scored completion the successor nc-1 asks nc-0 to stop."""
    HANDOFF["n"] += 1
    if HANDOFF["n"] == 1:
        request_handoff(HANDOFF["store"], "nc-0", "nc-1")
    return 0.5


# ------------------------------------------------------------------ exit status, Modal

def test_the_not_run_exit_is_its_own_verdict():
    cfg = el.ElasticConfig()
    active = el.LeaseStatus("active", lease_until=10_000)
    for joined in (True, False):
        verdict = el.classify(active, el.Probe(node="ok", sampler="exited", exit_code=entry.NOT_RUN_EXIT_CODE),
                              now=0, unreachable_since=None, cfg=cfg, joined=joined)
        assert verdict.kind == el.NOT_RUN
    assert entry.NOT_RUN_EXIT_CODE in el.NOT_RUN_EXIT_CODES
    assert entry.NOT_RUN_EXIT_CODE not in el.CONFIG_EXIT_CODES | {0}


def test_a_modal_call_that_did_not_run_is_not_lost_and_probes_as_not_run(tmp_path, monkeypatch):
    monkeypatch.setenv("YETO_NC_STATE_DIR", str(tmp_path / "state"))

    class Call:
        def __init__(self, rc):
            self.rc = rc

        def get(self, timeout=0):
            return {"rc": self.rc}

    calls = {"a": Call(3), "b": Call(0), "c": Call(1)}
    modal = SimpleNamespace(FunctionCall=SimpleNamespace(from_id=lambda call: calls[call]))
    client = ModalSdkClient({}, run_id="r", modal=modal)
    assert [client.status(c) for c in "abc"] == [NOT_RUN, "SUCCEEDED", "FAILED"]
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"event_tape": "/work/sampler/events.jsonl", "prompts": {"groups_per_round": 4},
                                "store": "__RUN_PREFIX__"}))
    env = tmp_path / "r2.env"
    env.write_text("R2_BUCKET=b\n")
    env.chmod(0o600)
    cfg = {"target": 1, "accounts": [1], "shard_base": "m", "run": "run-m", "spec": str(spec),
           "run_prefix": "runs-m", "r2_env_file": str(env),
           "provider": {"kind": "modal", "image": "registry.example/sampler@sha256:00", "secret": "s",
                        "bucket": "b", "gpu": "B300", "gpus": 8, "price_usd_h": 50.0}}
    states = {}

    class Client:
        def spawn(self, payload):
            states["fc-1"] = RUNNING
            return "fc-1"

        def status(self, call):
            return states[call]

    from yeto.launch.nc.modal_provider import from_config

    ops = ModalPoolOps(cfg, provider=from_config(cfg, "run-m", client=Client()),
                       fence_store=SimpleNamespace(objects=Objects()))
    ticket = ops.order(1)
    m = SimpleNamespace(ticket=ticket, account=1, shard="m-0", data_slot=None, replaces_shard=None,
                        predecessor_lost=False, predecessor_retired=False)
    payload = ops.payload(m, ["m-0"])
    assert payload["spec"]["store_uri"] == "s3://b/runs/runs-m" and payload["spec"]["store"].startswith("/tmp/stage")
    ops.start(m, ["m-0"])
    states["fc-1"] = NOT_RUN
    assert ops.poll(m).state == "active"
    assert ops.probe(m) == el.Probe(node="ok", sampler="exited", exit_code=3)
    # the Modal ops read the run store like the SSH ops do
    assert ops.exit_evidence(m)["handoff"] is None and ops.shard_ended("m-0") is None
    request_handoff(ops._fence_store(), "m-0", "pool-fence:x")  # noqa: SLF001
    assert ops.clear_stale_handoff(m, set())["successor"] == "pool-fence:x"


# ------------------------------------------------------------------ rl_sampler_start.store

def test_the_start_event_store_is_the_store_uri_not_the_staging_directory(tmp_path, bucket):
    spec = _spec(tmp_path, "nc-0", rounds=1)
    entry.run(spec, resource_sampler=lambda emit: None)
    (start,) = _of(_events(tmp_path, "nc-0"), "rl_sampler_start")
    assert start["store"] == URI and start["local_store"] == spec["store"]
    # a spec that names the URI (the Modal pool's payload) wins
    spec = _spec(tmp_path, "nc-1", rounds=1, store_uri="s3://b/runs/other")
    entry.run(spec, resource_sampler=lambda emit: None)
    (start,) = _of(_events(tmp_path, "nc-1"), "rl_sampler_start")
    assert start["store"] == "s3://b/runs/other"


def test_store_identity_of_a_store_without_a_uri_is_its_path():
    path = SimpleNamespace(weights=SimpleNamespace(store=SimpleNamespace(root="/x")))
    assert entry.store_identity({"store": "/x"}, path) == ("/x", None)
    path = SimpleNamespace(weights=SimpleNamespace(store=SimpleNamespace(uri="s3://b/p")))
    assert entry.store_identity({"store": "/tmp/stage/b/p"}, path) == ("s3://b/p", "/tmp/stage/b/p")


# ------------------------------------------------------------------ the CLI hint

def test_the_stopped_state_hint_says_what_moving_the_file_does(tmp_path):
    from yeto.launch.nc.lease_pool import PoolConfig
    from yeto.launch.nc.pool_run import stopped_state_hint
    from yeto.launch.pool.island_pool import IslandPool, Member as PoolMember

    pool = IslandPool(PoolConfig(target=1, accounts=[1], shard_base="fss-i1", shard_limit=2), SimpleNamespace())
    pool.stopped = "sampler fss-i1-0 finished (exit 0)"
    pool.members = [PoolMember(1, "t1", "fss-i1-0", state=GONE)]
    text = stopped_state_hint(tmp_path / "pool.json", pool, {"shard_base": "fss-i1"})
    assert "move it away to start a new pool" not in text
    for part in ("fss-i1-0 again", "this one used: fss-i1-0", "already ended there are skipped",
                 "shard_limit 2", "set aside", "new run_prefix"):
        assert part in text, part


def test_a_stop_request_unread_at_the_join_is_read_again_then_fails_the_start(monkeypatch):
    """Unjudged, a stale request would stop the sampler at a later poll: the join reads it
    again, and a store that keeps failing fails the start (the pool restarts it)."""
    monkeypatch.setattr(entry, "JOIN_READ_PAUSE_S", 0.0)
    objects = Objects()
    store = SimpleNamespace(objects=objects)
    request_handoff(store, "nc-0", "nc-2")
    fails = {"n": 2}
    read = objects.read

    def flaky(key):
        if "handoffs/" in key and fails["n"] > 0:
            fails["n"] -= 1
            raise ConnectionError("store blip")
        return read(key)

    objects.read = flaky
    events = []
    buffer = SimpleNamespace(data_ended=False)
    not_run, stale = entry.join_check(buffer, store, "nc-0", "inc", [], lambda e, **f: events.append((e, f)))
    assert not_run is None and stale is not None and events[0][1]["why"] == "successor_never_joined"
    request_handoff(store, "nc-0", "nc-2")
    fails["n"] = entry.JOIN_READ_ATTEMPTS
    with pytest.raises(ConnectionError):
        entry.join_check(buffer, store, "nc-0", "inc", [], lambda e, **f: None)

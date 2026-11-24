"""Node records in the store (``nodes/<ticket>/g<generation>/{ready,heartbeat,exit}``): the control plane
judges ready / stuck / exited from them alone; a machine opened by hand joins with its
token and is fenced and handed off like an ordered one."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest

from tests.test_node_runtime import _setup, code_tar  # noqa: F401 - the module fixture
from yeto.launch.nc.elastic import ObjectRecords, fence_shard
from yeto.launch.pool.island_pool import Member as PoolMember
from yeto.node import records
from yeto.node.assign import assign_manual, assign_member
from yeto.node.records import NodeReporter, node_status, probe_of
from yeto.node.report import SamplerNodeReport, parse_metrics
from yeto.node.runtime import NodeRuntime
from yeto.rl.engine.object_store import DirectoryObjectStore
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer, shard_index
from yeto.rl.engine.sampler_resume import MEMBERS_DIR, Member, write_member


def test_status_from_the_store_alone(tmp_path):
    objects = DirectoryObjectStore(tmp_path)
    assert node_status(objects, "t").state == "unassigned"
    code = records.CodeRef("abc", "node-code/abc.tar", "0" * 64)
    records.assign(objects, ticket="t", shard="s", spec={}, code=code, clock=lambda: 0.0)
    assert node_status(objects, "t").state == "assigned"
    rep = NodeReporter(objects, "t", 1, clock=lambda: 1000.0)
    rep.heartbeat("model")
    assert node_status(objects, "t", now=1010.0).state == "booting"
    assert node_status(objects, "t", now=1010.0).stage == "model"
    rep.ready(shard="s")
    rep.heartbeat("running", engine={"gen_tokens": 12.0, "running": 3, "queued": 1})
    ready = node_status(objects, "t", now=1010.0)
    assert ready.state == "ready" and ready.engine["running"] == 3
    probe = probe_of(ready)
    assert probe.joined and probe.running == 3 and probe.queued == 1 and probe.gen_tokens == 12.0
    stuck = node_status(objects, "t", now=1000.0 + records.DEFAULT_STALE_S + 1)
    assert stuck.state == "stuck" and "no heartbeat" in stuck.reason
    assert probe_of(stuck).progress_age_s > records.DEFAULT_STALE_S
    rep.exit(75, "reclaimed")
    gone = node_status(objects, "t", now=5000.0)
    assert gone.state == "exited" and gone.exit_code == 75 and gone.reason == "reclaimed"


def test_records_of_another_generation_are_not_this_assignment(tmp_path):
    objects = DirectoryObjectStore(tmp_path)
    code = records.CodeRef("abc", "node-code/abc.tar", "0" * 64)
    records.assign(objects, ticket="t", shard="s", spec={}, code=code)
    NodeReporter(objects, "t", 1).exit(1, "error")
    records.assign(objects, ticket="t", shard="s", spec={}, code=code)  # generation 2
    assert node_status(objects, "t").state == "assigned"


def test_an_older_generation_never_overwrites_the_newer_ones_records(tmp_path):
    """A node still running generation 1 keeps heartbeating after the ticket was
    reassigned: the healthy generation-2 node stays ready, not stuck."""
    objects = DirectoryObjectStore(tmp_path)
    code = records.CodeRef("abc", "node-code/abc.tar", "0" * 64)
    records.assign(objects, ticket="t", shard="s", spec={}, code=code)
    old = NodeReporter(objects, "t", 1, clock=lambda: 900.0)
    old.ready(shard="s")
    assignment, _ = assign_member(objects, {"run_prefix": "r"}, type("M", (), {"ticket": "t", "shard": "s"})(),
                                  ["s"], _profile(tmp_path), code, spec={"sampler_id": "s"})
    assert assignment.generation == 2
    assert objects.read(records.node_key("t", "ready", 1)) is not None  # each generation keeps its own
    new = NodeReporter(objects, "t", 2, clock=lambda: 1000.0)
    new.ready(shard="s")
    new.heartbeat("running", engine={"running": 2})
    old.heartbeat("running")  # the old node, after the new one: its own generation's key
    old.exit(1, "error")
    status = node_status(objects, "t", now=1010.0)
    assert status.state == "ready" and status.generation == 2 and status.engine["running"] == 2
    assert node_status(objects, "t", now=1010.0, generation=1).state == "exited"
    assert records.node_key("t", "heartbeat", 2) == "nodes/t/g2/heartbeat.json"


class _ReadFails(DirectoryObjectStore):
    def read(self, key):
        if key.startswith(records.ASSIGNMENTS_DIR):
            raise OSError("store outage")
        return super().read(key)


def test_a_generation_never_goes_back(tmp_path):
    """gen1 -> gen2 -> the old gen1 node writes its exit -> the assignment does not read:
    no new gen1 (a healthy node would be read as exited); a lost assignment continues past
    every generation its node records show; a revoked one keeps its generation."""
    objects = DirectoryObjectStore(tmp_path)
    code = records.CodeRef("abc", "node-code/abc.tar", "0" * 64)
    records.assign(objects, ticket="t", shard="s", spec={}, code=code)
    records.assign(objects, ticket="t", shard="s", spec={}, code=code)
    NodeReporter(objects, "t", 1).exit(1, "error")
    NodeReporter(objects, "t", 2).ready(shard="s")
    with pytest.raises(OSError, match="outage"):
        records.assign(_ReadFails(tmp_path), ticket="t", shard="s", spec={}, code=code)
    objects.write(records.assignment_key("t"), b"{not json")
    with pytest.raises(records.AssignmentError, match="not reassigned"):
        records.assign(objects, ticket="t", shard="s", spec={}, code=code)
    assert node_status(objects, "t").state == "unassigned"
    from yeto.node.__main__ import main as node_main  # noqa: PLC0415

    with pytest.raises(records.AssignmentError, match="python -m yeto.node revoke --assignment t "):
        records.assign(objects, ticket="t", shard="s", spec={}, code=code)
    assert node_main(["revoke", "--assignment", "t", "--store", str(tmp_path)]) == 0  # the command it names
    assert json.loads(objects.read(records.assignment_key("t")))["generation"] == 3
    again, token = records.assign(objects, ticket="t", shard="s", spec={}, code=code)
    assert again.generation == 4
    records.revoke_assignment(objects, "t")
    with pytest.raises(records.AssignmentError, match="revoked"):
        records.read_assignment(objects, "t", token)
    assert node_status(objects, "t").state == "unassigned"
    objects.delete(records.assignment_key("t"))  # lost out of band: the node records still count
    assert records.assign(objects, ticket="t", shard="s", spec={}, code=code)[0].generation == 3


def test_the_token_is_bound_to_its_ticket_and_its_deadline(tmp_path):
    objects = DirectoryObjectStore(tmp_path)
    code = records.CodeRef("abc", "node-code/abc.tar", "0" * 64)
    _, token = records.assign(objects, ticket="a", shard="s", spec={}, code=code, expires_at=100.0)
    assert records.read_assignment(objects, "a", token, clock=lambda: 99.0).ticket == "a"
    with pytest.raises(records.AssignmentError, match="expired"):
        records.read_assignment(objects, "a", token, clock=lambda: 101.0)
    with pytest.raises(records.AssignmentError, match="not the token"):
        records.read_assignment(objects, "a", "guess", clock=lambda: 99.0)
    objects.write(records.assignment_key("b"), objects.read(records.assignment_key("a")))  # copied under b
    with pytest.raises(records.AssignmentError, match="names ticket 'a'"):
        records.read_assignment(objects, "b", token, clock=lambda: 99.0)
    member, manual, manual_token = assign_manual(objects, {"run_prefix": "r"}, shard="m-0", shards=["m-0"],
                                                 profile=_profile(tmp_path), code=code, run_id="r",
                                                 spec={"sampler_id": "m-0"}, clock=lambda: 1000.0,
                                                 member_factory=lambda **kw: type("M", (), kw)())
    assert manual.expires_at == 1000.0 + 24 * 3600
    with pytest.raises(records.AssignmentError, match="expired"):
        records.read_assignment(objects, member.ticket, manual_token, clock=lambda: 1000.0 + 24 * 3600 + 1)


def _profile(tmp_path):
    from yeto.launch.node_inventory import NodePaths, NodeProfile  # noqa: PLC0415

    return NodeProfile.for_card("box", "L4", 1, hbm_gib=24.0, container="process",
                                paths=NodePaths(work=str(tmp_path / "w"), models=str(tmp_path / "m")))


def test_progress_age_is_the_last_round_not_the_last_heartbeat(tmp_path):
    objects = DirectoryObjectStore(tmp_path)
    code = records.CodeRef("abc", "node-code/abc.tar", "0" * 64)
    records.assign(objects, ticket="t", shard="s", spec={}, code=code)
    assert probe_of(node_status(objects, "t")).sampler == "running"  # assigned: starting, not absent
    now = [1000.0]
    rounds = [0]
    report = SamplerNodeReport(NodeReporter(objects, "t", 1, clock=lambda: now[0]), heartbeat_s=3600,
                               metrics=lambda port: {}, clock=lambda: now[0])
    report.progress = lambda: {"rounds": rounds[0]}
    report.ready(shard="s")  # joined at 1000
    now[0] = 1100.0
    rounds[0] = 1
    report._beat()  # a round done at 1100  # noqa: SLF001
    now[0] = 1500.0
    report._beat()  # alive, no new round  # noqa: SLF001
    probe = probe_of(node_status(objects, "t", now=1510.0))
    assert probe.joined and probe.progress_age_s == 410.0  # since the round, not since the heartbeat (10 s)


def test_a_sampler_waiting_for_the_trainer_is_not_stalled(tmp_path):
    objects = DirectoryObjectStore(tmp_path)
    code = records.CodeRef("abc", "node-code/abc.tar", "0" * 64)
    records.assign(objects, ticket="t", shard="s", spec={}, code=code)
    now = [1000.0]
    report = SamplerNodeReport(NodeReporter(objects, "t", 1, clock=lambda: now[0]), heartbeat_s=3600,
                               metrics=lambda port: {}, clock=lambda: now[0])
    report.progress = lambda: {"rounds": 3}
    tape = []
    emit = report.watching(lambda event, **fields: tape.append(event))
    report.ready(shard="s")
    now[0] = 2000.0
    emit("rl_sampler_waiting", reason="backlog_full")  # no new round: the trainer is away
    report._beat()  # noqa: SLF001
    probe = probe_of(node_status(objects, "t", now=2005.0))
    assert tape == ["rl_sampler_waiting"] and probe.trainer_wait and probe.progress_age_s == 5.0
    emit("rl_sampler_round_done")
    report._beat()  # noqa: SLF001
    assert not probe_of(node_status(objects, "t", now=2005.0)).trainer_wait


def test_engine_metrics_summary_of_vllm_and_sglang():
    vllm = ("# HELP x\nvllm:generation_tokens_total{model_name=\"m\"} 120.0\n"
            "vllm:num_requests_running{model_name=\"m\"} 4.0\nvllm:num_requests_waiting{model_name=\"m\"} 2.0\n"
            "vllm:kv_cache_usage_perc{model_name=\"m\"} 0.25\nvllm:kv_cache_usage_perc{model_name=\"n\"} 0.5\n")
    assert parse_metrics(vllm) == {"gen_tokens": 120.0, "running": 4.0, "queued": 2.0, "kv_cache": 0.5}
    assert parse_metrics("sglang:num_running_reqs 7\nsglang:num_queue_reqs 1\n") == {"running": 7.0, "queued": 1.0}


def test_sampler_heartbeat_carries_the_engine_summary(tmp_path):
    objects = DirectoryObjectStore(tmp_path)
    report = SamplerNodeReport(NodeReporter(objects, "t", 1), profile={"card": "L4", "source": "sky"},
                               heartbeat_s=3600, metrics_port=8000,
                               metrics=lambda port: {"gen_tokens": 5.0, "running": 1.0})
    report.start()
    report.ready(shard="s")
    report.stop()
    beat = json.loads((tmp_path / "nodes" / "t" / "g1" / "heartbeat.json").read_text())
    assert beat["stage"] == "running" and beat["engine"] == {"gen_tokens": 5.0, "running": 1.0}
    assert report.member_fields() == {"ticket": "t", "generation": 1, "source": "sky",
                                      "profile": {"card": "L4", "source": "sky"}}
    report.exit(0)
    assert json.loads((tmp_path / "nodes" / "t" / "g1" / "exit.json").read_text())["reason"] == "finished"


def test_a_member_record_without_a_node_is_unchanged(tmp_path):
    store = CheckpointStore(tmp_path)
    write_member(store, Member(sampler="a", incarnation="i", joined_at=1.0, resume={}))
    doc = json.loads((tmp_path / MEMBERS_DIR / "a" / "i.json").read_text())
    assert sorted(doc) == ["detail", "incarnation", "joined_at", "leave_reason", "left_at", "policy_version",
                           "resume", "rounds", "sampler"]
    write_member(store, Member(sampler="a", incarnation="j", joined_at=1.0, resume={}, node={"ticket": "t"}))
    assert json.loads((tmp_path / MEMBERS_DIR / "a" / "j.json").read_text())["node"] == {"ticket": "t"}


def test_a_sampler_without_a_ticket_writes_no_node_records(tmp_path, monkeypatch):
    from tests.test_stage1_gaps import _sampler_spec
    from yeto.rl.engine import engine_sampler_entry as entry

    monkeypatch.delenv("YETO_NODE_TICKET", raising=False)
    assert entry.run(_sampler_spec(tmp_path), resource_sampler=lambda emit: None) == 2
    assert not (tmp_path / "store" / "nodes").exists()
    [doc] = [json.loads(p.read_text()) for p in (tmp_path / "store" / MEMBERS_DIR / "ppu4").glob("*.json")]
    assert "node" not in doc


def _wait(predicate, timeout_s: float = 120.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.2)
    return False


@pytest.mark.parametrize("how", ["ordered", "manual"])
def test_fence_and_handoff_act_the_same_on_a_manual_machine(tmp_path, code_tar, how):  # noqa: F811
    """A sampler waiting on a full backlog is fenced by the pool (``pool-fence:<ticket>``):
    it stops and leaves; a replacement that continues its shard starts after its rounds."""
    over = {"num_rollouts": 1000, "max_backlog_groups": 4, "incarnation_lease": {"poll_s": 0.2},
            "buffer_shards": ["a", "b"]}
    objects, code, profile, spec = _setup(tmp_path, code_tar, shard="a", **over)
    cfg = {"run_prefix": "run1"}
    env = {k: v for k, v in os.environ.items() if not k.startswith("YETO_")}
    if how == "manual":
        member, assignment, token = assign_manual(objects, cfg, shard="a", shards=["a", "b"], profile=profile,
                                                  code=code, run_id="run1", spec=spec)
    else:
        member = PoolMember(account=1, ticket="nc-1", shard="a")
        assignment, token = assign_member(objects, cfg, member, ["a", "b"], profile, code, source="nc", spec=spec)
    rt = NodeRuntime(objects, records.read_assignment(objects, member.ticket, token), profile, store_uri=None,
                     environ=env, heartbeat_s=0.5, skip_preflight=True)
    result: list[int] = []
    thread = threading.Thread(target=lambda: result.append(rt.main()), daemon=True)
    thread.start()
    try:
        _fence_a_running_sampler(objects, member, profile, thread, result)
    finally:
        if thread.is_alive() and rt.child is not None:  # a failed wait: never leave the sampler behind
            rt.child.kill()
    _continue(objects, cfg, member, profile, code, spec, env)


def _fence_a_running_sampler(objects, member, profile, thread, result) -> None:
    log = Path(profile.paths.work) / "logs" / "a.log"
    assert _wait(lambda: node_status(objects, member.ticket).state == "ready", 60), log.read_text()[-2000:]

    def groups() -> int:
        buffer = FileSampleBuffer(CheckpointStore(objects.root), batch_size=1, index=shard_index("a"),
                                  create_index=False)
        buffer.refresh()
        return len(buffer)

    assert _wait(lambda: groups() >= 4, 60), log.read_text()[-2000:]  # waits on its full backlog now
    fence_shard(ObjectRecords(objects), "a", by=member.ticket)
    thread.join(timeout=120)
    assert result, log.read_text()[-2000:]


def _continue(objects, cfg, member, profile, code, spec, env) -> None:
    status = node_status(objects, member.ticket)
    # stopped by the pool's fence (its stop request names pool-fence:<ticket>), long before its 1000 rounds
    assert status.state == "exited" and status.exit_code == 0, status
    store = Path(objects.root)
    [rec] = [json.loads(p.read_text()) for p in (store / MEMBERS_DIR / "a").glob("*.json")]
    assert rec["leave_reason"] in ("FENCED", "FINISHED") and rec["rounds"] < 1000
    assert rec["node"]["ticket"] == member.ticket and status.heartbeat["leave_reason"] == rec["leave_reason"]
    tape_a = next((Path(profile.paths.work) / "sampler").glob("events-a.*.jsonl")).read_text()
    assert f"pool-fence:{member.ticket}" in tape_a
    # the replacement continues the fenced shard's prompts, on the same rules
    spec_b = dict(spec, sampler_id="b", num_rollouts=2, max_backlog_groups=100,
                  event_tape="/work/sampler/events-b.jsonl", handoff_wait_s=0)
    succ = PoolMember(account=1, ticket=f"{member.ticket}-r", shard="b", replaces_shard="a")
    assignment_b, token_b = assign_member(objects, cfg, succ, ["a", "b"], profile, code, source="nc", spec=spec_b)
    assert assignment_b.continues_shard == "a"
    rt_b = NodeRuntime(objects, records.read_assignment(objects, succ.ticket, token_b), profile, store_uri=None,
                       environ=env, skip_preflight=True)
    assert rt_b.main() == 0
    tape = next((Path(profile.paths.work) / "sampler").glob("events-b.*.jsonl"))
    handoff = [json.loads(line) for line in tape.read_text().splitlines() if '"rl_sampler_handoff"' in line]
    assert handoff and handoff[0]["predecessor"] == "a" and handoff[0]["start_rollout_id"] >= 2
    assert handoff[0]["predecessor_groups"] >= 4

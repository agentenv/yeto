"""rl-infra-spec 3.6 fault evidence (T36-REVIEW §3.3 C1/C2/C3'), CPU.

C2  real ``rollout_meta_hook.record_trained_groups`` -> ``extract_rollout_metadata``
    -> ``handle_from_metadata`` -> ``BatchLedger.prepare`` reconciliation with a
    partial round: one trained group has an overlong-filtered sample, one
    completed group is not trained (dynamic-filter drop), two submitted groups
    were aborted in flight. filtered + trained + engine_discarded == submitted.
C3' the full ``publish`` after ``outer_recorded`` raises PublicationError, the
    driver exits, and the strict-avg restart (syncer authoritative version k+1)
    continues without superseding, discarding or re-training anything.
C1  a train-step failure discards the prepared batch; the strict restart retries
    the same rollout (attempt 1) and every group is optimizer-applied exactly once.

The no-sync restart (``rebase(0)`` -> LedgerError, refusal) stays covered by
``test_rl_reconfig_recovery.py::test_no_sync_restart_hits_the_ledger_and_the_restart_loop_is_bounded``.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from types import SimpleNamespace

import pytest
import torch

from yeto.rl.algos import grpo_knobs as gk
from yeto.rl.engine.algorithm import AlgorithmSpec
from yeto.rl.engine.bridges import StrictAvgSync
from yeto.rl.engine.driver import PublicationError
from yeto.rl.engine.fake import FakeEngine
from yeto.rl.engine.journal import read_journal
from yeto.rl.engine.ledger import BatchLedger, LedgerError
from yeto.rl.adapters.miles import rollout_meta_hook as hook
from yeto.rl.adapters.miles.rollout import handle_from_metadata, policy_token

from test_rl_engine_driver import _strict_config, _strict_syncer
from test_rl_reconfig_e1 import NAME, _setup

H = "c" * 64


# --------------------------------------------------------------------------- C2: partial groups
class Status(Enum):
    COMPLETED = "completed"
    TRUNCATED = "truncated"
    ABORTED = "aborted"


@dataclass
class Span:
    version: str


@dataclass
class Call:
    spans: list


@dataclass
class Sample:  # the miles.utils.types.Sample fields the hooks read
    index: int
    group_index: int
    rollout_id: int | None = None
    reward: float = 0.0
    response_length: int = 5
    weight_versions: list = field(default_factory=list)
    status: Status = Status.COMPLETED
    metadata: dict = field(default_factory=dict)
    remove_sample: bool = False

    def get_reward_value(self, args):
        return self.reward


def _group(gi, token, statuses=("completed", "completed")):
    return [
        Sample(index=gi * 10 + i, group_index=gi, reward=float(i % 2),
               weight_versions=[Call([Span(token)])], status=Status(st))
        for i, st in enumerate(statuses)
    ]


def _rollout_through_the_hooks(args, source, sink, *, rollout_id, trained, dropped, submitted):
    """One Miles rollout as the rollout process runs it (recorder, all-samples hook)."""
    token = policy_token(rollout_id, H)
    hook.put_policy_token(token, sink)
    kept = [_group(gi, token, sts) for gi, sts in trained]
    lost = [_group(gi, token) for gi in dropped]
    source.sample_offset += submitted  # prompts drawn this rollout (over-sampling included)
    source.sample_group_index += submitted
    hook.record_trained_groups(args, kept)  # --rollout-sample-filter-path
    all_samples = sorted(kept + lost, key=lambda g: g[0].index)
    hook.extract_rollout_metadata(args, all_samples, source)  # --rollout-all-samples-process-path
    payload = json.loads((sink_dir(sink) / f"rollout-{rollout_id}.json").read_text())
    return handle_from_metadata(payload, rollout_id=rollout_id, policy_version=rollout_id,
                                policy_hash=H, data_pack=None), payload


def sink_dir(sink):
    from pathlib import Path

    return Path(sink.partition(":")[2])


def test_partial_round_reconciles_through_the_real_hooks(tmp_path, monkeypatch):
    sink = f"dir:{tmp_path / 'meta'}"
    monkeypatch.setenv(hook.META_SINK_ENV, sink)
    spec = gk.with_pipeline_plugins(AlgorithmSpec(sampling={"overlong_filter": True}))
    args = SimpleNamespace(yeto_rl_elastic_metadata=True, **gk.runtime_attrs(spec))
    source = SimpleNamespace(sample_offset=0, epoch_id=0, sample_group_index=0, sample_index=0,
                             get_buffer_length=lambda: 0)
    led = BatchLedger(tmp_path / "state")

    # rollout 0: first rollout, the drawn count is unknown (no previous offset)
    h0, p0 = _rollout_through_the_hooks(args, source, sink, rollout_id=0, trained=[(0, ("completed",) * 2),
                                                                                (1, ("completed",) * 2)],
                                        dropped=[], submitted=2)
    assert "submitted_groups" not in p0 and h0.aborted_in_flight_groups is None
    led.prepare(h0, policy_token=policy_token(0, H))
    led.optimizer_applied(0)
    led.outer_recorded(0)

    # rollout 1 (partial): 3 groups trained (g2 keeps one overlong-filtered sample), g5 completed
    # but not trained (dynamic-filter drop), 6 submitted -> 2 aborted in flight.
    h1, p1 = _rollout_through_the_hooks(
        args, source, sink, rollout_id=1,
        trained=[(2, ("completed", "truncated", "completed")), (3, ("completed",) * 2),
                 (4, ("completed",) * 2)],
        dropped=[5], submitted=6)
    assert p1["submitted_groups"] == 6 and p1["completed"] == 3 and p1["filtered"] == 1
    assert h1.aborted_in_flight_groups == 2 and h1.carried_over == 0 and h1.buffer_length == 0
    assert h1.data_cursor == {"sample_offset": 8, "epoch_id": 0, "sample_group_index": 8,
                              "sample_index": 0}
    led.prepare(h1, policy_token=policy_token(1, H))

    recs = [r for r in read_journal(tmp_path / "state/ledger") if r.get("rollout_id") == 1]
    kinds = {r["kind"]: r for r in recs}
    assert set(kinds) == {"prepared", "filtered", "engine_discarded", "carried_over_report"}
    prepared, filtered, discarded = kinds["prepared"], kinds["filtered"], kinds["engine_discarded"]
    assert prepared["group_ids"] == ["g2", "g3", "g4"] and prepared["attempt"] == 0
    assert prepared["samples"] == 7  # filtered samples stay in the batch identity (masked, not dropped)
    # no silent loss: every submitted group is accounted for exactly once
    assert filtered["detail"]["groups"] + len(prepared["group_ids"]) + discarded["groups"] == 6
    assert filtered["detail"]["mechanism"] == "rollout_meta_hook.record_trained_groups"
    assert filtered["detail"]["reason"] and "over-sampling" in filtered["detail"]["reason"]
    assert filtered["detail"]["samples"] == 1 and "sample filter" in filtered["detail"]["sample_mechanism"]
    assert discarded["mechanism"] == "miles generate_rollout abort" and "partial_rollout" in discarded["reason"]
    assert kinds["carried_over_report"] == {**kinds["carried_over_report"], "carried_over_reported": True,
                                            "carried_over": 0, "buffer_length": 0}
    # cursor semantics: filtered / engine_discarded are terminal and never consumed; a cut sees
    # only the prepared batch as unconsumed; carried_over is non-terminal and empty under Miles.
    summary = led.cut_summary()
    assert summary["ready_unconsumed_group_ids"] == ["g2", "g3", "g4"]
    assert summary["engine_discarded_groups"] == 2 and summary["carried_over"] == 0
    assert summary["engine_carried_over"] == 0 and led.open_carried_over() == {}
    led.optimizer_applied(1)
    assert led.unconsumed() == [] and led.cut_summary()["ready_unconsumed"] == 0
    led.outer_recorded(1)
    # the dropped group was never consumed: generating it again later is legal ...
    h2, _ = _rollout_through_the_hooks(args, source, sink, rollout_id=2,
                                       trained=[(5, ("completed",) * 2)], dropped=[], submitted=1)
    led.prepare(h2, policy_token=policy_token(2, H))
    # ... while a trained group reappearing is refused (no double consumption)
    h3, _ = _rollout_through_the_hooks(args, source, sink, rollout_id=3,
                                       trained=[(3, ("completed",) * 2)], dropped=[], submitted=1)
    with pytest.raises(LedgerError, match="already trained in rollout 1"):
        led.prepare(h3, policy_token=policy_token(3, H))
    assert led.state(3) is None  # the refused batch left no record
    led.close()
    replayed = BatchLedger(tmp_path / "state")
    assert replayed.batch(1)["filtered"]["groups"] == 1 and replayed.batch(1)["engine_discarded"] == 2
    assert replayed.state(1) == "outer_recorded" and replayed.state(2) == "prepared"
    replayed.close()


# --------------------------------------------------------------------------- strict-avg islands
def _strict_island(root, syncer, *, rounds, tape):
    sync_factory = lambda e: StrictAvgSync(  # noqa: E731
        _strict_config(root, e, learner_id=0, rounds=rounds, tape=tape),
        client_factory=lambda _bridge: syncer.client(0))
    return _setup(root, rounds=rounds, outer="strict-avg", sync_factory=sync_factory)


def _syncer(rounds):
    return _strict_syncer(FakeEngine(tensors={NAME: torch.zeros(1, 2)}), learners=1, rounds=rounds)


def _ledger_records(root):
    return read_journal(root / "state/ledger")


def _group_consumption(records):
    """group_id -> number of optimizer_applied records that consumed it."""
    prepared = {}
    for r in records:
        if r["kind"] == "prepared":
            prepared[(r["rollout_id"], r["attempt"])] = r["group_ids"]
    applied = Counter()
    for r in records:
        if r["kind"] == "optimizer_applied":
            applied.update(prepared[(r["rollout_id"], r["attempt"])])
    return applied


def _events(root):
    return [json.loads(x) for x in (root / "events.jsonl").read_text().splitlines()]


def _close(driver, ctl):
    ctl.close()
    driver.ledger.close()


def test_full_publish_failure_after_outer_recorded_restarts_at_the_next_rollout(tmp_path):
    """C3': publish of v2 (after rollout 1's outer_recorded) fails -> driver exits; the
    syncer's authoritative version is 2, the restart rebases at 2 and continues."""
    rounds = 4
    syncer = _syncer(rounds)
    driver, ctl, fork, pool, publisher, trained_a, _ = _strict_island(tmp_path, syncer, rounds=rounds,
                                                                   tape="b-a.jsonl")
    original = publisher.publish

    def publish(state):
        if state.policy_version == 2:
            raise PublicationError("injected: full publication of v2 failed")
        return original(state)

    publisher.publish = publish
    with pytest.raises(PublicationError, match="injected"):
        driver.run()
    engine_a = driver.trainer.engine
    assert [c[1] for c in engine_a.calls if c[0] == "generate"] == [0, 1]
    assert syncer.version == 2  # rollout 1's update is part of the outer progress
    recs = _ledger_records(tmp_path)
    assert [(r["kind"], r["rollout_id"]) for r in recs if r["kind"] in
            ("prepared", "optimizer_applied", "outer_recorded", "discarded", "superseded")] == [
        (k, i) for i in range(2) for k in ("prepared", "optimizer_applied", "outer_recorded")]
    assert driver.ledger.unconsumed() == []  # the batch whose publish failed is consumed, not dangling
    _close(driver, ctl)

    driver, ctl, fork, pool, publisher, trained_b, _ = _strict_island(tmp_path, syncer, rounds=rounds,
                                                                   tape="b-b.jsonl")
    driver.run()
    engine_b = driver.trainer.engine
    assert [c[1] for c in engine_b.calls if c[0] == "generate"] == [2, 3]
    assert [c[1] for c in engine_b.calls if c[0] == "publish"][0] == 2  # restarted at the syncer's version
    recs = _ledger_records(tmp_path)
    assert not [r for r in recs if r["kind"] in ("discarded", "superseded")]
    assert not [r for r in recs if r["kind"] == "prepared" and r["attempt"] != 0]
    assert [r["rollout_id"] for r in recs if r["kind"] == "prepared"] == [0, 1, 2, 3]
    assert all(driver.ledger.state(i) == "outer_recorded" for i in range(rounds))
    assert not [r for r in recs if r["kind"] == "outer_recorded" and r.get("recovered")]
    # group level: every generated group consumed exactly once, none twice, none lost
    consumption = _group_consumption(recs)
    assert set(consumption) == {f"r{i}-g{g}" for i in range(rounds) for g in range(engine_b.groups)}
    assert set(consumption.values()) == {1}
    assert sorted(trained_a + trained_b) == sorted(set(trained_a + trained_b))
    assert len(trained_a + trained_b) == rounds
    assert syncer.history == [0, 1, 2, 3, 4]
    ev = _events(tmp_path)
    assert [e["event"] for e in ev].count("rl_driver_start") == 2
    # the restart's first publication is the rebase point and the first prepared rollout
    restart = max(i for i, e in enumerate(ev) if e["event"] == "rl_driver_start")
    first_pub = next(e for e in ev[restart:] if e["event"] == "rl_publication")
    assert first_pub["policy_version"] == 2
    assert [r["rollout_id"] for r in recs if r["kind"] == "prepared"][2] == 2
    _close(driver, ctl)


def test_train_step_failure_is_discarded_and_retried_once_on_strict_restart(tmp_path):
    """C1 at driver level: rollout 1's train step raises -> ``discarded`` (error kept); the
    restart rebases at the syncer's version 1 and retries rollout 1 as attempt 1 with the
    same groups; group-level consumption is exactly once."""
    rounds = 3
    syncer = _syncer(rounds)
    driver, ctl, *_rest, trained_a, _ = _strict_island(tmp_path, syncer, rounds=rounds, tape="c-a.jsonl")
    original = driver.trainer.train_step

    def train_step(batch):
        if batch.rollout_id == 1:
            raise RuntimeError("injected: optimizer step failed before apply")
        return original(batch)

    driver.trainer.train_step = train_step
    with pytest.raises(RuntimeError, match="injected"):
        driver.run()
    recs = _ledger_records(tmp_path)
    discarded = [r for r in recs if r["kind"] == "discarded"]
    assert [(r["rollout_id"], r["attempt"]) for r in discarded] == [(1, 0)]
    assert "RuntimeError: injected" in discarded[0]["error"]
    assert driver.ledger.unconsumed() == [] and syncer.version == 1
    _close(driver, ctl)

    driver, ctl, *_rest, trained_b, _ = _strict_island(tmp_path, syncer, rounds=rounds, tape="c-b.jsonl")
    driver.run()
    recs = _ledger_records(tmp_path)
    prepared = [(r["rollout_id"], r["attempt"], r["group_ids"]) for r in recs if r["kind"] == "prepared"]
    assert [(rid, att) for rid, att, _ in prepared] == [(0, 0), (1, 0), (1, 1), (2, 0)]
    assert prepared[1][2] == prepared[2][2]  # the same groups, retried after an explicit discard
    assert not [r for r in recs if r["kind"] == "superseded"]
    assert all(driver.ledger.state(i) == "outer_recorded" for i in range(rounds))
    consumption = _group_consumption(recs)
    assert set(consumption) == {f"r{i}-g{g}" for i in range(rounds) for g in range(driver.trainer.engine.groups)}
    assert set(consumption.values()) == {1}
    assert len(trained_a + trained_b) == rounds and len(set(trained_a + trained_b)) == rounds
    _close(driver, ctl)

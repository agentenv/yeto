"""S27 rollout ahead: the engines generate round r+1 while the trainer trains round r.

No Ray. Covered here (``yeto.rl.engine.rollout_ahead`` module doc):

* the driver's round order: batch r+1 starts before train r and is joined
  before the publication of v_{r+1}; a weight update never runs while the
  engines generate; every batch after round 0 lags exactly one version;
* the final round: a batch started before an unannounced stop is waited for
  and discarded; a known final round starts nothing;
* failures: a failed join fails the round before the publication; a failed
  train step cancels the batch in flight;
* the data position recorded at boundary r excludes batch r+1's prompts;
* fail-closed refusals (driver and launcher);
* the Miles pool's split generation over the island loop, the explicit
  rollout round on the rollout side, the round evidence binding, and the
  Miles argv (no fully_async, no partial rollout).
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import threading
from types import SimpleNamespace

import pytest
import torch

from yeto.rl.engine.algorithm import AlgorithmSpec, CorrectionSpec, ExecutionSpec
from yeto.rl.engine.bridges import LocalOnlySync
from yeto.rl.engine.driver import DriverError, EventTape, IslandDriver, PublicationError, RoundFailedError
from yeto.rl.engine.execution_profile import ExecutionProfile
from yeto.rl.engine.fake import FakeEngine, fake_capabilities
from yeto.rl.engine.rollout_ahead import PendingBatch, launch_problems

NAME = "base_model.model.layer.lora_A.weight"
HASH = "a" * 64
OLDER = "c" * 64


def _events(tmp_path):
    return [json.loads(line) for line in (tmp_path / "e.jsonl").read_text().splitlines()]


def _driver(tmp_path, *, rounds=4, limit=1, ahead=1, sync=None, **engine_kw):
    engine_kw.setdefault("placement_kind", "fixed-partition")
    engine = FakeEngine(tensors={NAME: torch.zeros(1, 2)}, step_delta=1.0, **engine_kw)
    mode = "colocated-serial" if engine.placement_kind == "colocated" else "partitioned-serial"
    algorithm = AlgorithmSpec(execution=ExecutionSpec(max_policy_staleness=max(limit, 1)),
                              correction=CorrectionSpec(method="tis", tis_clip=2.0, tis_clip_low=0.0))
    caps = fake_capabilities(execution_modes={"colocated-serial", "partitioned-serial"})
    caps = dataclasses.replace(caps, execution=dataclasses.replace(
        caps.execution, max_policy_staleness=max(limit, 1)))
    profile = ExecutionProfile("p", mode, "strict-avg", max_policy_age=limit,
                               algorithm_contract="bounded-staleness" if limit else "on-policy",
                               groups_per_batch=engine.groups, samples_per_group=engine.samples_per_group,
                               algorithm_spec_sha256=algorithm.sha256())
    driver = IslandDriver(learner_id=0, rollout=engine.rollout, trainer=engine.trainer,
                          policy_state=engine.policy_state, publisher=engine.publisher,
                          placement=engine.placement, capabilities=caps, algorithm=algorithm,
                          sync=sync or LocalOnlySync(rounds), profile=profile,
                          events=EventTape(tmp_path / "e.jsonl", 0))
    driver.rollout_ahead = ahead
    return driver, engine


def _calls(engine):
    return [c for c in engine.calls if c[0] in ("generate", "generate_done", "generate_cancel",
                                                "train", "publish")]


class _UnannouncedStop(LocalOnlySync):
    """A session whose stop is known only at the boundary (the full-parameter syncer)."""

    def is_final_round(self, driver, *, rollout_id):
        return False


# -- the round pipeline ------------------------------------------------------------------


def test_next_batch_generates_during_training_and_publication_waits_for_it(tmp_path):
    driver, engine = _driver(tmp_path, rounds=4)
    final = driver.run()

    assert _calls(engine) == [
        ("publish", 0), ("generate", 0), ("generate", 1), ("train", 0),
        ("generate_done", 1), ("publish", 1),
        ("generate", 2), ("train", 1), ("generate_done", 2), ("publish", 2),
        ("generate", 3), ("train", 2), ("generate_done", 3), ("publish", 3),
        ("train", 3), ("publish", 4),  # known final round: nothing starts ahead
    ]
    assert final.policy_version == 4
    assert torch.equal(final.tensors[NAME], torch.full((1, 2), 4.0))
    events = _events(tmp_path)
    ahead = [e for e in events if e["event"] == "rl_rollout_ahead"]
    assert [(e["rollout_id"], e["policy_version"]) for e in ahead] == [(1, 0), (2, 1), (3, 2)]
    assert all(e["wait_seconds"] >= 0 and e["generate_seconds"] >= e["wait_seconds"] for e in ahead)
    # every group of batch r >= 1 names the publication r - 1 (the policy that generated it)
    labels = {e["policy_version"]: e["rl/policy_token"] for e in events if e["event"] == "rl_publication"}
    groups = {e["rollout_id"]: {g[1] for g in e["groups"]} for e in events if e["event"] == "rl_group_policies"}
    assert groups == {0: {labels[0]}, 1: {labels[0]}, 2: {labels[1]}, 3: {labels[2]}}
    phases = [e["phase"] for e in events if e["event"] == "rl_driver_phase"]
    assert phases.count("generate") == 1 and phases.count("generate_wait") == 3


def test_trained_batches_lag_exactly_one_version(tmp_path):
    driver, engine = _driver(tmp_path, rounds=3)
    trained = []
    original = engine.trainer.train_step
    engine.trainer.train_step = lambda batch: trained.append(
        (batch.rollout_id, batch.policy_version)) or original(batch)
    driver.run()
    assert trained == [(0, 0), (1, 0), (2, 1)]


def test_a_batch_started_before_an_unannounced_stop_is_discarded(tmp_path):
    driver, engine = _driver(tmp_path, sync=_UnannouncedStop(2))
    driver.run()

    assert _calls(engine)[-4:] == [("generate", 2), ("train", 1), ("generate_done", 2), ("publish", 2)]
    assert ("train", 2) not in engine.calls
    last = [e for e in _events(tmp_path) if e["event"] == "rl_rollout_ahead"][-1]
    assert last["rollout_id"] == 2 and last["discarded"] is True


def test_hf_exports_run_while_the_next_batch_generates(tmp_path):
    # --rl-export-hf-rounds under rollout ahead: the export of a round runs after its sync
    # boundary with the next batch still generating, then the driver joins that batch and
    # publishes; the final export (an unannounced stop) runs before the batch in flight is
    # discarded and before the last publication (the final ACK).
    from yeto.rl.engine.hf_export import COMPLETE_MARKER, ExportRounds, HfRoundExports
    from yeto.rl.engine.resume import CheckpointStore

    driver, engine = _driver(tmp_path, sync=_UnannouncedStop(3))

    def write(rollout_id, directory):
        engine.calls.append(("hf_export", rollout_id))
        directory.mkdir(parents=True)
        (directory / "config.json").write_text("{}")
        (directory / COMPLETE_MARKER).touch()

    driver.hf_exports = HfRoundExports(ExportRounds((1,), final=True), CheckpointStore(tmp_path / "store"),
                                       write, staging=tmp_path / "staging")
    driver.run()

    calls = [c for c in engine.calls if c[0] in ("generate", "generate_done", "train", "publish", "hf_export")]
    assert calls == [
        ("publish", 0), ("generate", 0), ("generate", 1), ("train", 0),
        ("hf_export", 0), ("generate_done", 1), ("publish", 1),
        ("generate", 2), ("train", 1), ("generate_done", 2), ("publish", 2),
        ("generate", 3), ("train", 2), ("hf_export", 2), ("generate_done", 3), ("publish", 3),
    ]
    events = _events(tmp_path)
    assert [(e["rollout_id"], e["final"], e["ok"]) for e in events if e["event"] == "rl_hf_export"] == [
        (0, False, True), (2, True, True)]
    assert [e for e in events if e["event"] == "rl_rollout_ahead"][-1]["discarded"] is True


def test_serial_rounds_are_unchanged_without_rollout_ahead(tmp_path):
    driver, engine = _driver(tmp_path, rounds=2, limit=0, ahead=0)
    driver.run()
    assert _calls(engine) == [("publish", 0), ("generate", 0), ("train", 0), ("publish", 1),
                              ("generate", 1), ("train", 1), ("publish", 2)]
    assert '"rl_rollout_ahead"' not in (tmp_path / "e.jsonl").read_text()


# -- mutual exclusion and failures ------------------------------------------------------


def test_publication_is_refused_while_the_engines_generate(tmp_path):
    driver, engine = _driver(tmp_path)
    driver.handshake()
    driver.publish(engine.policy_state.export(), rollout_id=0)
    driver._ahead = PendingBatch(rollout_id=1, policy_version=0, handle=None, data_cursor=None,
                                 started_at=0.0, started_clock=0.0)
    with pytest.raises(PublicationError, match="while the engines generate rollout 1"):
        driver.publish(engine.policy_state.export(), rollout_id=0)


def test_a_failed_join_fails_the_round_before_the_next_publication(tmp_path):
    driver, engine = _driver(tmp_path)

    def broken(pending):
        engine.generating.discard(pending.rollout_id)
        raise RuntimeError("engine lost")

    engine.rollout.finish_generate = broken
    with pytest.raises(RuntimeError, match="engine lost"):
        driver.run()
    assert ("publish", 1) not in engine.calls


def test_a_failed_train_step_cancels_the_batch_in_flight(tmp_path):
    driver, engine = _driver(tmp_path, failed_step_rounds={1})
    with pytest.raises(RoundFailedError):
        driver.run()
    assert ("generate_cancel", 2) in engine.calls and not engine.generating


# -- data position ----------------------------------------------------------------------


class _PositionLog:
    slot = "0/2"
    commit_errors: list = []

    def __init__(self):
        self.records = []

    def start(self, cursor):
        self.records.append(("start", cursor))

    def record(self, round_id, cursor):
        self.records.append((round_id, cursor))

    def close(self):
        return True


def test_the_recorded_position_excludes_the_batch_generated_ahead(tmp_path):
    driver, engine = _driver(tmp_path, rounds=3)
    pool = engine.rollout
    drawn = {"n": 0}
    original = pool.generate

    def generate(rollout_id, **kw):
        drawn["n"] += engine.groups
        return original(rollout_id, **kw)

    pool.generate = generate
    pool.data_cursor = lambda: {"sample_offset": drawn["n"]}
    driver.data_positions = log = _PositionLog()
    driver.run()
    # after round r the island trained r + 1 batches: a relaunch draws batch r+1 again
    assert log.records == [("start", {"sample_offset": 0})] + [
        (r, {"sample_offset": (r + 1) * engine.groups}) for r in range(3)]


# -- refusals ----------------------------------------------------------------------------


@pytest.mark.parametrize(("change", "message"), [
    (dict(limit=0), "max_policy_age >= 1"),
    (dict(placement_kind="colocated"), "fixed-partition island"),
])
def test_the_driver_refuses_an_unsupported_rollout_ahead(tmp_path, change, message):
    driver, engine = _driver(tmp_path, **change)
    with pytest.raises(DriverError, match=message):
        driver.run()
    assert not [c for c in engine.calls if c[0] in ("generate", "publish")]


def test_the_driver_refuses_pools_ledgers_and_round_cuts(tmp_path):
    driver, engine = _driver(tmp_path)
    engine.rollout.start_generate = None
    with pytest.raises(DriverError, match="split generation"):
        driver.run()
    driver, _ = _driver(tmp_path)
    driver.ledger = object()
    with pytest.raises(DriverError, match="batch ledger"):
        driver.run()
    driver, _ = _driver(tmp_path)
    driver.sync.round_cuts = object()
    with pytest.raises(DriverError, match="round cuts"):
        driver.run()


def test_launch_problems_name_every_missing_condition():
    ok = dict(backend_max=1, max_policy_age=1, placements=["fixed-partition"], sync_supported=True)
    assert launch_problems(0, **{**ok, "max_policy_age": 0}) == []
    assert launch_problems(1, **ok) == []
    assert launch_problems(2, **ok) == ["--rl-rollout-ahead 0..1"]
    problems = launch_problems(1, backend_max=0, max_policy_age=0, placements=["colocated"],
                               sync_supported=False, sample_roles=True, elastic=True,
                               island_cuts=True, overlap_eval=True)
    assert len(problems) == 8


# -- launcher ------------------------------------------------------------------------------


def _launch(extra):
    from test_rl_full_parameter_async import ASYNC, TWO_ISLANDS, _launcher_args
    from yeto.launcher import _prepare_rl_args, dry_run_plan

    args = _launcher_args("ports", ASYNC + ("--rl-algorithm-spec", "tools/eval/efficiency/algo-eff-miles.json",
                                            *extra), gpu=TWO_ISLANDS)
    args.controller = "local"
    _prepare_rl_args(args)
    return dry_run_plan(args)


def test_the_launcher_forwards_rollout_ahead_to_every_island(monkeypatch):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "k3y")
    plan = _launch(("--rl-max-policy-age", "1", "--rl-rollout-ahead", "1"))
    for island in plan["island_requests"]:
        assert " --rl-rollout-ahead 1" in island["learner_command"]
    assert "--rl-rollout-ahead" not in json.dumps(_launch(("--rl-max-policy-age", "1"))["island_requests"])


@pytest.mark.parametrize(("extra", "message"), [
    (("--rl-rollout-ahead", "1"), r"--rl-max-policy-age >= 1"),
    (("--rl-max-policy-age", "1", "--rl-rollout-ahead", "1", "--rl-overlap-eval"), "no --rl-overlap-eval"),
])
def test_the_launcher_refuses_rollout_ahead_before_any_machine(monkeypatch, extra, message):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "k3y")
    with pytest.raises(ValueError, match=message):
        _launch(extra)


# -- Miles adapter -------------------------------------------------------------------------


def test_miles_argv_has_neither_fully_async_nor_partial_rollout():
    from tests.test_rl_miles_adapter_config import make_config, sub
    from yeto.rl.adapters.miles import carry_over
    from yeto.rl.adapters.miles import config as mc

    partitioned = sub(make_config(colocated=False), "serving", offload_train=False)
    spec = AlgorithmSpec(execution=ExecutionSpec(max_policy_staleness=1),
                         correction=CorrectionSpec(method="tis", tis_clip=2.0, tis_clip_low=0.0))
    strict = mc.translate_run_config(partitioned, spec)
    ahead = mc.translate_run_config(partitioned, spec, max_policy_age=1, rollout_ahead=1)
    assert not ahead.rollout_overlap and ahead.argv == strict.argv
    with pytest.raises(mc.MilesConfigError, match="overlaps training"):
        mc.translate_run_config(partitioned, spec, max_policy_age=1, rollout_ahead=1,
                                async_buffer_rounds=1)
    # the rollout side keeps no carry-over window: each batch is one publication's
    assert carry_over.max_policy_age(SimpleNamespace(yeto_rl_max_policy_age=1)) == 1
    assert carry_over.max_policy_age(SimpleNamespace(yeto_rl_max_policy_age=1,
                                                     yeto_rl_rollout_ahead=1)) == 0


class _Metadata:
    def __init__(self, policy_token):
        self.calls = []
        self.policy_token = policy_token

    def set_policy_token(self, token):
        self.calls.append(("token", token))

    def set_rollout_round(self, rollout_id):
        self.calls.append(("round", rollout_id))

    def take(self, rollout_id):
        from yeto.rl.adapters.miles.rollout_meta_hook import METADATA_SCHEMA

        return {"schema": METADATA_SCHEMA, "rollout_id": rollout_id, "completed": 1, "aborted": 0,
                "groups": [{"group_id": "g0", "sample_ids": ["s0"], "policy_token": self.policy_token,
                            "reward_mean": 0.5, "reward_std": 0.5, "token_count": 3}]}


class _Executor:
    """A rollout executor whose ``get`` finishes when the test releases it (from
    another thread, as a Ray reference completes)."""

    def __init__(self):
        self.started = []
        self.release = threading.Event()

    async def get(self, rollout_id):
        self.started.append(rollout_id)
        await asyncio.get_running_loop().run_in_executor(None, self.release.wait)
        return SimpleNamespace(sample_indices=None, data_ref=f"batch-{rollout_id}")


class _Controller:
    def __init__(self):
        self.prepared = []

    async def prepare_rollout(self, rollout_id):
        self.prepared.append(rollout_id)


def test_miles_pool_sends_the_batch_and_returns_while_it_generates():
    from yeto.rl.adapters.miles import LoopRunner
    from yeto.rl.adapters.miles.rollout import MilesRolloutPool, policy_token

    runner, executor, controller = LoopRunner(), _Executor(), _Controller()
    token = policy_token(3, HASH)
    metadata = _Metadata(token)
    pool = MilesRolloutPool(inference_controller=controller, rollout_executor=executor,
                            metadata=metadata, expected_policy=lambda: (3, HASH), runner=runner)
    pending = pool.start_generate(4, expected_policy_version=token)
    # sent before start_generate returned; still running while the "trainer" uses the loop
    assert executor.started == [4] and controller.prepared == [4]
    assert metadata.calls == [("token", token), ("round", 4)]
    runner.run(asyncio.sleep(0.01))
    assert not pending.task.done()
    executor.release.set()
    batch = pool.finish_generate(pending)
    assert (batch.rollout_id, batch.policy_version, batch.policy_hash) == (4, 3, HASH)
    assert batch.payload.data_ref == "batch-4"
    runner.close()


def test_miles_pool_cancels_a_batch_nobody_trains():
    from yeto.rl.adapters.miles import LoopRunner
    from yeto.rl.adapters.miles.rollout import MilesRolloutPool, policy_token

    runner, executor = LoopRunner(), _Executor()
    pool = MilesRolloutPool(inference_controller=_Controller(), rollout_executor=executor,
                            metadata=_Metadata(policy_token(0, HASH)),
                            expected_policy=lambda: (0, HASH), runner=runner)
    pending = pool.start_generate(1)
    pool.cancel_generate(pending)
    executor.release.set()
    with pytest.raises(asyncio.CancelledError):
        runner.run(pending.task)
    runner.close()


# -- rollout side: explicit round, evidence ------------------------------------------------


def test_the_rollout_round_wins_over_the_token_version(tmp_path, monkeypatch):
    from yeto.rl.adapters.miles import rollout_meta_hook as hook

    sink = f"dir:{tmp_path}"
    monkeypatch.setenv(hook.META_SINK_ENV, sink)
    hook.put_policy_token(f"yeto:3:{HASH}", sink)
    assert hook.current_round_id() == 3 and hook.current_rollout_round() is None
    hook.put_rollout_round(4, sink)
    assert hook.current_round_id() == 4 and hook.current_rollout_round() == 4


def test_ahead_evidence_names_the_round_and_the_generating_policy(tmp_path, monkeypatch):
    from yeto.rl import trajectory_evidence
    from yeto.rl.adapters.miles.full_parameter import evidence
    from yeto.rl.engine.policy_versions import PublishedPolicies

    built = []
    monkeypatch.setattr(trajectory_evidence, "build_trajectory_batch_evidence",
                        lambda groups, **fields: built.append(fields) or "evidence")
    monkeypatch.setattr(trajectory_evidence, "write_trajectory_batch_evidence", lambda *_a: None)
    args = SimpleNamespace(yeto_rl_trajectory_evidence_dir=str(tmp_path), yeto_rl_reward_sha256="b" * 64,
                           yeto_rl_trajectory_evidence_kind="secrlenv",
                           yeto_rl_trajectory_evidence_schema_version=1, yeto_rl_max_policy_age=1)
    published = PublishedPolicies.from_dict({"3": HASH})
    evidence.persist_published_round_evidence(args, [["s"]], f"yeto:3:{HASH}", published, rollout_id=4)
    assert (built[0]["rollout_id"], built[0]["behavior_policy_hash"], built[0]["max_policy_age"]) == (4, HASH, 1)
    with pytest.raises(RuntimeError, match="older than its policy token"):
        evidence.persist_published_round_evidence(args, [["s"]], f"yeto:3:{HASH}", published, rollout_id=2)


def _schema4(rollout_id, version, *, round_hash=OLDER, policy_hash=OLDER, max_policy_age=1):
    from test_rl_trajectory_evidence import _reward_function_sample
    from yeto.rl.trajectory_evidence import build_trajectory_batch_evidence

    return build_trajectory_batch_evidence(
        [[_reward_function_sample(group_index=4, index=0, value=1.0),
          _reward_function_sample(group_index=4, index=1, value=0.0)]],
        rollout_id=rollout_id, behavior_policy_hash=round_hash, reward_contract_hash="b" * 64,
        evidence_kind="reward-function", behavior_policy_of=lambda _s: (version, policy_hash),
        max_policy_age=max_policy_age)


def test_a_batch_generated_ahead_binds_to_the_policy_its_round_trains_from():
    from yeto.rl.trajectory_evidence import bind_round_policy

    written = _schema4(4, 3)  # round 4, generated by v3
    bound = bind_round_policy(written, HASH)
    assert bound.behavior_policy_hash == HASH and bound.input_batch_hash == written.input_batch_hash
    assert [item.behavior_policy_version for item in bound.envelopes] == [3, 3]
    own = _schema4(4, 4, round_hash=HASH, policy_hash=HASH)  # a serial batch names its round
    with pytest.raises(ValueError, match="round's own version"):
        bind_round_policy(own, "d" * 64)
    from test_rl_trajectory_evidence import _reward_function_sample
    from yeto.rl.trajectory_evidence import build_trajectory_batch_evidence

    schema3 = build_trajectory_batch_evidence(
        [[_reward_function_sample(group_index=0, index=0, value=1.0)]], rollout_id=4,
        behavior_policy_hash=OLDER, reward_contract_hash="b" * 64, evidence_kind="reward-function")
    with pytest.raises(ValueError, match="per-trajectory"):
        bind_round_policy(schema3, HASH)


def test_the_session_binds_round_evidence_only_under_rollout_ahead():
    from test_rl_full_parameter_policy_versions import _MonotonicRole, _session

    session = _session(_MonotonicRole("actor", [], []))
    written = _schema4(4, 3)
    session.trajectories = lambda rollout_id: written
    session._driver = SimpleNamespace(rollout_ahead=0, expected_token=f"yeto:4:{HASH}")
    assert session._round_trajectories(4) is written
    session._driver = SimpleNamespace(rollout_ahead=1, expected_token=f"yeto:4:{HASH}")
    assert session._round_trajectories(4).behavior_policy_hash == HASH
    session._driver = SimpleNamespace(rollout_ahead=1, expected_token=f"yeto:5:{HASH}")
    with pytest.raises(RuntimeError, match="trains from published policy v5"):
        session._round_trajectories(4)


# -- a single-island store that holds only the HF exports (S27 G-CO) -------------------------------

SINGLE = ("--rl-single-island-no-sync", "--rl-placement", "fixed-partition", "--rl-rollout-gpus", "1",
          "--rl-algorithm-spec", "tools/eval/efficiency/algo-eff-miles.json")
AHEAD = ("--rl-max-policy-age", "1", "--rl-rollout-ahead", "1")
STORE = ("--rl-checkpoint-store", "modal-volume://vol/run-d", "--rl-export-hf-rounds", "final")


def _single_launch(extra, monkeypatch, *, ahead=True):
    from test_rl_full_parameter_async import _launcher_args
    from yeto.launcher import _prepare_rl_args, dry_run_plan

    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "k3y")
    args = _launcher_args("ports", (*SINGLE, *(AHEAD if ahead else ()), *extra), gpu="modal:2xh100")
    args.controller = "local"
    _prepare_rl_args(args)
    return args, dry_run_plan(args)


def test_rollout_ahead_still_refuses_the_single_island_resume_store(monkeypatch):
    # the store of a single island is its resume store: a ledger and round cuts,
    # which assume nothing in flight at their safe point (ROLLOUT-OVERLAP 2.4)
    with pytest.raises(ValueError, match="resume ledger"):
        _single_launch(STORE, monkeypatch)


def test_rollout_ahead_accepts_a_store_that_holds_only_the_hf_exports(monkeypatch):
    from yeto.launch import rl_ports, storage
    from yeto.launcher import _resume_marker_env

    args, plan = _single_launch((*STORE, "--rl-checkpoint-store-export-only"), monkeypatch)
    (island,) = plan["island_requests"]
    command = island["learner_command"]
    assert " --rl-rollout-ahead 1" in command
    assert "--rl-export-hf-rounds final" in command
    assert "--rl-export-hf-store /root/yeto-checkpoint-store/run-d/hf" in command
    for flag in ("--rl-resume-store", "--rl-cut-every", "--rl-island-cut-store", "--rl-data-position-store"):
        assert flag not in command
    assert rl_ports._check_resume_launch_flags(args) is False
    # no resume store: a Modal re-run is a failure, not a resume, and no reclaim marker
    assert storage.resumes_in_new_container(args) is False
    assert _resume_marker_env(args, "modal") == {}
    args.rl_checkpoint_store_export_only = False
    assert storage.resumes_in_new_container(args) is True and _resume_marker_env(args, "modal")


@pytest.mark.parametrize(("extra", "message"), [
    (("--rl-checkpoint-store-export-only",), "needs --rl-checkpoint-store"),
    (("--rl-checkpoint-store", "modal-volume://vol/run-d", "--rl-checkpoint-store-export-only"),
     "needs --rl-export-hf-rounds"),
    ((*STORE, "--rl-checkpoint-store-export-only", "--rl-cut-every", "4"), "no --rl-cut-every"),
    ((*STORE, "--rl-checkpoint-store-export-only", "--rl-stop-after-rounds", "3"),
     "no --rl-stop-after-rounds"),
    ((*STORE, "--rl-checkpoint-store-export-only", "--rl-resume-allow-config-change"),
     "no --rl-resume-allow-config-change"),
])
def test_the_export_only_store_refuses_what_needs_a_resume_store(monkeypatch, extra, message):
    with pytest.raises(ValueError, match=message):
        _single_launch(extra, monkeypatch, ahead=False)


def test_the_export_only_store_needs_a_single_island_launch():
    from yeto.launch import rl_ports

    args = SimpleNamespace(rl_checkpoint_store="/s", rl_checkpoint_store_export_only=True,
                           rl_single_island_no_sync=False, rl_elastic=False, rl_export_hf_rounds="final")
    with pytest.raises(ValueError, match="needs --rl-single-island-no-sync"):
        rl_ports._check_resume_launch_flags(args)
    args.rl_single_island_no_sync, args.rl_elastic = True, True
    with pytest.raises(ValueError, match="without --rl-elastic"):
        rl_ports._check_resume_launch_flags(args)

"""S27 G-ED: round cuts of verl full-parameter islands (VERL-PARITY #20, #21).

* #20 (V9): a single full-parameter island (``--rl-resume-store``, no syncer)
  that stops after a round cut and resumes from it ends with the same
  parameters, Adam state and LR schedule, bit for bit, as a run that never
  stopped (the judge of the Miles and verl LoRA round cuts);
* #21 (S27 R-2): a synced verl island (``--rl-island-cut-store``) writes its
  own round cuts and never resumes from them (the syncer is authoritative);
* the timed cut interval (``--rl-cut-auto-mtbf-s``) reaches the verl island.

The real cut controller (``ResumeController`` + ``RoundCutCheckpoint``, wired
by ``adapters/verl/round_cuts.py``), the real verl cut trainer
(``VerlCutTrainer``: FSDP2 rank files, digests, resume assertions) and the
real full-parameter policy state, over an in-process torch model with Adam,
behind the fake driver ports. CPU only.
"""
from __future__ import annotations

import hashlib
import json

import pytest
import torch

from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from test_rl_verl_round_cuts import PROMPTS, _events, _Pool
from test_s27_verl_full_parameter_launch import MILES_ONLY, _flags, _island_argv, _prepared, _verl_args
from test_s27_verl_full_parameter_sync import COMPONENT, MAX_FRAGMENT_BYTES, SHAPES, _Model
from test_s27_verl_peer_recovery import _AdamIsland, _assert_same_optimizer, _optimizer_snapshot
from yeto.rl.adapters.verl import cut as vcut
from yeto.rl.adapters.verl import full_parameter_stream as vfs
from yeto.rl.adapters.verl import island_entry, round_cuts
from yeto.rl.contracts import InferencePublicationManifest
from yeto.rl.engine import bridges, recovery
from yeto.rl.engine.algorithm import AlgorithmSpec
from yeto.rl.engine.bridges import LocalOnlySync
from yeto.rl.engine.cut import RestoreExpectation, load_manifest, verify_cut
from yeto.rl.engine.cut_interval import AutoCutInterval, FixedCutInterval
from yeto.rl.engine.driver import EventTape, IslandDriver
from yeto.rl.engine.fake import FakeEngine, FakeTrainerGroup, fake_capabilities
from yeto.rl.engine.ports import PublicationResult
from yeto.rl.engine.recovery import CUT_KEY

ROUNDS = 5
NAME = "base_model.model.layer.lora_A.weight"  # the fake engine's own tensor (the policy is the torch model)


class _FullTrainer(FakeTrainerGroup):
    """The verl trainer port's cut verbs (``VerlCutTrainer`` over ``call_workers``)
    and one Adam round per train step."""

    def __init__(self, engine, island: _AdamIsland, learner_id: int = 0):
        super().__init__(engine)
        self.island = island
        self.learner_id = learner_id
        self.cuts = vcut.VerlCutTrainer(island.call_workers, optimizer_steps_per_round=1)

    def actual_layout(self):
        return self.cuts.actual_layout()

    def layout(self):
        return self.actual_layout()

    def save_cut(self, *, epoch, context):
        return self.cuts.save_cut(epoch=epoch, context=context)

    def restore_cut(self, cut_id, *, epoch, root, expect, shared_filesystem=True):
        return self.cuts.restore_cut(cut_id, epoch=epoch, root=root, expect=expect,
                                     shared_filesystem=shared_filesystem)

    def confirm_resumed_data_cursor(self, cursor):
        self.cuts.confirm_resumed_data_cursor(cursor)

    def train_step(self, batch):
        receipt = super().train_step(batch)
        self.island.train_round(self.learner_id, batch.rollout_id)
        self.cuts.check_first_step(1)
        return receipt


class _ResidentPublisher:
    """Publishes a trainer-resident full-parameter state (digests only), as
    ``VerlFullParameterPublisher`` does without vLLM."""

    def __init__(self, engine):
        self.engine = engine
        self.published = []

    def publish(self, state):
        digest = state.policy_tensor_hash()
        self.published.append((int(state.policy_version), digest))
        self.engine.published = (state.policy_version, digest)  # the fake rollout pool reads it
        manifest = InferencePublicationManifest(
            publication_mode="full", base_policy_version=None, target_policy_version=state.policy_version,
            target_policy_hash=digest,
            target_manifest_hash=hashlib.sha256(f"{state.policy_version}:{digest}".encode()).hexdigest(),
            payload_hash=state.digest.payload_hash, payload_bytes=int(state.digest.payload_bytes), complete=True)
        return PublicationResult(manifest, frozenset(self.engine.members_ids))


class _Synced(bridges.RoundCutHooks):
    """A synced island's sync as the round cuts see it: a ``RoundCutHooks`` that
    is no ``LocalOnlySync`` (periodic cuts only, never a resume). The outer
    merge is the identity here (the local rounds run as in ``LocalOnlySync``)."""

    local_rounds_restart = True  # as FullParameterSync: every process counts its rounds from 0

    def __init__(self, rounds: int, *, restart: bool = True):
        self.inner = LocalOnlySync(rounds)
        self.round_cuts = None
        self.local_rounds_restart = restart

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def start(self, driver):
        return self.inner.start(driver)  # inner.round_cuts is None: no resume


def _plan(store=None, state_dir=None, *, sync="none", **resume):
    config = None
    if store is not None:
        config = {"store": str(store), "state_dir": str(state_dir), "every": 1, "keep": 2,
                  "allow_config_change": False, "stop_after": None, **resume}
    return {"sync": sync, "resume": config, "model_revision": "rev", "lora_targets": "all-linear",
            "parameter_mode": "full",
            "run_config": {"lr": 5e-2, "seed": 1, "total_rounds": ROUNDS, "out_dir": "/x", "parameter_mode": "full"}}


def _island(tmp_path, plan, *, name, rounds=ROUNDS, sync=None, learner_id=0):
    engine = FakeEngine(tensors={NAME: torch.zeros(1, 2)}, step_delta=1.0)
    engine.rollout = _Pool(engine)
    island = _AdamIsland(_Model(seed=0))
    engine.trainer = _FullTrainer(engine, island, learner_id)
    policy_state = vfs.VerlFullParameterLocalState(vfs.VerlFullParameterGroup(island), component=COMPONENT,
                                                   max_fragment_bytes=MAX_FRAGMENT_BYTES)
    sync = sync or LocalOnlySync(rounds)
    resume = round_cuts.resume_wiring_for(plan, runtime_fingerprint="verl@test", sync=sync)
    tape = tmp_path / f"{name}.jsonl"
    driver = IslandDriver(
        learner_id=learner_id, rollout=engine.rollout, trainer=engine.trainer, policy_state=policy_state,
        publisher=_ResidentPublisher(engine), placement=engine.placement, algorithm=AlgorithmSpec(),
        capabilities=fake_capabilities("miles-like"), sync=sync,
        events=EventTape(tape, learner_id), ledger=None if resume is None else resume.ledger)
    checkpoint = round_cuts.wire_round_cuts(driver, resume, plan=plan, algorithm=AlgorithmSpec(),
                                            install_signals=False)
    return driver, island, checkpoint, tape


def _state(island: _AdamIsland) -> dict:
    return {"params": {n: p.detach().clone() for n, p in island.model.named_parameters()},
            **_optimizer_snapshot(island)}


# ----------------------------------------------------------------------------- #20 (V9)
def test_a_full_parameter_island_stopped_at_a_cut_resumes_to_the_uninterrupted_run(tmp_path):
    store = tmp_path / "store"
    driver, _, checkpoint, tape = _island(tmp_path, _plan(store, tmp_path / "a", stop_after=3), name="a")
    round_cuts.run_driver(driver, None)
    pointer = checkpoint.pointer()
    assert pointer["next_rollout_id"] == 3 and pointer["final"] is True
    assert pointer[CUT_KEY]["trigger"] == recovery.MANUAL
    assert [e["rollout_id"] for e in _events(tape, "rl_cut_saved")] == [1, 2, 3]
    # the cut holds every parameter of the model (not a LoRA subset) and its Adam state
    manifest = load_manifest(str(store / "round-cuts"), pointer["cut_id"])
    [rank] = manifest.files
    saved = torch.load(store / "round-cuts" / pointer["cut_id"] / rank.path, weights_only=True)
    assert sorted(saved["model"]) == sorted(SHAPES)
    assert all(set(entry) >= {"exp_avg", "exp_avg_sq", "step"} for entry in saved["optim"]["state"].values())
    assert manifest.rank_summaries[0]["rng_digest"]

    # a fresh container: new state dir, restored from the store's LATEST
    driver, resumed, checkpoint, tape = _island(tmp_path, _plan(store, tmp_path / "b"), name="b")
    round_cuts.run_driver(driver, None)
    [resume] = _events(tape, "rl_resume")
    assert resume["rollout_id"] == 3 and resume["checks"]["policy_hash"] == "same"
    assert _events(tape, "rl_data_cursor_restored")[0]["data_cursor"] == {"sample_offset": 3 * PROMPTS}
    assert checkpoint.pointer()["next_rollout_id"] == ROUNDS
    assert driver.rounds_completed == ROUNDS - 3

    driver, continuous, _, _ = _island(tmp_path, _plan(), name="c")
    round_cuts.run_driver(driver, None)
    a, b = _state(resumed), _state(continuous)
    assert all(torch.equal(a["params"][n], b["params"][n]) for n in SHAPES)
    _assert_same_optimizer(a, b)
    assert resumed.scheduler.last_epoch == continuous.scheduler.last_epoch == ROUNDS
    assert resumed.adam_steps() == continuous.adam_steps() == {float(ROUNDS)}


def test_a_full_parameter_resume_refuses_a_cut_whose_rng_does_not_come_back(tmp_path):
    store = tmp_path / "store"
    driver, _, _, _ = _island(tmp_path, _plan(store, tmp_path / "a", stop_after=2), name="a")
    round_cuts.run_driver(driver, None)
    real = vcut.remote_restore_rank

    def drifting(worker, *args):
        result = real(worker, *args)
        torch.rand(1)  # the RNG moved after the restore
        result["rng_digest"] = vcut.rng_digest(vcut._rng_state())
        return result

    driver, _, _, _ = _island(tmp_path, _plan(store, tmp_path / "b"), name="b")
    trainer = driver.trainer
    trainer.cuts.fns["restore"] = drifting
    with pytest.raises(Exception, match="RNG state differs from the cut"):
        round_cuts.run_driver(driver, None)


# ----------------------------------------------------------------------------- #21 (R-2)
def test_a_synced_full_parameter_island_writes_its_cuts_and_never_resumes(tmp_path):
    island_store = tmp_path / "store" / "island-cuts"
    store = island_store / "island1"
    plan = _plan(store, tmp_path / "a", sync="pool", write_only=True, every=2)
    driver, island, checkpoint, tape = _island(tmp_path, plan, name="a", sync=_Synced(ROUNDS), learner_id=1)
    assert isinstance(checkpoint.interval, FixedCutInterval)
    round_cuts.run_driver(driver, None)
    saved = [e["rollout_id"] for e in _events(tape, "rl_cut_saved")]
    assert saved == [2, 4]  # periodic only: the syncer holds the final policy
    pointer = checkpoint.pointer()
    assert pointer["next_rollout_id"] == 4 and not pointer.get("final")
    expect = RestoreExpectation(algorithm=checkpoint.source.identity, layout={"world": 1, "dp": 1},
                                backend_fingerprint="verl@test", local_step=4, policy_version=4, epoch=0)
    manifest = verify_cut(str(store / "round-cuts"), pointer["cut_id"], expect)
    assert manifest.runtime["backend"] == "verl"
    # a relaunch of the synced island does not resume from its own cut
    driver, fresh, _, tape = _island(tmp_path, _plan(store, tmp_path / "b", sync="pool", write_only=True,
                                                     every=2), name="b", sync=_Synced(ROUNDS), learner_id=1)
    round_cuts.run_driver(driver, None)
    assert _events(tape, "rl_resume") == [] and driver.rounds_completed == ROUNDS
    assert fresh.adam_steps() == {float(ROUNDS)}
    # the earlier process's ledger (rollouts 0..4) moved aside: this process counts from 0 again
    assert len(list((tmp_path / "b").glob("ledger.before-*"))) == 1


def test_without_the_restart_marker_a_relaunched_island_meets_its_old_ledger(tmp_path):
    """Why the ledger moves aside: a ledger restored from the earlier process's
    cut names rollouts 0.. that the new process trains again."""
    from yeto.rl.engine.ledger import LedgerError

    store = tmp_path / "store" / "island-cuts" / "island1"
    plan = _plan(store, tmp_path / "a", sync="pool", write_only=True, every=2)
    driver, _, _, _ = _island(tmp_path, plan, name="a", sync=_Synced(ROUNDS, restart=False), learner_id=1)
    round_cuts.run_driver(driver, None)
    plan = _plan(store, tmp_path / "b", sync="pool", write_only=True, every=2)
    driver, _, _, _ = _island(tmp_path, plan, name="b", sync=_Synced(ROUNDS, restart=False), learner_id=1)
    with pytest.raises(LedgerError, match="behind outer-recorded rollouts"):
        round_cuts.run_driver(driver, None)


def test_miles_island_cuts_of_a_full_parameter_sync_start_an_empty_ledger(tmp_path):
    from types import SimpleNamespace

    from yeto.rl.adapters.miles.entry import resume_wiring_for
    from yeto.rl.engine.full_parameter_sync import FullParameterSync

    store = tmp_path / "store" / "island-cuts" / "island1"
    plan = _plan(store, tmp_path / "a", sync="pool", write_only=True, every=2)
    driver, _, _, _ = _island(tmp_path, plan, name="a", sync=_Synced(ROUNDS), learner_id=1)
    round_cuts.run_driver(driver, None)
    config = {"store": str(store), "state_dir": str(tmp_path / "m"), "every": 2, "keep": 2,
              "allow_config_change": False, "stop_after": None, "write_only": True,
              "runtime_fingerprint": "miles@x"}
    full = FullParameterSync.__new__(FullParameterSync)
    wiring = resume_wiring_for(SimpleNamespace(yeto_rl_resume=config), sync=full, environ={})
    assert wiring.ledger.rebase(0) == [] and len(list((tmp_path / "m").glob("ledger.before-*"))) == 1
    strict = bridges.StrictAvgSync.__new__(bridges.StrictAvgSync)  # a LoRA island keeps its ledger
    kept = resume_wiring_for(SimpleNamespace(yeto_rl_resume={**config, "state_dir": str(tmp_path / "s")}),
                             sync=strict, environ={})
    assert list((tmp_path / "s").glob("ledger.before-*")) == [] and kept.ledger.batch(0) is not None


def test_write_only_cuts_need_a_synced_sync_and_resume_cuts_need_local_only(tmp_path):
    plan = _plan(tmp_path / "s", tmp_path / "a", sync="pool", write_only=True)
    with pytest.raises(ValueError, match="need a synced island's sync"):
        _island(tmp_path, plan, name="a")
    with pytest.raises(ValueError, match="belong to a synced island"):
        round_cuts.resume_wiring_for(_plan(tmp_path / "s", tmp_path / "b", write_only=True),
                                     runtime_fingerprint="verl@test")
    with pytest.raises(ValueError, match="need sync 'none'"):
        round_cuts.resume_wiring_for(_plan(tmp_path / "s", tmp_path / "c", sync="pool"),
                                     runtime_fingerprint="verl@test")


def test_a_timed_interval_reaches_the_verl_cut_controller(tmp_path):
    plan = _plan(tmp_path / "s", tmp_path / "a", stop_after=None)
    plan["resume"].pop("every")
    plan["resume"].update(auto_mtbf_s=600.0, max_interval_s=900.0)
    _, _, checkpoint, _ = _island(tmp_path, plan, name="a")
    assert isinstance(checkpoint.interval, AutoCutInterval)


# ----------------------------------------------------------------------------- launcher -> island
@pytest.mark.parametrize("cadence, expect", [
    (("--rl-cut-every", "2"), {"every": 2}),
    (("--rl-cut-auto-mtbf-s", "600", "--rl-cut-max-interval-s", "900"),
     {"auto_mtbf_s": 600.0, "max_interval_s": 900.0}),
])
def test_a_synced_verl_launch_sends_island_cuts_to_each_island(cadence, expect):
    args = _prepared(_verl_args("--rl-checkpoint-store", "/mnt/store", "--rl-cut-keep", "3", *cadence))
    for learner_id in range(2):
        parsed, ignored = island_entry.parse_args(_island_argv(args, learner_id))
        assert _flags(ignored) <= MILES_ONLY, _flags(ignored) - MILES_ONLY
        config = round_cuts.resume_config(parsed)
        assert config == {"store": f"/mnt/store/island-cuts/island{learner_id}",
                          "state_dir": round_cuts.DEFAULT_STATE_DIR, **expect, "keep": 3,
                          "allow_config_change": False, "stop_after": None, "write_only": True}


def test_a_single_full_parameter_verl_launch_with_a_store_resumes_from_its_cuts():
    args = _prepared(_launcher_args("ports", (
        "--rl-backend", "verl", "--tuning", "full", "--rl-single-island-no-sync",
        "--rl-checkpoint-store", "/mnt/store", "--rl-stop-after-rounds", "2"), gpu="modal:1xh200"))
    parsed, ignored = island_entry.parse_args(_island_argv(args, 0))
    assert _flags(ignored) <= MILES_ONLY, _flags(ignored) - MILES_ONLY
    assert parsed.parameter_mode == "full"
    config = round_cuts.resume_config(parsed)
    assert config["store"] == "/mnt/store" and config["stop_after"] == 2 and "write_only" not in config


@pytest.mark.parametrize("extra, match", [
    (dict(rl_island_cut_store="/s", rl_single_island_no_sync=True), "is a synced island's"),
    (dict(rl_island_cut_store="/s", rl_resume_store="/r"), "exclude each other"),
    (dict(rl_island_cut_store="/s", rl_stop_after_rounds=2), "does not resume from its round cuts"),
    (dict(rl_island_cut_store="/s", rl_sample_role="trainer"), "not wired for a verl buffer trainer"),
    (dict(rl_resume_store="/r", rl_single_island_no_sync=True, rl_cut_every=2, rl_cut_auto_mtbf_s=60.0),
     "exclude each other"),
    (dict(rl_cut_max_interval_s=60.0), "need --rl-resume-store"),
])
def test_the_verl_island_refuses_cut_flags_that_do_not_apply(extra, match):
    from types import SimpleNamespace

    with pytest.raises(ValueError, match=match):
        round_cuts.resume_config(SimpleNamespace(learner_id=0, **extra))


def test_run_fingerprint_names_the_full_parameter_mode():
    plan = _plan()
    assert round_cuts.run_fingerprint_of(plan, AlgorithmSpec())["parameter_mode"] == "full"
    assert json.dumps(round_cuts.run_fingerprint_of(plan, AlgorithmSpec()), sort_keys=True)

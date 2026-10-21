"""S25 N7.6b (CPU): the verl island's round-cut controller.

The real cut controller (ResumeController + RoundCutCheckpoint on LocalOnlySync,
wired by ``adapters.verl.round_cuts``) and the real verl cut trainer
(``VerlCutTrainer``: rank file, digests, resume assertions) over a torch model,
behind the fake driver ports. The policy the driver publishes is read from the
torch model, so the resume's policy-hash check compares real restored weights.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from test_rl_verl_cut import K, _handles, _model, _round

from yeto.rl.adapters.verl import cut as vcut
from yeto.rl.adapters.verl import round_cuts
from yeto.rl.elastic_client import LeaveReason
from yeto.rl.engine import recovery, trainer_remote
from yeto.rl.engine.algorithm import AlgorithmSpec
from yeto.rl.engine.bridges import LocalOnlySync
from yeto.rl.engine.driver import EventTape, IslandDriver
from yeto.rl.engine.fake import FakeEngine, FakeRolloutPool, FakeTrainerGroup, fake_capabilities
from yeto.rl.engine.recovery import CUT_KEY
from yeto.rl.engine.resume_manifest import ResumeAssertionError

NAME = "base_model.model.layer.lora_A.weight"
PROMPTS = 4  # prompts drawn per round
ROUNDS = 5


class _Pool(FakeRolloutPool):
    """The fake pool with the verl data cursor shape (``sample_offset`` only)."""

    def __init__(self, engine):
        super().__init__(engine)
        self.drawn = 0

    def generate(self, rollout_id, **kwargs):
        self.drawn += PROMPTS
        return super().generate(rollout_id, **kwargs)

    def data_cursor(self):
        return {"sample_offset": self.drawn}

    def seek_data_cursor(self, cursor):
        target = int(cursor["sample_offset"])
        if target < self.drawn or (target - self.drawn) % PROMPTS:
            raise ValueError(f"cannot seek {self.drawn} -> {target}")
        self.drawn = target
        return self.data_cursor()


class _TorchTrainer(FakeTrainerGroup):
    """Fake trainer port whose train state is a torch model behind ``VerlCutTrainer``."""

    def __init__(self, engine, handles, *, skip_optimizer_round=None):
        super().__init__(engine)
        self.handles = handles
        worker = SimpleNamespace(module=handles.module, optimizer=handles.optimizer,
                                 lr_scheduler=handles.scheduler)
        self.cuts = vcut.VerlCutTrainer(lambda fn, *a: [fn(worker, *a)], optimizer_steps_per_round=K)
        self.skip_optimizer_round = skip_optimizer_round
        self._sync_policy()

    def _sync_policy(self):
        self.engine.tensors[NAME] = self.handles.module[0].weight.detach()[:1, :2].clone().float()

    def actual_layout(self):
        return self.cuts.actual_layout()

    def layout(self):
        return self.actual_layout()

    def save_cut(self, *, epoch, context):
        return self.cuts.save_cut(epoch=epoch, context=context)

    def restore_cut(self, cut_id, *, epoch, root, expect, shared_filesystem=True):
        manifest = self.cuts.restore_cut(cut_id, epoch=epoch, root=root, expect=expect,
                                         shared_filesystem=shared_filesystem)
        self._sync_policy()
        return manifest

    def confirm_resumed_data_cursor(self, cursor):
        self.cuts.confirm_resumed_data_cursor(cursor)

    def train_step(self, batch):
        receipt = super().train_step(batch)
        if batch.rollout_id == self.skip_optimizer_round:
            self.handles.scheduler.step()  # injected: the round took no optimizer step
        else:
            _round(self.handles, batch.rollout_id)
        self._sync_policy()
        self.cuts.check_first_step(1)
        return receipt


def _plan(store=None, state_dir=None, **resume):
    config = None
    if store is not None:
        config = {"store": str(store), "state_dir": str(state_dir), "every": 1, "keep": 2,
                  "allow_config_change": False, "stop_after": None, **resume}
    return {"sync": "none", "resume": config, "model_revision": "rev", "lora_targets": "all-linear",
            "run_config": {"lr": 1e-2, "seed": 1, "total_rounds": ROUNDS, "out_dir": "/x"}}


def _island(tmp_path, plan, *, name, rounds=ROUNDS, **trainer_kwargs):
    engine = FakeEngine(tensors={NAME: torch.zeros(1, 2)}, step_delta=1.0)
    engine.rollout = _Pool(engine)
    handles = _handles(_model())
    engine.trainer = _TorchTrainer(engine, handles, **trainer_kwargs)
    resume = round_cuts.resume_wiring_for(plan, runtime_fingerprint="verl@test")
    tape = tmp_path / f"{name}.jsonl"
    driver = IslandDriver(
        learner_id=0, rollout=engine.rollout, trainer=engine.trainer, policy_state=engine.policy_state,
        publisher=engine.publisher, placement=engine.placement, algorithm=AlgorithmSpec(),
        capabilities=fake_capabilities("miles-like"), sync=LocalOnlySync(rounds),
        events=EventTape(tape, 0), ledger=None if resume is None else resume.ledger)
    checkpoint = round_cuts.wire_round_cuts(driver, resume, plan=plan, algorithm=AlgorithmSpec(),
                                            install_signals=False)
    return driver, handles, checkpoint, resume, tape


def _events(path, kind):
    import json

    return [e for e in map(json.loads, path.read_text().splitlines()) if e.get("event") == kind]


def _params(handles):
    return {n: p.detach().clone() for n, p in handles.module.named_parameters()}


def test_stop_after_then_resume_equals_continuous_run(tmp_path):
    store = tmp_path / "store"
    driver, _, checkpoint, _, tape = _island(tmp_path, _plan(store, tmp_path / "a", stop_after=3), name="a")
    round_cuts.run_driver(driver, None)
    pointer = checkpoint.pointer()
    assert pointer["next_rollout_id"] == 3 and pointer["final"] is True
    assert pointer[CUT_KEY]["trigger"] == recovery.MANUAL
    assert [e["rollout_id"] for e in _events(tape, "rl_cut_saved")] == [1, 2, 3]

    # a fresh container: new state dir, restored from the store's LATEST
    driver, resumed, checkpoint, _, tape = _island(tmp_path, _plan(store, tmp_path / "b"), name="b")
    round_cuts.run_driver(driver, None)
    [resume] = _events(tape, "rl_resume")
    assert resume["rollout_id"] == 3 and resume["checks"]["policy_hash"] == "same"
    assert _events(tape, "rl_data_cursor_restored")[0]["data_cursor"] == {"sample_offset": 3 * PROMPTS}
    assert checkpoint.pointer()["next_rollout_id"] == ROUNDS
    assert driver.rounds_completed == ROUNDS - 3

    driver, continuous, _, _, _ = _island(tmp_path, _plan(), name="c")
    round_cuts.run_driver(driver, None)
    a, b = _params(resumed), _params(continuous)
    assert all(torch.equal(a[n], b[n]) for n in a)
    assert resumed.scheduler.last_epoch == continuous.scheduler.last_epoch == ROUNDS


def test_cut_every_and_keep(tmp_path):
    store = tmp_path / "store"
    driver, _, _, _, tape = _island(tmp_path, _plan(store, tmp_path / "a", every=2, keep=1), name="a")
    round_cuts.run_driver(driver, None)
    assert [e["rollout_id"] for e in _events(tape, "rl_cut_saved")] == [2, 4, 5]  # every 2, and the end


def test_first_resumed_round_without_an_optimizer_step_is_refused(tmp_path):
    store = tmp_path / "store"
    driver, _, _, _, _ = _island(tmp_path, _plan(store, tmp_path / "a", stop_after=2), name="a")
    round_cuts.run_driver(driver, None)
    driver, _, _, _, _ = _island(tmp_path, _plan(store, tmp_path / "b"), name="b", skip_optimizer_round=2)
    with pytest.raises(Exception) as info:
        round_cuts.run_driver(driver, None)
    chain, error = [], info.value
    while error is not None:
        chain.append(error)
        error = error.__cause__ or error.__context__
    assert any(isinstance(e, ResumeAssertionError) and "Adam step" in str(e) for e in chain)


def test_scheduler_stop_cuts_with_scheduled_and_resumes(tmp_path):
    """The head's trainer scheduler: run 2 rounds, stop (STOPPED -> SCHEDULED cut);
    the next incarnation restores that cut."""
    store = tmp_path / "store"
    driver, _, checkpoint, resume, tape = _island(tmp_path, _plan(store, tmp_path / "a"), name="a")
    files = trainer_remote.LocalStoreFiles(resume.controller.store)
    writer = trainer_remote.CommandWriter(files, 0)
    writer.send(trainer_remote.RUN, rounds=2)
    writer.send(trainer_remote.STOP, reason=LeaveReason.STOPPED)
    env = {trainer_remote.CONTROL_ENV: "1", trainer_remote.INCARNATION_ENV: "0",
           trainer_remote.RESUME_ENV: "", trainer_remote.POLL_ENV: "0.01"}
    round_cuts.run_driver(driver, resume, environ=env)
    pointer = checkpoint.pointer()
    assert pointer["next_rollout_id"] == 2 and pointer[CUT_KEY]["trigger"] == recovery.SCHEDULED
    phases = [e["phase"] for e in _events(tape, trainer_remote.EVENT)]
    assert phases == ["started", "ready", "trained", "stopped"]

    driver, _, _, _, tape = _island(tmp_path, _plan(store, tmp_path / "b"), name="b")
    round_cuts.run_driver(driver, None)
    assert _events(tape, "rl_resume")[0]["rollout_id"] == 2


def test_commanded_run_needs_round_cuts(tmp_path):
    driver, _, _, _, _ = _island(tmp_path, _plan(), name="a")
    with pytest.raises(RuntimeError, match="needs round cuts"):
        round_cuts.run_driver(driver, None, environ={trainer_remote.CONTROL_ENV: "1"})


def _args(**kw):
    base = dict(rl_resume_store=None, rl_resume_state_dir=None, rl_cut_every=None, rl_cut_keep=None,
                rl_stop_after_rounds=None, rl_resume_allow_config_change=False,
                rl_single_island_no_sync=True, rl_max_policy_age=0)
    return SimpleNamespace(**{**base, **kw})


def test_resume_config_from_flags():
    assert round_cuts.resume_config(_args()) is None
    config = round_cuts.resume_config(_args(rl_resume_store="/s", rl_cut_every=2, rl_stop_after_rounds=3))
    assert config == {"store": "/s", "state_dir": round_cuts.DEFAULT_STATE_DIR, "every": 2, "keep": 2,
                      "allow_config_change": False, "stop_after": 3}
    with pytest.raises(ValueError, match="need --rl-resume-store"):
        round_cuts.resume_config(_args(rl_cut_every=2))
    with pytest.raises(ValueError, match="--rl-single-island-no-sync"):
        round_cuts.resume_config(_args(rl_resume_store="/s", rl_single_island_no_sync=False))
    # S26 E9.2: the fully_async path (--rl-max-policy-age > 0) cuts too
    assert round_cuts.resume_config(_args(rl_resume_store="/s", rl_max_policy_age=2))["store"] == "/s"
    with pytest.raises(ValueError, match="must be >= 1"):
        round_cuts.resume_config(_args(rl_resume_store="/s", rl_cut_keep=0))


def test_buffer_trainer_store_is_not_a_cut_store():
    # S25 C2 x N7.6b: a verl buffer trainer's --rl-resume-store is its sample buffer
    trainer = dict(rl_sample_role="trainer", rl_resume_store="/s", rl_max_policy_age=1)
    assert round_cuts.resume_config(_args(**trainer)) is None
    with pytest.raises(ValueError, match="not wired for a verl buffer trainer"):
        round_cuts.resume_config(_args(**trainer, rl_cut_every=2))


def test_changed_run_config_is_refused(tmp_path):
    store = tmp_path / "store"
    driver, _, _, _, _ = _island(tmp_path, _plan(store, tmp_path / "a", stop_after=2), name="a")
    round_cuts.run_driver(driver, None)
    plan = _plan(store, tmp_path / "b")
    plan["run_config"] = {**plan["run_config"], "lr": 2e-2, "total_rounds": 9}  # rounds may change
    driver, _, _, _, _ = _island(tmp_path, plan, name="b")
    with pytest.raises(Exception, match="run configuration differs"):
        round_cuts.run_driver(driver, None)

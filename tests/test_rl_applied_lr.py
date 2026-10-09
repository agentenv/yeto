"""fix-decoupled-lr-schedule 2.2: applied learning rate + zero-LR invariant (legacy path).

The ports-path equivalents live in ``test_rl_engine_driver.py`` (2.1).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from yeto.protocol import FinalManifest
from yeto.rl import applied_lr, grad_audit
from yeto.rl.core import PolicySnapshot, StrictRlInvariantError, require_nonzero_learning_rate
from yeto.rl.decoupled import BroadcastBatch
from yeto.rl.adapters.miles.legacy.engine import DecoupledMilesPolicySync

import test_rl_core as core_tests
import test_rl_decoupled as decoupled_tests


class _Optimizer:
    def __init__(self, *lrs):
        self.param_groups = [{"lr": lr, "params": []} for lr in lrs]


# ---------------------------------------------------------------------------
# reading and transporting the applied LR
# ---------------------------------------------------------------------------
def test_optimizer_lr_reads_param_groups_and_chained_children():
    assert applied_lr.optimizer_lr(_Optimizer(1e-5, 1e-5)) == 1e-5
    chained = SimpleNamespace(chained_optimizers=[_Optimizer(0.0), _Optimizer(2e-5)])
    assert applied_lr.optimizer_lr(chained) == 2e-5
    assert applied_lr.optimizer_lr(_Optimizer(0.0)) == 0.0
    with pytest.raises(applied_lr.AppliedLrError):
        applied_lr.optimizer_lr(SimpleNamespace(param_groups=[]))


def test_record_and_collect_round_trip(tmp_path):
    args = SimpleNamespace(**{applied_lr.APPLIED_LR_DIR_ATTR: str(tmp_path)})
    applied_lr.record(args, 4, 0, _Optimizer(1e-5), is_writer=lambda: True)
    applied_lr.record(args, 4, 1, _Optimizer(5e-6), is_writer=lambda: True)
    assert applied_lr.record(args, 4, 2, _Optimizer(1.0), is_writer=lambda: False) == 1.0
    assert applied_lr.collect(tmp_path, 4, 2) == (1e-5, 5e-6)
    assert list(tmp_path.iterdir()) == []  # consumed
    with pytest.raises(applied_lr.AppliedLrError, match="optimizer step 0"):
        applied_lr.collect(tmp_path, 4, 1)
    # not armed: nothing recorded
    assert applied_lr.record(SimpleNamespace(), 0, 0, _Optimizer(1e-5)) is None


def test_combined_hook_records_lr_and_runs_the_grad_audit(tmp_path, monkeypatch):
    audited = []
    monkeypatch.setattr(
        grad_audit, "before_train_step", lambda *a: audited.append(a[1:3])
    )
    args = SimpleNamespace(
        **{
            applied_lr.APPLIED_LR_DIR_ATTR: str(tmp_path / "lr"),
            grad_audit.GRAD_AUDIT_DIR_ATTR: str(tmp_path / "audit"),
        }
    )
    applied_lr.before_train_step(args, 2, 0, [], _Optimizer(3e-6), None)
    assert audited == [(2, 0)]
    assert applied_lr.collect(tmp_path / "lr", 2, 1) == (3e-6,)
    # grad audit not configured: only the LR is recorded
    del args.yeto_rl_grad_audit_dir
    applied_lr.before_train_step(args, 3, 0, [], _Optimizer(3e-6), None)
    assert audited == [(2, 0)]
    assert applied_lr.collect(tmp_path / "lr", 3, 1) == (3e-6,)


def test_learner_arms_one_combined_hook_with_grad_audit(tmp_path, monkeypatch):
    from yeto.rl import learner

    args = SimpleNamespace(
        audit_dir=str(tmp_path / "audit"),
        event_tape=str(tmp_path / "tape" / "events.jsonl"),
        learner_id=1,
    )
    miles_args = SimpleNamespace(custom_megatron_before_train_step_hook_path=None)
    monkeypatch.setenv(grad_audit.GRAD_AUDIT_ENV, "1")
    assert learner._configure_applied_lr(args, miles_args, "legacy", dense_full=False)
    assert learner._configure_grad_audit(args, miles_args, "legacy")
    # neither overwrote the other: the combined hook runs both
    assert miles_args.custom_megatron_before_train_step_hook_path == applied_lr.HOOK_PATH
    assert miles_args.yeto_rl_grad_audit_dir == str(tmp_path / "audit")
    lr_dir = getattr(miles_args, applied_lr.APPLIED_LR_DIR_ATTR)
    assert lr_dir.startswith(str(tmp_path / "tape" / "applied-lr-1-"))

    # ports reads the LR in its train_one_step recorder; eval-only never trains
    assert not learner._configure_applied_lr(args, SimpleNamespace(), "ports", dense_full=False)
    eval_args = SimpleNamespace(**vars(args), eval_only=True)
    assert not learner._configure_applied_lr(eval_args, SimpleNamespace(), "legacy", dense_full=False)
    with pytest.raises(ValueError, match="conflicts"):
        learner._configure_applied_lr(
            args,
            SimpleNamespace(custom_megatron_before_train_step_hook_path="x.y"),
            "legacy",
            dense_full=False,
        )


def test_legacy_round_stats_carry_the_applied_lr(tmp_path, monkeypatch):
    hook = core_tests._round_stats_hook(tmp_path, monkeypatch)
    hook.args.yeto_rl_applied_lr_dir = str(tmp_path / "lr")
    applied_lr.record(hook.args, 3, 0, _Optimizer(7.5e-6), is_writer=lambda: True)
    stats = hook._round_stats(
        3,
        core_tests._round_stats_data_pack([1.0, -1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
        core_tests._ROUND_TRAIN_STATE,
    )
    assert stats.applied_lr == 7.5e-6 and stats.applied_lrs == (7.5e-6,)
    with pytest.raises(applied_lr.AppliedLrError):  # consumed; a missing record fails
        hook._round_stats(
            3,
            core_tests._round_stats_data_pack([1.0, -1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
            core_tests._ROUND_TRAIN_STATE,
        )


def test_invariant_message_names_round_and_lr():
    stats = replace(
        decoupled_tests._stats(4), applied_lr=0.0, applied_lrs=(1e-5, 0.0)
    )
    with pytest.raises(StrictRlInvariantError) as info:
        require_nonzero_learning_rate(stats, final_round=False)
    assert info.value.metric == "zero_lr_before_final_round"
    assert "local round 5" in str(info.value) and "learning rate 0.0" in str(info.value)
    require_nonzero_learning_rate(stats, final_round=True)
    require_nonzero_learning_rate(replace(stats, applied_lr=None, applied_lrs=None), final_round=False)
    with pytest.raises(ValueError):
        replace(stats, applied_lr=1e-5)  # must be the minimum of the per-step values


# ---------------------------------------------------------------------------
# legacy strict-avg bridge
# ---------------------------------------------------------------------------
def _zero_lr_stats():
    return replace(
        core_tests._training_progress_stats(grad_norm=0.5, nonzero_advantage_count=3),
        applied_lr=0.0,
        applied_lrs=(0.0,),
    )


def test_legacy_strict_zero_lr_before_last_round_fails_before_submission(tmp_path):
    hook, actor, events = core_tests._submitting_hook(tmp_path, _zero_lr_stats())
    hook.args.num_rollout = 3  # local round 1 of 3: the island keeps training
    with pytest.raises(StrictRlInvariantError) as info:
        asyncio.run(hook.after_local_train(rollout_id=0, actor_model=actor, rollout_data={}))
    assert info.value.metric == "zero_lr_before_final_round"
    assert "local round 1" in str(info.value)
    assert "submit" not in events and "release" not in events
    tape = core_tests._tape_events(tmp_path)
    assert [e["event"] for e in tape] == ["rl_strict_failure"]
    assert tape[0]["metric"] == "zero_lr_before_final_round"


def test_legacy_strict_last_round_with_zero_lr_is_not_a_failure(tmp_path):
    hook, actor, events = core_tests._submitting_hook(tmp_path, _zero_lr_stats())
    assert hook.args.num_rollout == 1  # local round 1 of 1
    asyncio.run(hook.after_local_train(rollout_id=0, actor_model=actor, rollout_data={}))
    assert events == ["export", "submit", "release", "wait", "apply"]


# ---------------------------------------------------------------------------
# legacy decoupled bridge
# ---------------------------------------------------------------------------
def _decoupled_hook(tmp_path, *, finalizing, budget=None):
    args = decoupled_tests._checkpoint_args(tmp_path)
    args.yeto_rl_event_tape = str(tmp_path / "events.jsonl")
    args.yeto_rl_learner_id = 0
    args.yeto_rl_learner_budget_steps = budget
    current = decoupled_tests._state(0)
    snapshot = PolicySnapshot.create(0, current, (0, 0))
    args.yeto_rl_policy_token = snapshot.token
    calls = []

    class Bridge:
        fragment_versions = (0, 0)
        final_payload_bytes_received = 0
        client = SimpleNamespace(close=lambda: calls.append("close"))

        def __init__(self):
            self.finalizing = finalizing

        def drain_broadcasts(self, local, **_):
            calls.append("drain")
            return BroadcastBatch(local, ())

        def submit_ready(self, _local, **_):
            calls.append("submit")
            return ()

        def consolidate_budget(self, *_a, **_k):
            calls.append("consolidate")
            raise RuntimeError("stop here")

    class Actor:
        async def export_trainable_state(self):
            return object()

    hook = DecoupledMilesPolicySync(args)
    hook.actor_model = Actor()
    hook.bridge = Bridge()
    hook.current = current
    hook.snapshot = snapshot
    hook._canonical_at_progress = lambda _exported, version: decoupled_tests._state(version)
    hook._round_stats = lambda *_a, **_k: replace(
        decoupled_tests._stats(0), applied_lr=0.0, applied_lrs=(0.0,)
    )
    hook._record_local_round = lambda *_a, **_k: calls.append("record")
    hook._save_progress = lambda *_a, **_k: None

    async def publish(_snapshot):
        pass

    async def finish(*, policy_version, stats):
        calls.append("final_cut")
        return True

    hook._publish_snapshot = publish
    hook._finish = finish
    return hook, calls


def test_legacy_decoupled_zero_lr_before_final_cut_fails(tmp_path):
    hook, calls = _decoupled_hook(tmp_path, finalizing=False)
    with pytest.raises(StrictRlInvariantError) as info:
        asyncio.run(
            hook.after_local_train(rollout_id=0, actor_model=hook.actor_model, rollout_data=object())
        )
    assert info.value.metric == "zero_lr_before_final_round"
    assert calls == ["close"]  # nothing drained, submitted or recorded
    tape = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [e["metric"] for e in tape if e["event"] == "rl_strict_failure"] == [
        "zero_lr_before_final_round"
    ]


def test_legacy_decoupled_zero_lr_once_the_final_cut_is_known_is_not_a_failure(tmp_path):
    hook, calls = _decoupled_hook(tmp_path, finalizing=True)
    assert asyncio.run(
        hook._after_local_train(rollout_id=0, actor_model=hook.actor_model, rollout_data=object())
    )
    assert calls == ["record", "final_cut"]


def test_legacy_decoupled_round_exhausting_the_budget_is_final(tmp_path):
    hook, calls = _decoupled_hook(tmp_path, finalizing=False, budget=1)
    with pytest.raises(RuntimeError, match="stop here"):  # got past the invariant
        asyncio.run(
            hook._after_local_train(rollout_id=0, actor_model=hook.actor_model, rollout_data=object())
        )
    assert calls == ["consolidate"]


# ---------------------------------------------------------------------------
# Review F2/F3: legacy rejects LR-schedule overrides; zero inner LR fails at startup.


@pytest.mark.parametrize("flag", ["--lr-decay-style", "--lr-decay-iters", "--lr-warmup-iters", "--min-lr"])
@pytest.mark.parametrize("form", ["split", "equals"])
def test_legacy_rejects_lr_schedule_overrides_like_ports(flag, form):
    from yeto.rl.adapters.miles import config as mc
    from yeto.rl.adapters.miles.lr_schedule import LR_SCHEDULE_FLAGS
    from yeto.rl.adapters.miles.island_entry import _reject_lr_schedule_overrides

    extra = [flag, "5"] if form == "split" else [f"{flag}=5"]
    assert flag in LR_SCHEDULE_FLAGS and flag in mc.ADAPTER_OWNED_FLAGS
    with pytest.raises(ValueError, match=f"{flag} is owned by yeto's LR schedule"):
        _reject_lr_schedule_overrides(extra)
    with pytest.raises(mc.MilesConfigError, match=flag):
        mc.check_extra_argv(extra)


def test_legacy_allows_unrelated_extra_argv():
    from yeto.rl.adapters.miles.island_entry import _reject_lr_schedule_overrides

    _reject_lr_schedule_overrides(["--lr", "1e-5", "--clip-grad", "1.0", "--lr-decay-foo", "x"])


def test_legacy_build_path_calls_the_lr_override_check():
    import inspect

    from yeto.rl import learner

    source = inspect.getsource(learner)
    check = source.index("_reject_lr_schedule_overrides(extra_argv)")
    assert check < source.index("miles_argv.extend(extra_argv)")


@pytest.mark.parametrize("lr", ["0", "0.0", "-0.00001"])
def test_learner_rejects_non_positive_inner_lr_before_start(lr, capsys):
    from test_rl_engine_selection import _learner_argv

    from yeto.rl import learner

    argv = _learner_argv()
    argv[argv.index("--inner-lr") + 1] = lr
    with pytest.raises(SystemExit):
        learner.parse_args(argv)
    assert "--inner-lr must be > 0" in capsys.readouterr().err
    argv += ["--eval-only"]  # eval-only never steps the optimizer
    try:
        learner.parse_args(argv)
    except SystemExit:
        assert "--inner-lr must be > 0" not in capsys.readouterr().err


@pytest.mark.parametrize("lr", ["0", "-0.00001"])
def test_launcher_rejects_non_positive_inner_lr_before_launch(lr):
    from test_rl_engine_selection import _cli

    from yeto.launcher import _prepare_rl_args

    args = _cli(("--inner-lr", lr))
    with pytest.raises(ValueError, match=r"--inner-lr > 0"):
        _prepare_rl_args(args)

"""fix-decoupled-lr-schedule 2.2: applied learning rate + zero-LR invariant.

The driver-level checks live in ``test_rl_engine_driver.py`` (2.1).
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from yeto.rl import applied_lr, grad_audit
from yeto.rl.core import StrictRlInvariantError, require_nonzero_learning_rate

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
# Review F2/F3: LR-schedule overrides are refused; zero inner LR fails at startup.


@pytest.mark.parametrize("flag", ["--lr-decay-style", "--lr-decay-iters", "--lr-warmup-iters", "--min-lr"])
@pytest.mark.parametrize("form", ["split", "equals"])
def test_ports_rejects_lr_schedule_overrides(flag, form):
    from yeto.rl.adapters.miles import config as mc
    from yeto.rl.adapters.miles.lr_schedule import LR_SCHEDULE_FLAGS

    extra = [flag, "5"] if form == "split" else [f"{flag}=5"]
    assert flag in LR_SCHEDULE_FLAGS and flag in mc.ADAPTER_OWNED_FLAGS
    with pytest.raises(mc.MilesConfigError, match=flag):
        mc.check_extra_argv(extra)


@pytest.mark.parametrize("lr", ["0", "0.0", "-0.00001"])
def test_learner_rejects_non_positive_inner_lr_before_start(lr, capsys):
    from test_rl_engine_selection import _learner_argv

    from yeto.rl.adapters.miles import island_entry as learner

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

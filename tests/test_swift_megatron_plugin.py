"""Hook-layer tests for yeto.swift_megatron.swift_plugin without Megatron/NPU."""

import types

import pytest

from yeto.swift_megatron import swift_plugin as sp

ENV = {
    "YETO_DILOCO_SYNCER": "tcp://syncer:7000",
    "YETO_DILOCO_LEARNER_ID": "1",
    "YETO_DILOCO_NUM_LEARNERS": "3",
    "YETO_DILOCO_FRAGMENTS": "4",
    "YETO_DILOCO_SYNC_INTERVAL_STEPS": "2",
    "YETO_DILOCO_WIRE_DTYPE": "bf16",
}


class FakeSync:
    instances = []

    def __init__(self, model_chunks, optimizer, cfg, *, resume_dir=None, resume_iteration=None):
        self.model_chunks, self.optimizer, self.cfg = model_chunks, optimizer, cfg
        self.resume_dir, self.resume_iteration = resume_dir, resume_iteration
        self.saves = []
        self.calls = []
        self.closed = 0
        FakeSync.instances.append(self)

    def after_optimizer_step(self, iteration, target_tokens, *, update_successful=True):
        self.calls.append((iteration, target_tokens))

    def save_state(self, output_dir, *, iteration, save_anchors=False):
        self.saves.append((output_dir, iteration, save_anchors))
        return True

    def close(self):
        self.closed += 1


class FakeTrainer:
    """Mimics BaseMegatronTrainer.run_train_step / train / finalize_training."""

    def __init__(self, steps, start_iteration=0, dp_size=2):
        # steps: list of (micro-batch token counts, update_successful)
        self.args = types.SimpleNamespace(train_iters=start_iteration + len(steps), callbacks=[])
        self.state = types.SimpleNamespace(iteration=start_iteration)
        self.wrapped_models = ["chunk0", "chunk1"]
        self.optimizer = object()
        self.steps = list(steps)
        self.dp_size = dp_size
        self.optimizer_steps = 0
        self.callbacks = []

    def train_step(self, it):
        mbs, ok = self.steps.pop(0)
        self.optimizer_steps += 1
        return [{"loss": [1.0, float(n)]} for n in mbs], 1.0, ok

    def call_event(self, event):
        for cb in self.callbacks:
            getattr(cb, event)()

    def train(self):
        self.call_event("on_train_begin") if hasattr(sp._CallbackBase, "on_train_begin") else None
        while self.state.iteration < self.args.train_iters:
            self.call_event("on_step_begin") if hasattr(sp._CallbackBase, "on_step_begin") else None
            self.train_step(None)
            self.state.iteration += 1
            self.call_event("on_step_end")
        self.call_event("on_train_end")


@pytest.fixture
def make(monkeypatch):
    FakeSync.instances.clear()
    for k in list(__import__("os").environ):
        if k.startswith(sp.ENV_PREFIX):
            monkeypatch.delenv(k)
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(sp.YetoDiLoCoCallback, "sync_factory", staticmethod(FakeSync))

    def _make(steps, **kw):
        trainer = FakeTrainer(steps, **kw)
        monkeypatch.setattr(sp.YetoDiLoCoCallback, "token_reducer",
                            staticmethod(lambda n: n * trainer.dp_size))
        cb = sp.YetoDiLoCoCallback(trainer)
        trainer.callbacks.append(cb)
        return trainer, cb

    return _make


def test_called_once_per_optimizer_step_with_global_tokens(make):
    trainer, cb = make([([3, 5], True), ([2], True), ([4, 4, 4], True)])
    trainer.train()
    sync = FakeSync.instances[0]
    assert sync.calls == [(1, 16), (2, 4), (3, 24)]
    assert len(sync.calls) == trainer.optimizer_steps
    assert sync.model_chunks == ["chunk0", "chunk1"] and sync.optimizer is trainer.optimizer
    assert sync.cfg == sp.DiLoCoHookConfig("tcp://syncer:7000", 1, 3, 4, 2, "bf16", None)
    assert sync.closed == 1


def test_training_ends_at_the_step_that_installed_the_final_cut(make):
    trainer, cb = make([([1], True)] * 5)
    sync = FakeSync.instances[0]
    original = sync.after_optimizer_step

    def step(iteration, target_tokens, **kw):
        original(iteration, target_tokens, **kw)
        sync.finished = iteration == 2

    sync.after_optimizer_step = step
    trainer.train()
    assert [c[0] for c in sync.calls] == [1, 2] and trainer.args.train_iters == 2 and sync.closed == 1


def test_skipped_step_still_calls_with_zero_tokens(make):
    trainer, cb = make([([3], True), ([7], False), ([1], True)])
    trainer.train()
    assert FakeSync.instances[0].calls == [(1, 6), (2, 0), (3, 2)]


def test_non_last_pipeline_stage_contributes_zero_locally(make):
    trainer, cb = make([([], True)])
    trainer.train()
    assert FakeSync.instances[0].calls == [(1, 0)]


def test_resume_continues_iteration(make):
    trainer, cb = make([([1], True), ([1], True)], start_iteration=100)
    trainer.train()
    assert FakeSync.instances[0].calls == [(101, 2), (102, 2)]


def test_cold_start_has_no_resume_dir(make):
    make([([1], True)])
    assert FakeSync.instances[0].resume_dir is None and FakeSync.instances[0].resume_iteration == 0


def test_resume_reads_the_record_of_the_loaded_checkpoint(make, tmp_path):
    from yeto.swift_megatron.sync import RESUME_STATE_FILE

    ckpt = tmp_path / "checkpoint-15"
    ckpt.mkdir()
    trainer = FakeTrainer([([1], True)], start_iteration=15)
    trainer.args.mcore_model = str(ckpt)
    cfg = sp.DiLoCoHookConfig.from_env()
    assert sp.resume_dir_for(cfg, trainer.args) is None  # no record there: cold start
    (ckpt / RESUME_STATE_FILE).write_text("{}")
    assert sp.resume_dir_for(cfg, trainer.args) == str(ckpt)
    explicit = sp.DiLoCoHookConfig(**{**cfg.__dict__, "resume_dir": "/elsewhere"})
    assert sp.resume_dir_for(explicit, trainer.args) == "/elsewhere"
    _, cb = make([([1], True)], start_iteration=15)
    assert FakeSync.instances[-1].resume_iteration == 15


def test_on_save_records_the_sync_state_next_to_the_checkpoint(make, monkeypatch):
    monkeypatch.setenv("YETO_DILOCO_SAVE_ANCHORS", "1")
    trainer, cb = make([([1], True), ([1], True)])
    trainer.args.output_dir = "/runs/x"
    trainer.state.iteration = 2  # the step whose checkpoint was just written
    cb.on_save()
    cb.on_save(output_dir="/runs/x/checkpoint-2-explicit")
    assert FakeSync.instances[0].saves == [("/runs/x/checkpoint-2", 2, True),
                                           ("/runs/x/checkpoint-2-explicit", 2, True)]
    cb.on_train_end()
    cb.on_save()  # closed: nothing more to record
    assert len(FakeSync.instances[0].saves) == 2


def test_close_is_idempotent(make):
    trainer, cb = make([([1], True)])
    trainer.train()
    cb.close()
    cb.on_train_end()
    assert FakeSync.instances[0].closed == 1


def test_step_end_without_train_step_raises(make):
    trainer, cb = make([([1], True)])
    trainer.state.iteration += 1
    with pytest.raises(RuntimeError):
        cb.on_step_end()


def test_disable_flag(make, monkeypatch):
    monkeypatch.setenv("YETO_DILOCO_DISABLE", "1")
    trainer, cb = make([([1], True)])
    trainer.train()
    assert FakeSync.instances == [] and cb.sync is None


@pytest.mark.parametrize("key,value", [
    ("YETO_DILOCO_SYNCER", ""),
    ("YETO_DILOCO_LEARNER_ID", "3"),
    ("YETO_DILOCO_SYNC_INTERVAL_STEPS", "x"),
    ("YETO_DILOCO_HMAC_KEY_PATH", "/nonexistent/key"),
])
def test_bad_config_fails_at_construction(make, monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError):
        make([([1], True)])


def test_local_loss_tokens():
    assert sp.local_loss_tokens([{"loss": [0.5, 3.0]}, {"loss": [0.1, 2]}, {}, None]) == 5
    assert sp.local_loss_tokens([]) == 0


def test_register_into_swift_map(monkeypatch):
    callbacks_mod = types.ModuleType("swift.megatron.callbacks")
    callbacks_mod.megatron_callbacks_map = {}
    base_mod = types.ModuleType("swift.megatron.trainers.base")

    class BaseMegatronTrainer:
        def __init__(self, args):
            self.seen = list(args.callbacks)

    base_mod.BaseMegatronTrainer = BaseMegatronTrainer
    for name, mod in {
        "swift": types.ModuleType("swift"),
        "swift.megatron": types.ModuleType("swift.megatron"),
        "swift.megatron.callbacks": callbacks_mod,
        "swift.megatron.trainers": types.ModuleType("swift.megatron.trainers"),
        "swift.megatron.trainers.base": base_mod,
    }.items():
        monkeypatch.setitem(__import__("sys").modules, name, mod)
    assert sp.register(auto_attach=True)
    assert callbacks_mod.megatron_callbacks_map["yeto_diloco"] is sp.YetoDiLoCoCallback
    t = BaseMegatronTrainer(types.SimpleNamespace(callbacks=["print", "default_flow"]))
    assert t.seen == ["yeto_diloco", "print", "default_flow"]
    t = BaseMegatronTrainer(types.SimpleNamespace(callbacks=["yeto_diloco", "print"]))
    assert t.seen == ["yeto_diloco", "print"]


def test_composes_with_class_level_train_step_patch(make, monkeypatch):
    # Production plugins (e.g. ascend_stage_timer) replace the class-level
    # train_step at import time, before the trainer and callback exist.
    seen = []
    orig = FakeTrainer.train_step

    def patched(self, *a, **kw):
        out = orig(self, *a, **kw)
        seen.append(out[2])
        return out

    monkeypatch.setattr(FakeTrainer, "train_step", patched)
    trainer, cb = make([([2], True), ([3], False)])
    trainer.train()
    assert seen == [True, False]
    assert FakeSync.instances[0].calls == [(1, 4), (2, 0)]


def test_repo_root_on_sys_path():
    import os
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(sp.__file__)))
    assert os.path.dirname(root) in sys.path or any(
        os.path.abspath(p) == os.path.dirname(root) for p in sys.path)


def test_step_log_records_sync_time_and_counters(make, monkeypatch, tmp_path):
    import json

    path = tmp_path / "steps.jsonl"
    monkeypatch.setenv("YETO_DILOCO_STEP_LOG", str(path))
    trainer, cb = make([([3], True), ([2], True)])
    sync = FakeSync.instances[0]
    sync.rank = 0
    sync.stats = {"applied": 0, "pushed": 0, "apply_s": 0.0, "export_s": 0.0}
    original = sync.after_optimizer_step

    def step(iteration, target_tokens, *, update_successful=True):
        original(iteration, target_tokens, update_successful=update_successful)
        if iteration == 2:
            sync.stats["applied"] += 1
            sync.stats["apply_s"] += 1.5
            sync.stats["export_s"] += 0.25

    sync.after_optimizer_step = step
    trainer.train()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["iteration"] for r in rows] == [1, 2]
    assert rows[0]["applies"] == 0 and rows[0]["apply_s"] == 0
    assert rows[1]["applies"] == 1 and rows[1]["apply_s"] == 1.5 and rows[1]["export_s"] == 0.25
    assert rows[1]["target_tokens"] == 4 and rows[1]["sync_s"] >= 0


def test_step_log_only_on_sync_rank_zero(make, monkeypatch, tmp_path):
    path = tmp_path / "steps.jsonl"
    monkeypatch.setenv("YETO_DILOCO_STEP_LOG", str(path))
    trainer, cb = make([([3], True)])
    FakeSync.instances[0].rank = 5
    trainer.train()
    assert not path.exists()


class _Scheduler:
    """Megatron-Core ``OptimizerParamScheduler`` in miniature: sample-based
    ``num_steps``, linear warmup then cosine decay, ``step(increment)`` sets
    every parameter group's lr (the constructor steps by 0)."""

    def __init__(self, groups, *, max_lr=3e-4, min_lr=3e-5, warmup=5 * 8, decay=140 * 8):
        self.groups, self.max_lr, self.min_lr, self.warmup, self.decay = groups, max_lr, min_lr, warmup, decay
        self.num_steps = 0
        self.step(0)

    def get_lr(self):
        import math

        if self.warmup > 0 and self.num_steps <= self.warmup:
            return self.max_lr * self.num_steps / self.warmup
        ratio = min(1.0, (self.num_steps - self.warmup) / (self.decay - self.warmup))
        return self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (math.cos(math.pi * ratio) + 1.0)

    def step(self, increment):
        self.num_steps += increment
        for group in self.groups:
            group["lr"] = self.get_lr()


def _lr_trainer(iteration, gbs=8, scheduler=None):
    return types.SimpleNamespace(args=types.SimpleNamespace(global_batch_size=gbs),
                                 state=types.SimpleNamespace(iteration=iteration),
                                 opt_param_scheduler=scheduler or _Scheduler([{}]))


def test_a_resume_without_scheduler_state_continues_the_lr_schedule_in_place(monkeypatch):
    """A params-only checkpoint (--no_save_optim) restores the iteration but
    no scheduler: without the rebuild the resumed lr at iteration i is the
    uninterrupted one at i - 5 (warmup again)."""

    monkeypatch.delenv("YETO_DILOCO_LR_RESUME", raising=False)
    reference = _Scheduler([{}])
    want = {}
    for it in range(1, 141):
        reference.step(8)
        want[it] = reference.groups[0]["lr"]
    trainer = _lr_trainer(5)
    assert sp.restore_lr_schedule(trainer) == 40
    resumed = trainer.opt_param_scheduler
    assert resumed.groups[0]["lr"] == want[5]
    for it in range(6, 141):
        resumed.step(8)
        assert resumed.groups[0]["lr"] == want[it], it
    # Without it: shifted by the resumed iteration.
    shifted = _Scheduler([{}])
    for it in range(6, 141):
        shifted.step(8)
        assert shifted.groups[0]["lr"] == want[it - 5]


def test_lr_schedule_is_left_alone_when_loaded_cold_or_off(monkeypatch):
    loaded = _Scheduler([{}])
    loaded.step(5 * 8)  # restored from a checkpoint with optimizer state
    assert sp.restore_lr_schedule(_lr_trainer(5, scheduler=loaded)) is None and loaded.num_steps == 40
    cold = _lr_trainer(0)
    assert sp.restore_lr_schedule(cold) is None and cold.opt_param_scheduler.num_steps == 0
    assert sp.restore_lr_schedule(types.SimpleNamespace(state=types.SimpleNamespace(iteration=5))) is None
    monkeypatch.setenv("YETO_DILOCO_LR_RESUME", "off")
    off = _lr_trainer(5)
    assert sp.restore_lr_schedule(off) is None and off.opt_param_scheduler.num_steps == 0


def test_callback_rebuilds_the_lr_schedule_on_attach(make):
    trainer = FakeTrainer([([1], True)], start_iteration=15)
    trainer.args.global_batch_size = 384
    trainer.opt_param_scheduler = _Scheduler([{}], warmup=10 * 384, decay=1000 * 384)
    cb = sp.YetoDiLoCoCallback(trainer)
    assert trainer.opt_param_scheduler.num_steps == 15 * 384
    cb.close()


def test_initial_cut_wait_is_on_by_default(monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("YETO_DILOCO_INITIAL_CUT_TIMEOUT", raising=False)
    assert sp.DiLoCoHookConfig.from_env().initial_cut_timeout == 3600.0
    monkeypatch.setenv("YETO_DILOCO_INITIAL_CUT_TIMEOUT", "0")
    assert sp.DiLoCoHookConfig.from_env().initial_cut_timeout == 0.0


def _logger_under(root_handler_level):
    """A logger whose root mimics the Megatron SFT entry point's: one stdout
    handler at ``root_handler_level`` (ERROR in the real process)."""
    import io
    import logging

    root = logging.RootLogger(logging.INFO)
    sink = io.StringIO()
    handler = logging.StreamHandler(sink)
    handler.setLevel(root_handler_level)
    root.addHandler(handler)
    logger = logging.Logger("yeto-test-visible")
    logger.parent = root
    return logger, sink


def test_logs_are_made_visible_when_the_root_handler_drops_them(capsys):
    import logging

    from yeto.swift_megatron.sync import ensure_visible_logs

    logger, root_sink = _logger_under(logging.ERROR)
    ensure_visible_logs(logger)
    logger.info("info line")
    logger.warning("warning line")
    logger.error("error line")
    err = capsys.readouterr().err
    assert "info line" in err and "warning line" in err and err.count("error line") == 1
    assert root_sink.getvalue() == ""  # no longer propagated (no double ERROR)
    ensure_visible_logs(logger)  # idempotent
    assert len(logger.handlers) == 1


def test_logs_are_left_alone_when_the_process_prints_info(capsys):
    import logging

    from yeto.swift_megatron.sync import ensure_visible_logs

    logger, root_sink = _logger_under(logging.INFO)
    ensure_visible_logs(logger)
    assert logger.handlers == [] and logger.propagate
    logger.info("via root")
    assert "via root" in root_sink.getvalue() and capsys.readouterr().err == ""


def test_the_lr_rebuild_warning_survives_an_error_level_root(make, monkeypatch, capsys):
    import logging

    import yeto.swift_megatron.sync as sync_mod

    plugin_logger, _ = _logger_under(logging.ERROR)
    sync_logger, _ = _logger_under(logging.ERROR)
    monkeypatch.setattr(sp, "logger", plugin_logger)
    monkeypatch.setattr(sync_mod, "log", sync_logger)
    trainer = FakeTrainer([([1], True)], start_iteration=15)
    trainer.args.global_batch_size = 384
    trainer.opt_param_scheduler = _Scheduler([{}], warmup=10 * 384, decay=1000 * 384)
    cb = sp.YetoDiLoCoCallback(trainer)
    err = capsys.readouterr().err
    assert "restored no LR schedule; stepped it by 5760" in err
    assert sync_logger.handlers and not sync_logger.propagate
    cb.close()

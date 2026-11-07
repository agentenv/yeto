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

    def __init__(self, model_chunks, optimizer, cfg):
        self.model_chunks, self.optimizer, self.cfg = model_chunks, optimizer, cfg
        self.calls = []
        self.closed = 0
        FakeSync.instances.append(self)

    def after_optimizer_step(self, iteration, target_tokens, *, update_successful=True):
        self.calls.append((iteration, target_tokens))

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

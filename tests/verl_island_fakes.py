"""CPU fakes that drive the verl island code paths up to a chosen call.

* :func:`stub_verl_and_ray` imports the image-side verl modules (``trainer``,
  ``buffer_trainer_runner``) without verl or Ray: every ``verl.*`` / ``ray.*``
  module is a permissive stub. ``sys.modules`` is restored afterwards.
* :func:`drive_trainer` runs ``trainer.run_island`` or
  ``trainer.run_fully_async_island`` on a full-parameter plan until the
  ``IslandDriver`` is built and returns the driver's keyword arguments.
* :func:`run_island_entry` runs ``island_entry.main`` with the node work faked
  (runtime manifest, Ray, data) until a stop point and returns what happened.
"""
from __future__ import annotations

import contextlib
import importlib
import importlib.abc
import importlib.machinery
import sys
import types


def _deco(*args, **kwargs):
    return args[0] if len(args) == 1 and not kwargs and callable(args[0]) else _deco


class _StubMeta(type):
    def __getattr__(cls, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _deco


class _StubModule(types.ModuleType):
    def __call__(self, *args, **kwargs):
        return _deco(*args, **kwargs)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        if name[:1].islower():  # a submodule (or a function: the stub module is callable)
            return importlib.import_module(f"{self.__name__}.{name}")
        cls = _StubMeta(name, (), {"__init__": lambda self, *a, **k: None})
        setattr(self, name, cls)
        return cls


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    roots = ("verl", "ray")

    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in self.roots:
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None

    def create_module(self, spec):
        return _StubModule(spec.name)

    def exec_module(self, module):
        module.__path__ = []
        if module.__name__ == "ray":
            module.remote = lambda *a, **k: (lambda cls: cls)


@contextlib.contextmanager
def stub_verl_and_ray():
    saved = dict(sys.modules)
    for name in list(sys.modules):
        if name.split(".")[0] in _StubFinder.roots:
            del sys.modules[name]
    finder = _StubFinder()
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)
        sys.modules.clear()
        sys.modules.update(saved)


def modules(*names: str):
    """The modules as ``sys.modules`` holds them (what a relative import in the code
    under test resolves to; a package attribute can be an older module object)."""
    return tuple(importlib.import_module(name) for name in names)


class Reached(Exception):
    """Raised by a fake at the stop point; ``kwargs`` are what it was called with."""

    def __init__(self, name, *args, **kwargs):
        super().__init__(name)
        self.name, self.args_, self.kwargs = name, args, kwargs


def _identity():
    return types.SimpleNamespace(compat_group="nvidia-h100", sha256=lambda: "c" * 64, to_dict=lambda: {})


def full_plan(tmp_path, **extra) -> dict:
    return {"learner_id": 0, "max_policy_age": 1, "groups_per_round": 1, "samples_per_group": 2,
            "sync": "none", "parameter_mode": "full", "model_config_sha256": "b" * 64,
            "model_path": str(tmp_path), "model_revision": "a" * 40, "out_dir": str(tmp_path),
            "event_tape": str(tmp_path / "tape.jsonl"), "thresholds_key": [], **extra}


def drive_trainer(monkeypatch, tmp_path, path: str, *, plan: dict | None = None, buffer_ports=None):
    """Run ``trainer.run_island`` (``path="sync"``) or ``trainer.run_fully_async_island``
    (``"fully_async"``) on a full-parameter plan up to ``IslandDriver(...)``; the
    island-side objects are fakes. Returns the driver's keyword arguments."""
    (fully_async_full, fully_async_ports, fully_async_round, identity, ports_impl, round_cuts, separate,
     trainer, driver_module, catalog) = modules(*(f"yeto.rl.adapters.verl.{name}" for name in (
         "fully_async_full", "fully_async_ports", "fully_async_round", "identity", "ports_impl",
         "round_cuts", "separate", "trainer")), "yeto.rl.engine.driver", "yeto.hw.catalog")

    def driver(**kwargs):
        raise Reached("IslandDriver", **kwargs)

    island = types.SimpleNamespace(layout_hash="d" * 64)
    monkeypatch.setattr(driver_module, "IslandDriver", driver)
    monkeypatch.setattr(identity, "backend_identity", lambda **k: _identity())
    monkeypatch.setattr(catalog, "runtime_versions", lambda: {})
    monkeypatch.setattr(round_cuts, "resume_wiring_for", lambda *a, **k: None)
    plan = plan or full_plan(tmp_path)
    if path == "sync":
        for name in ("VerlIsland", "VerlRolloutPool", "VerlTrainerGroup"):
            monkeypatch.setattr(ports_impl, name, lambda *a, **k: island)
        monkeypatch.setattr(separate, "full_parameter_publisher", lambda *a, **k: "publisher")
        monkeypatch.setattr(separate, "driver_options", lambda *a, **k: {})
        monkeypatch.setattr(trainer, "require_sync_trainer", lambda: None)
        config = types.SimpleNamespace(trainer=types.SimpleNamespace(project_name="p", experiment_name="e",
                                                                     logger=[]))
        fake = types.SimpleNamespace(config=config, _reissue_inflight_prompts=lambda: None,
                                     on_train_begin=lambda: None)
        try:
            trainer.run_island(fake, None, plan, {})
        except Reached as stop:
            return stop.kwargs
        raise AssertionError("run_island did not build its driver")
    for name in ("FullyAsyncIsland", "FullyAsyncRolloutPool", "FullyAsyncTrainerGroup",
                 "FullyAsyncFullParameterPublisher", "FullyAsyncPlacement"):
        monkeypatch.setattr(fully_async_ports, name, lambda *a, **k: island)
    monkeypatch.setattr(fully_async_round, "execution_profile_for", lambda *a, **k: None)
    monkeypatch.setattr(fully_async_full, "full_parameter_hashes", lambda plan: ("d" * 64, "e" * 64))
    try:
        trainer.run_fully_async_island(lambda *a, **k: None, None, plan, {}, buffer_ports=buffer_ports)
    except Reached as stop:
        return stop.kwargs
    raise AssertionError("run_fully_async_island did not build its driver")


ISLAND_ARGV = ("--model", "m", "--model-revision", "a" * 40, "--data", "d", "--learner-id", "0",
               "--reward-function", "yeto.rl.gsm8k_reward:score", "--global-rounds", "1", "--groups-per-round", "1",
               "--samples-per-group", "2", "--rollout-max-response-len", "8", "--event-tape", "t",
               "--inner-lr", "1", "--seq-len", "64", "--seed", "1", "--rl-single-island-no-sync")


def run_island_entry(monkeypatch, tmp_path, *argv: str, stop: str = "fsdp_island_run"):
    """Run ``verl.island_entry.main`` on ``ISLAND_ARGV + argv`` with the node work
    faked. Returns ``("refused", message)`` for a SystemExit, or ``(name, kwargs)``
    for the first call of ``stop``: ``fsdp_island_run`` (the island accepted the
    configuration and builds its FSDP2 run), ``refuse_unwired`` or ``eval_section``
    (the heldout eval decision). The calls before the stop return fakes."""
    eval_port, full_parameter_stream, island_entry, round_cuts, task_dispense, train_options = modules(
        *(f"yeto.rl.adapters.verl.{name}" for name in ("eval_port", "full_parameter_stream", "island_entry",
                                                       "round_cuts", "task_dispense", "train_options")))

    def reached(name, result=None):
        def fake(*args, **kwargs):
            if name == stop:
                raise Reached(name, *args, **kwargs)
            return result
        return fake

    run = types.SimpleNamespace(temperature=1.0, top_p=1.0, n_gpus=1, to_dict=lambda: {})
    monkeypatch.setattr(island_entry, "OUTPUT", tmp_path / "out")
    monkeypatch.setattr(island_entry, "runtime_manifest", lambda lid, engine=None: {"problems": []})
    monkeypatch.setattr(island_entry.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(island_entry, "ensure_ray_cpus", lambda: {})
    monkeypatch.setattr(island_entry, "island_model_path", lambda *a: str(tmp_path))
    monkeypatch.setattr(island_entry, "prepare_data", lambda *a, **k: ("train", "val"))
    monkeypatch.setattr(island_entry, "fsdp_separate_pools",
                        lambda args, info: "pools" if args.rl_placement == "fixed-partition" else None)
    monkeypatch.setattr(island_entry, "fsdp_island_run", reached("fsdp_island_run", (run, [])))
    monkeypatch.setattr(island_entry, "megatron_island_plan",
                        reached("megatron_island_plan", {"run": run, "overrides": [], "asserted_keys": ()}))
    monkeypatch.setattr(island_entry, "megatron_resume", lambda args, resume: (resume, None))
    monkeypatch.setattr(task_dispense, "island_dispense", lambda *a, **k: None)
    monkeypatch.setattr(train_options, "refuse_unmapped_argv", lambda *a, **k: None)
    monkeypatch.setattr(train_options, "verify_plan", lambda *a, **k: {})
    monkeypatch.setattr(train_options, "debug_plan", lambda *a, **k: {})
    monkeypatch.setattr(train_options, "apply_training_options", lambda a, o, k, **kw: (o, k))
    monkeypatch.setattr(round_cuts, "resume_config", lambda args: None)
    monkeypatch.setattr(full_parameter_stream, "plan_section", lambda *a, **k: {})
    monkeypatch.setattr(eval_port, "refuse_unwired", reached("refuse_unwired"))
    monkeypatch.setattr(eval_port, "eval_section", reached("eval_section"))
    try:
        island_entry.main([*ISLAND_ARGV, *argv])
    except Reached as hit:
        return hit.name, (hit.args_, hit.kwargs)
    except SystemExit as error:
        return "refused", str(error)
    raise AssertionError(f"island_entry.main passed {stop} without calling it")

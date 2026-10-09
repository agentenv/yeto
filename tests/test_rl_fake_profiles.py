"""decoupling 2.8 (and the fake half of 2.2): capability-parameterized fake engine.

The core runs against the fake's two capability profiles; the core imports and
drives a fake island in a process where Miles, Megatron, SGLang and the Miles
adapter cannot be imported at all.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from yeto.rl.elastic_benchmark.capabilities import ResourceConfig, attestation_from_dict
from yeto.rl.engine.fake import FAKE_PROFILES, FakeEngine, fake_capabilities
from yeto.rl.engine.ports import CuttableTrainer, TrainerGroup
from yeto.rl.engine.trainer_transition import TrainerEdgeRejected, plan_trainer_edge

REPO = Path(__file__).resolve().parents[1]
SHA = "a" * 64
CONFIGS = {
    "T2R2": ResourceConfig("T2R2", 2, 2, placement={"trainer": ["g0", "g1"], "rollout": ["g2", "g3"]}),
    "T1R3": ResourceConfig("T1R3", 1, 3, placement={"trainer": ["g0"], "rollout": ["g1", "g2", "g3"]}),
}
ATTESTATION = attestation_from_dict({
    "runtime_fingerprint": "fp", "execution_modes": [],
    "certified_edges": [{"source": "T2R2", "target": "T1R3", "kind": "role-transfer",
                         "algorithm_spec_sha256": [SHA]}],
})


def _engine(profile):
    return FakeEngine(tensors={"base_model.model.layer.lora_A.weight": torch.zeros(1, 2)}, capability_profile=profile)


def _plan(trainer, gbs=4):
    return plan_trainer_edge(configs=CONFIGS, attestation=ATTESTATION, source="T2R2", target="T1R3",
                             expected_config_epoch=1, spec=SimpleNamespace(sha256=lambda: SHA), args=None,
                             global_batch_size=gbs, micro_batch_size=1, trainer=trainer)


def test_profiles_declare_different_capabilities():
    miles_like, ports_only = (fake_capabilities(p) for p in FAKE_PROFILES)
    assert miles_like.traits.publish_while_offloaded and not ports_only.traits.publish_while_offloaded
    assert ports_only.corrections == {"none"} and "tis" in miles_like.corrections
    assert ports_only.traits.transport_label("collective-broadcast") == "collective-broadcast"
    with pytest.raises(ValueError, match="unknown fake capability profile"):
        fake_capabilities("verl")


def test_fake_trainer_reshard_through_the_core_plan():
    trainer = _engine("miles-like").trainer
    assert isinstance(trainer, TrainerGroup)
    plan = _plan(trainer)
    assert plan.reshard.source["dp"] == 2 and plan.reshard.target["dp"] == 1
    with pytest.raises(TrainerEdgeRejected, match="not divisible"):
        _plan(trainer, gbs=3)
    ports_only = _engine("ports-only").trainer
    assert isinstance(ports_only, TrainerGroup) and not isinstance(ports_only, CuttableTrainer)
    with pytest.raises(TrainerEdgeRejected, match="does not advertise reshard_problems"):
        _plan(ports_only)


_BLOCKER = textwrap.dedent('''
    import importlib, importlib.abc, pkgutil, sys
    BLOCK = tuple(sys.argv[2].split(","))
    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if any(name == b or name.startswith(b + ".") for b in BLOCK):
                raise ImportError("not installed: " + name)
    sys.meta_path.insert(0, Block())
''')

# Every core module imports even when the Miles adapter and the legacy engine are absent.
IMPORT_CORE = _BLOCKER + textwrap.dedent('''
    import yeto.rl.engine as core
    for m in pkgutil.iter_modules(core.__path__):
        if m.name != "miles_adapter":
            importlib.import_module("yeto.rl.engine." + m.name)
    print("OK")
''')

# Miles (and Megatron/SGLang) not installed: the fake island runs under both profiles.
# (Algorithm extensions still register flags through the yeto-side adapter module
# until task 4.x moves them, so only the frameworks are blocked here.)
DRIVE_FAKE = _BLOCKER + textwrap.dedent('''
    from pathlib import Path
    import torch
    from yeto.rl.engine.algorithm import AlgorithmSpec
    from yeto.rl.engine.bridges import LocalOnlySync
    from yeto.rl.engine.driver import EventTape, IslandDriver
    from yeto.rl.engine.fake import FakeEngine, fake_capabilities
    out = Path(sys.argv[1])
    for profile in ("miles-like", "ports-only"):
        e = FakeEngine(tensors={"base_model.model.layer.lora_A.weight": torch.zeros(1, 2)}, capability_profile=profile)
        d = IslandDriver(learner_id=0, rollout=e.rollout, trainer=e.trainer, policy_state=e.policy_state,
                         publisher=e.publisher, placement=e.placement, algorithm=AlgorithmSpec(),
                         sync=LocalOnlySync(2), events=EventTape(out / (profile + ".jsonl"), 0),
                         capabilities=fake_capabilities(profile))
        assert d.run().policy_version == 2, profile
    assert not [m for m in sys.modules if m.split(".")[0] in ("miles", "megatron", "sglang")]
    print("OK")
''')


def _run(script, tmp_path, blocked):
    out = subprocess.run([sys.executable, "-c", script, str(tmp_path), ",".join(blocked)], cwd=REPO,
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0 and out.stdout.strip().endswith("OK"), out.stderr[-3000:]


def test_core_imports_without_miles_or_its_adapter(tmp_path):
    _run(IMPORT_CORE, tmp_path, ("miles", "megatron", "sglang", "yeto.rl.adapters.miles.legacy.engine", "yeto.rl.adapters.miles.overlay",
                                 "yeto.rl.adapters.miles"))


def test_fake_driver_runs_without_miles_installed(tmp_path):
    _run(DRIVE_FAKE, tmp_path, ("miles", "megatron", "sglang"))

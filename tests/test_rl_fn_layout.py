"""S11 try24: Flash-Next fnA died in ``_run_ports`` on ``canonical_layout_hash(())``.

The Flash-Next LoRA layout is not predictable CPU-side (Miles' native
qwen3_8_next exporter: tied q/k/v A, rank-padded experts), so the ports island
learns it from the first trainable-state export, before any receipt, and pins it.
``_run_ports`` is NOT replaced here; only ``run_ports_island`` is a recording stub.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from test_rl_fn_provider_view import REPO, _probe, snapshots  # noqa: F401,E402


def test_run_miles_flash_next_reaches_run_ports_island_with_learned_layout(snapshots):  # noqa: F811
    body = f"""
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        sys.path.insert(0, {str(REPO / 'tests')!r})
        sys.path.insert(0, {str(REPO / 'tests' / 'multinode_gpu')!r})
        import _pytest.monkeypatch as _m
        from rl_e2e_launch import island_run, learner_from_run
        from fp_fn import fnrun_cli
        from yeto.rl import learner
        from yeto.rl.engine import run_config
        from yeto.rl.engine.miles_adapter import config as mac
        from yeto.rl.engine.miles_adapter import entry
        from yeto.rl.engine.miles_adapter import state as mas

        mp = _m.MonkeyPatch()
        run = island_run(tuple(fnrun_cli("fn8s")), mp)
        args, _env = learner_from_run(run, Path(tempfile.mkdtemp()) / "home")
        args.event_tape = str(Path(tempfile.mkdtemp()) / "tape.jsonl")
        mp.setattr(run_config, "_resolve_ref_load",
                   lambda a, model_path: a.megatron_ref_load or str(model_path))
        mp.setattr(mas, "require_run_plugin", lambda: None)
        captured = {{}}

        def parse(launch):  # the real parser needs Miles; keep the launch's argv
            return SimpleNamespace(argv=list(launch.argv))

        mp.setattr(mac, "parse_miles_args", parse)
        # needs a parsed Miles namespace; covered by its own tests
        mp.setattr(learner, "verify_ports_algorithm", lambda *a: None)
        mp.setattr(learner, "require_ports_router_mode", lambda *a: None)

        def island(miles_args, launch, algorithm, **kw):
            captured.update({{k: v for k, v in kw.items()}})
            captured["native"] = getattr(miles_args, "yeto_rl_native_lora_export", None)

        mp.setattr(entry, "run_ports_island", island)
        learner.run_miles(args, model_path={str(snapshots['4layer'])!r},
                          prompt_path="/root/yeto-rl/prompts.jsonl",
                          yeto_policy_sync=False)
        print(json.dumps(captured))
    """
    out = _probe(body)
    assert out["layout_hash"] is None
    assert out["native"] == "qwen3_8_next"
    assert out["yeto_policy_sync"] is False
    assert len(out["lora_config_hash"]) == 64


def test_run_ports_predicted_layout_unchanged(monkeypatch):
    from yeto.rl import learner
    from yeto.rl.core import CanonicalTensorSpec, canonical_layout_hash
    from yeto.rl.engine.miles_adapter import entry

    seen = {}
    monkeypatch.setattr(entry, "run_ports_island", lambda m, l, a, **kw: seen.update(kw))
    monkeypatch.setattr(learner, "apply_ports_infra_switches", lambda *a: None)
    specs = (CanonicalTensorSpec("base_model.model.x.lora_A.weight", (2, 4), "float32", 8),)
    args = SimpleNamespace(lora_r=2, event_tape=None, learner_id=0, model_revision="r")
    miles_args = SimpleNamespace()
    learner._run_ports(args, miles_args, None, None, specs=specs,
                       canonical_targets=["q_proj"], yeto_policy_sync=False)
    assert seen["layout_hash"] == canonical_layout_hash(specs)
    assert not hasattr(miles_args, "yeto_rl_native_lora_export")
    with pytest.raises(ValueError, match="canonical LoRA layout is empty"):
        learner._run_ports(args, SimpleNamespace(), None, None, specs=(),
                           canonical_targets=["q_proj"], yeto_policy_sync=False)
    with pytest.raises(ValueError, match="without Yeto policy sync"):
        learner._run_ports(args, SimpleNamespace(), None, None, specs=(),
                           canonical_targets=["q_proj"], yeto_policy_sync=True,
                           native_lora_export="qwen3_8_next")


# --- Miles-native export naming (c35702e lora.py _export_gdn/_export_qsa/...) ---
P = "model.language_model.layers."


def _fake_chunks(rank=4, hidden=8, bump=0.0):
    t = lambda *s: torch.full(s, 0.5 + bump, dtype=torch.bfloat16)  # noqa: E731
    a = t(rank, hidden)
    gdn = [(f"{P}0.linear_attn.{n}.lora_{s}.weight", t(rank, hidden) if s == "A" else t(6, rank))
           for n in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj") for s in "AB"]
    qsa = [(f"{P}3.self_attn.{n}_proj.lora_A.weight", a) for n in "qkv"]
    qsa += [(f"{P}3.self_attn.{n}_proj.lora_B.weight", t(6, rank)) for n in "qkv"]
    qsa += [(f"{P}3.self_attn.o_proj.lora_A.weight", lambda: t(rank, hidden)),
            (f"{P}3.self_attn.o_proj.lora_B.weight", t(hidden, rank))]
    experts = [(f"{P}0.mlp.experts.gate_up_proj.lora_A.weight", t(3, rank, hidden)),
               (f"{P}0.mlp.experts.down_proj.lora_B.weight", t(3, hidden, rank))]
    return [gdn, [(n, v() if callable(v) else v) for n, v in qsa], experts]


def _actor(main=True):
    return SimpleNamespace(args=SimpleNamespace(yeto_rl_native_lora_export="qwen3_8_next"),
                           model=[object()], _is_first_replica_megatron_main_rank=main)


class _FakeActorModel:
    def __init__(self):
        self.bump = 0.0
        self.calls = 0

    def run_plugin(self, path, kwargs):
        from yeto.rl.engine.miles_adapter import state_plugin as sp

        self.calls += 1
        assert path == sp.EXPORT_STATE
        chunks = _fake_chunks(bump=self.bump)
        return [sp._export_flash_next(_actor(), exporter=lambda m: iter(chunks), **kwargs),
                sp._export_flash_next(_actor(False), exporter=lambda m: iter(chunks), **kwargs)]


def test_native_export_validates_and_policy_state_learns_layout(capsys):
    from yeto.rl.core import canonical_layout_hash, canonical_state_from_owned_tensors
    from yeto.rl.engine.miles_adapter import state_plugin as sp
    from yeto.rl.engine.miles_adapter.state import MilesPolicyState, PolicyStateError

    out = sp._export_flash_next(_actor(), policy_version=0, exporter=lambda m: iter(_fake_chunks()))
    names = set(out["tensors"])
    assert len(names) == 10 + 8 + 2 and all(n.startswith("base_model.model.model.language_model.")
                                            for n in names)
    lora = canonical_state_from_owned_tensors(0, out["tensors"], base_model_revision="a" * 40,
                                              lora_config_hash="c" * 64)
    assert lora.layout_hash == canonical_layout_hash(lora.specs)

    model = _FakeActorModel()
    ps = MilesPolicyState(actor_model=model, base_model_revision="a" * 40, config_hash="c" * 64,
                          expected_layout_hash=None)
    with pytest.raises(PolicyStateError, match="before the first"):
        ps.layout_hash
    first = ps.export()
    assert ps.layout_hash == lora.layout_hash
    assert "learned LoRA layout from first export: 20 tensors" in capsys.readouterr().out
    model.bump = 1.0  # trained values change, layout pinned
    assert ps.export().policy_tensor_hash() != first.policy_tensor_hash()
    model.run_plugin = lambda path, kw: [{"policy_version": 0, "tensors": {
        k: v for k, v in first.to_lora().tensors.items() if "o_proj" not in k}}]
    with pytest.raises(ValueError, match="layout hash changed|names, shapes"):
        ps.export()


def test_native_export_refuses_duplicates_and_apply():
    from yeto.rl.engine.miles_adapter import state_plugin as sp

    dup = [[(f"{P}0.x.lora_A.weight", torch.zeros(2, 2))], [(f"{P}0.x.lora_A.weight", torch.zeros(2, 2))]]
    with pytest.raises(sp.StatePluginError, match="duplicate"):
        sp._export_flash_next(_actor(), policy_version=0, exporter=lambda m: iter(dup))
    with pytest.raises(sp.StatePluginError, match="no apply_state contract"):
        sp._apply_state(_actor(), tensors={}, policy_version=0, local_step=0, optimizer=sp.OPTIMIZER_MODES[0])


def test_receipt_layout_reads_learned_hash_after_first_sync_start_export():
    """Driver order: LocalOnlySync.start exports before any train/receipt."""
    import inspect

    from yeto.rl.engine import bridges
    from yeto.rl.engine.miles_adapter import entry

    assert "driver.export_local()" in inspect.getsource(bridges.LocalOnlySync.start)
    src = inspect.getsource(entry)
    assert "parameter_layout_hash=lambda: policy_state.layout_hash" in src
    assert "parameter_layout_hash=lambda: layout_hash" not in src

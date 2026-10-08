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


def _layer_chunks(layer, kind, rank=4, hidden=8, bump=0.0):
    """One layer as Miles exports it: attention (GDN 10 / QSA 8) + shared 6 + experts 4."""
    t = lambda *s: torch.full(s, 0.5 + bump, dtype=torch.bfloat16)  # noqa: E731
    a = t(rank, hidden)
    if kind == "linear_attention":
        attn = [(f"{P}{layer}.linear_attn.{n}.lora_{s}.weight", t(rank, hidden) if s == "A" else t(6, rank))
                for n in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj") for s in "AB"]
    else:
        attn = [(f"{P}{layer}.self_attn.{n}_proj.lora_A.weight", a) for n in "qkv"]
        attn += [(f"{P}{layer}.self_attn.{n}_proj.lora_B.weight", t(6, rank)) for n in "qkv"]
        attn += [(f"{P}{layer}.self_attn.o_proj.lora_A.weight", t(rank, hidden)),
                 (f"{P}{layer}.self_attn.o_proj.lora_B.weight", t(hidden, rank))]
    shared = [(f"{P}{layer}.mlp.shared_expert.{n}_proj.lora_{s}.weight", t(rank, hidden) if s == "A" else t(hidden, rank))
              for n in ("gate", "up", "down") for s in "AB"]
    experts = [(f"{P}{layer}.mlp.experts.{n}.lora_{s}.weight", t(3, rank, hidden) if s == "A" else t(3, hidden, rank))
               for n in ("gate_up_proj", "down_proj") for s in "AB"]
    return [attn, shared, experts]


LAYER_TYPES_4 = ("linear_attention",) * 3 + ("full_attention",)


def _fake_chunks(rank=4, hidden=8, bump=0.0, layers=range(4)):
    return [c for i in layers for c in _layer_chunks(i, LAYER_TYPES_4[i], rank, hidden, bump)]


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
    assert len(names) == 78 and all(n.startswith("base_model.model.model.language_model.")
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
    assert "learned LoRA layout from first export: 78 tensors" in capsys.readouterr().out
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

    # rl-publish-fastpath: start goes through _local_state, which exports (digest
    # export or full export); both learn the layout hash on the first export.
    assert "_local_state(driver" in inspect.getsource(bridges.LocalOnlySync.start)
    local_state = inspect.getsource(bridges._local_state)
    assert "export_local_resident" in local_state and "driver.export_local()" in local_state
    from yeto.rl.engine.miles_adapter import state as miles_state

    assert "self._expected_layout_hash = layout_hash" in inspect.getsource(
        miles_state.MilesPolicyState._digest_result)
    src = inspect.getsource(entry)
    assert "parameter_layout_hash=lambda: policy_state.layout_hash" in src
    assert "parameter_layout_hash=lambda: layout_hash" not in src


# --- PP gather of the native export (FN-A-RESULT 38 vs 78) -------------------


def _stage_exporter(stage_layers):
    return lambda m: iter(_fake_chunks(layers=stage_layers))


def test_profile_expected_native_export_tensors():
    from yeto.rl.profiles import qwen3_8_next as prof

    assert prof.expected_native_export_tensors("4layer") == 78
    # FN-A-RESULT: stage 1 (layers 2-3) 38, stage 0 (layers 0-1) 40.
    assert prof.expected_native_export_tensors("4layer", LAYER_TYPES_4) == 40 + 38
    assert prof.LAYER_TYPES["4layer"] == LAYER_TYPES_4
    with pytest.raises(ValueError, match="not pinned"):
        prof.expected_native_export_tensors("full")
    with pytest.raises(ValueError, match="layer_types"):
        prof.expected_native_export_tensors("4layer", LAYER_TYPES_4[:3])


def test_native_export_gathers_all_pp_stages(capsys, monkeypatch):
    """TP2 PP2 EP4: main rank (last stage) merges stage 0's 40 with its own 38."""
    from yeto.rl.engine.miles_adapter import state_plugin as sp
    from yeto.rl.profiles import qwen3_8_next as prof

    sent = {}

    def gather_stage0(local, is_main):  # stage 0 peer: sends, gets nothing back
        assert not is_main and local is not None
        sent["stage0"] = local
        return None

    actor0 = _actor(False)
    actor0.args.num_layers = 4
    monkeypatch.setattr(sp, "_tp_ep_leader", lambda actor: True)  # stage-0 peer of the main rank
    assert sp._export_flash_next(actor0, policy_version=0, exporter=_stage_exporter([0, 1]),
                                 pp_gather=gather_stage0) is None
    monkeypatch.undo()
    assert len(sent["stage0"]) == 40

    def gather_main(local, is_main):
        assert is_main and len(local) == 38
        return [sent["stage0"], local]

    actor1 = _actor(True)
    actor1.args.num_layers = 4
    out = sp._export_flash_next(actor1, policy_version=3, exporter=_stage_exporter([2, 3]), pp_gather=gather_main)
    assert len(out["tensors"]) == prof.expected_native_export_tensors("4layer") == 78
    assert {sp._fn_layer(n) for n in out["tensors"]} == {0, 1, 2, 3}
    assert "per-PP-stage tensors [40, 38], merged 78" in capsys.readouterr().out


def _canon_stage(layers):
    from yeto.rl.engine.miles_adapter import state_plugin as sp

    return {sp._canonical(n): v for c in _fake_chunks(layers=layers) for n, v in c}


def test_merge_pp_stage_exports_validates():
    from yeto.rl.engine.miles_adapter import state_plugin as sp

    s0, s1 = _canon_stage([0, 1]), _canon_stage([2, 3])
    assert len(sp.merge_pp_stage_exports([s0, s1], num_layers=4)) == 78
    # stage-local numbering (both stages say layers 0-1) is refused, not silently merged
    with pytest.raises(sp.StatePluginError, match="overlap|global"):
        sp.merge_pp_stage_exports([s0, _canon_stage([0, 1])], num_layers=4)
    with pytest.raises(sp.StatePluginError, match="overlap|global"):
        sp.merge_pp_stage_exports([s1, s0], num_layers=4)
    # missing stage / layer (the pre-fix 38-tensor export)
    with pytest.raises(sp.StatePluginError, match=r"missing layers \[0, 1\]"):
        sp.merge_pp_stage_exports([s1], num_layers=4)
    with pytest.raises(sp.StatePluginError, match="no LoRA tensors"):
        sp.merge_pp_stage_exports([s0, {}], num_layers=4)
    # duplicate across stages
    dup = dict(s1)
    dup[next(iter(s0))] = torch.zeros(1)
    with pytest.raises(sp.StatePluginError, match="duplicate|overlap"):
        sp.merge_pp_stage_exports([s0, dup], num_layers=4)
    # incomplete layer
    short = {k: v for k, v in s1.items() if "o_proj.lora_B" not in k}
    with pytest.raises(sp.StatePluginError, match="layer 3 exported 17 tensors, expected 18"):
        sp.merge_pp_stage_exports([s0, short], num_layers=4)
    with pytest.raises(sp.StatePluginError, match="missing a stage"):
        sp._export_flash_next(_actor(), policy_version=0, exporter=_stage_exporter([2, 3]),
                              pp_gather=lambda local, is_main: [None, local])


def test_pp_gather_unwraps_miles_reloadable_process_group():
    """s13-h100-20261007a: gather_object(dst=...) rejected Miles' ReloadableProcessGroup
    wrapper ("is not registered"); the export must hand torch the inner group."""
    from yeto.rl.engine.miles_adapter.state_plugin import _torch_group

    class Reloadable:  # mirrors miles.utils.reloadable_process_group.ReloadableProcessGroup
        def __init__(self, group):
            self.group = group

        def __getattr__(self, name):
            return getattr(self.group, name)

    inner = object()
    assert _torch_group(Reloadable(inner)) is inner
    plain = object()
    assert _torch_group(plain) is plain

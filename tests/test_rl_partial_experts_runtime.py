"""Generic partial experts on Miles: trainer-side freezing, the layout it yields, the real syncer.

Two small randomly initialised MoE structures stand in for the trainer's model
chunks on each EP rank:

* ``qwen3_moe``-like: every layer MoE, TE grouped-GEMM experts named per
  expert (``mlp.experts.linear_fc1.weight<local>``), EP 2 folded onto TP 2;
* ``deepseek``-like: layer 0 dense, ``SequentialMLP`` experts
  (``mlp.experts.local_experts.<local>.linear_fc1.weight``), EP 4.

:func:`in_place.freeze_chunk` freezes each chunk; the fake engine then lists
the ``requires_grad`` parameters under Megatron global names, as Miles'
``full_parameter_state`` does, so the syncer layout holds exactly the
selected experts. Two islands push deltas of those experts through the real
Rust syncer; the final cut must equal ``init + mean(deltas)``.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from free_ports import free_port  # noqa: E402
from process_groups import group_popen, kill_group  # noqa: E402
from yeto.protocol import DTYPE_F32, SyncerClient  # noqa: E402
from yeto.rl.adapters.miles.models.partial_experts import in_place  # noqa: E402
from yeto.rl.adapters.miles.models.partial_experts import selection as sel  # noqa: E402
from yeto.rl.local_learner import ComponentIdentity, ParameterLayout, ParameterSpec  # noqa: E402
from yeto.rl.moe_geometry import from_hf_config  # noqa: E402
from yeto.rl.moe_layout import check_expert_partition, sequential_expert_global_name  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CARGO = shutil.which("cargo") or str(Path.home() / ".cargo/bin/cargo")
PREFIX = "module.module."
HIDDEN, FFN = 8, 4

STRUCTURES = {
    "qwen3_moe": dict(
        config={"model_type": "qwen3_moe", "num_hidden_layers": 2, "num_experts": 8, "num_experts_per_tok": 2,
                "decoder_sparse_step": 1, "mlp_only_layers": []},
        ep=2, grouped=True, num_trainable=4),
    "deepseek_v3": dict(
        config={"model_type": "deepseek_v3", "num_hidden_layers": 3, "n_routed_experts": 8,
                "num_experts_per_tok": 2, "first_k_dense_replace": 1},
        ep=4, grouped=False, num_trainable=4),
}


def _param(*shape):
    return torch.nn.Parameter(torch.randn(*shape))


def _chunk(structure: dict, ep_rank: int) -> torch.nn.Module:
    """One EP rank's model chunk (random weights, Megatron module names)."""

    geometry = from_hf_config(structure["config"])
    n_local = geometry.num_experts // structure["ep"]
    root = torch.nn.Module()
    root.embedding = torch.nn.Module()
    root.embedding.word_embeddings = torch.nn.Module()
    root.embedding.word_embeddings.weight = _param(32, HIDDEN)
    root.decoder = torch.nn.Module()
    root.decoder.layers = torch.nn.ModuleList()
    for layer in range(geometry.num_layers):
        block = torch.nn.Module()
        block.self_attention = torch.nn.Module()
        block.self_attention.linear_qkv = torch.nn.Linear(HIDDEN, 3 * HIDDEN, bias=False)
        mlp = torch.nn.Module()
        if layer not in geometry.moe_layers:
            mlp.linear_fc1 = torch.nn.Linear(HIDDEN, 2 * FFN, bias=False)
            mlp.linear_fc2 = torch.nn.Linear(FFN, HIDDEN, bias=False)
        else:
            mlp.router = torch.nn.Linear(HIDDEN, geometry.num_experts, bias=False)
            mlp.experts = torch.nn.Module()
            if structure["grouped"]:
                mlp.experts.linear_fc1 = torch.nn.Module()
                mlp.experts.linear_fc2 = torch.nn.Module()
                for local in range(n_local):
                    mlp.experts.linear_fc1.register_parameter(f"weight{local}", _param(2 * FFN, HIDDEN))
                    mlp.experts.linear_fc2.register_parameter(f"weight{local}", _param(HIDDEN, FFN))
            else:
                mlp.experts.local_experts = torch.nn.ModuleList()
                for _ in range(n_local):
                    expert = torch.nn.Module()
                    expert.linear_fc1 = torch.nn.Linear(HIDDEN, 2 * FFN, bias=False)
                    expert.linear_fc2 = torch.nn.Linear(FFN, HIDDEN, bias=False)
                    mlp.experts.local_experts.append(expert)
        block.mlp = mlp
        root.decoder.layers.append(block)
    return root


_GROUPED_LOCAL = re.compile(r"^(decoder\.layers\.\d+\.mlp\.experts\.linear_fc[12]\.weight)(\d+)$")


def _global_name(name: str, *, ep_rank: int, n_local: int) -> str:
    """Miles ``named_params_and_buffers(convert_to_global_name=True)`` (grouped GEMM) and yeto's
    ``sequential_expert_global_name`` (SequentialMLP)."""

    match = _GROUPED_LOCAL.match(name)
    if match:
        return f"{match.group(1)}{ep_rank * n_local + int(match.group(2))}"
    return sequential_expert_global_name(name, ep_rank=ep_rank, num_local_experts=n_local)


def _selection(structure: dict, mode: str = "random") -> sel.ExpertSelection:
    geometry = from_hf_config(structure["config"])
    return sel.ExpertSelection.in_place(mode, model=geometry.model_type, num_experts=geometry.num_experts,
                                        num_trainable=structure["num_trainable"], ep_size=structure["ep"],
                                        layers=geometry.moe_layers, seed=11)


def _island_specs(structure: dict, selection: sel.ExpertSelection) -> list[ParameterSpec]:
    """Every EP rank freezes its chunk; the trainable masters under global names (the fake engine)."""

    ep = structure["ep"]
    n_local = selection.num_experts // ep
    specs = []
    for rank in range(ep):
        chunk = _chunk(structure, rank)
        coords = in_place.RankCoordinates(layer_offset=0, ep_rank=rank, ep_size=ep)
        trainable, _ = in_place.freeze_chunk(chunk, selection=selection, coords=coords)
        owner = f"tp{rank}-of-{ep}.pp0-of-1.ep{rank}-of-{ep}.cp0-of-1.dp0-of-1"
        names = [name for name, parameter in chunk.named_parameters() if parameter.requires_grad]
        assert len(names) == trainable
        for name, parameter in chunk.named_parameters():
            if parameter.requires_grad:
                shape = tuple(parameter.shape)
                specs.append(ParameterSpec("actor", PREFIX + _global_name(name, ep_rank=rank, n_local=n_local),
                                           shape, "float32", int(np.prod(shape)), owner))
    return specs


@pytest.mark.parametrize("name", sorted(STRUCTURES))
def test_freeze_keeps_only_the_selected_experts(name):
    structure = STRUCTURES[name]
    selection = _selection(structure)
    specs = _island_specs(structure, selection)
    expected = {(layer, expert) for layer, ids in selection.trainable.items() for expert in ids}
    got = set()
    for spec in specs:
        assert ".mlp.experts." in spec.name, spec.name  # no dense, router, attention or embedding master
        match = re.search(r"layers\.(\d+)\.mlp\.experts\.linear_fc[12]\.weight(\d+)$", spec.name)
        got.add((int(match.group(1)), int(match.group(2))))
    assert got == expected
    partition = check_expert_partition(specs, subset=in_place.expert_subset(selection))
    assert partition.moe_layers == tuple(sorted(selection.trainable))
    with pytest.raises(ValueError, match="experts are not"):
        check_expert_partition(specs)  # without the subset the frozen experts look missing


def test_freeze_refusals():
    structure = STRUCTURES["qwen3_moe"]
    selection = _selection(structure)
    coords = in_place.RankCoordinates(layer_offset=0, ep_rank=0, ep_size=2)
    with pytest.raises(in_place.PartialExpertsError, match="trainer EP"):
        in_place.freeze_chunk(_chunk(structure, 0), selection=selection,
                              coords=in_place.RankCoordinates(layer_offset=0, ep_rank=0, ep_size=4))
    # a pipeline stage whose layer offset lands outside the selection
    with pytest.raises(in_place.PartialExpertsError, match="not in the selection"):
        in_place.freeze_chunk(_chunk(structure, 0), selection=selection,
                              coords=in_place.RankCoordinates(layer_offset=5, ep_rank=0, ep_size=2))
    # one grouped parameter for every local expert cannot be frozen per expert
    single = _chunk(structure, 0)
    experts = single.decoder.layers[0].mlp.experts
    experts.linear_fc1 = torch.nn.Module()
    experts.linear_fc1.weight = _param(2 * FFN, HIDDEN)
    with pytest.raises(in_place.PartialExpertsError, match="NVTE_GROUPED_LINEAR_SINGLE_PARAM=0"):
        in_place.freeze_chunk(single, selection=selection, coords=coords)
    # a selected expert with one projection missing
    broken = _chunk(structure, 0)
    fc2 = broken.decoder.layers[0].mlp.experts.linear_fc2
    for local in range(4):
        delattr(fc2, f"weight{local}")
    with pytest.raises(in_place.PartialExpertsError, match="projections"):
        in_place.freeze_chunk(broken, selection=selection, coords=coords)
    # the same chunk frozen twice
    twice = _chunk(structure, 0)
    in_place.freeze_chunk(twice, selection=selection, coords=coords)
    with pytest.raises(RuntimeError, match="configured twice"):
        in_place.freeze_chunk(twice, selection=selection, coords=coords)


def test_selection_is_checked_against_the_model_geometry():
    structure = STRUCTURES["deepseek_v3"]
    selection = _selection(structure)
    geometry = from_hf_config(structure["config"])
    in_place.check_against_geometry(selection, geometry, ep_size=4)
    with pytest.raises(in_place.PartialExpertsError, match="EP 4"):
        in_place.check_against_geometry(selection, geometry, ep_size=2)
    other = from_hf_config({**structure["config"], "first_k_dense_replace": 2})
    with pytest.raises(in_place.PartialExpertsError, match="MoE layers"):
        in_place.check_against_geometry(selection, other, ep_size=4)
    with pytest.raises(in_place.PartialExpertsError, match="mixture-of-experts"):
        in_place.check_against_geometry(selection, None, ep_size=4)


def test_provider_hook_freezes_actor_chunks_only(monkeypatch, tmp_path):
    structure = STRUCTURES["qwen3_moe"]
    selection = _selection(structure)
    path = tmp_path / "selection.json"
    selection.write(path)
    monkeypatch.setenv(in_place.ENV_SELECTION, str(path))
    in_place._cached_selection.cache_clear()
    monkeypatch.setattr(in_place, "_rank_coordinates",
                        lambda model, vp_stage: in_place.RankCoordinates(0, 1, 2))
    monkeypatch.setattr(in_place, "_check_router_bias", lambda model: None)

    def get_model_provider_func(args, role="actor"):
        return lambda pre_process=True, post_process=True, vp_stage=None: _chunk(structure, 1)

    fake = types.ModuleType(in_place._PROVIDER_MODULE)
    fake.get_model_provider_func = get_model_provider_func
    monkeypatch.setitem(sys.modules, in_place._PROVIDER_MODULE, fake)
    model_module = types.ModuleType(in_place._MODEL_MODULE)
    model_module.get_model_provider_func = get_model_provider_func
    monkeypatch.setitem(sys.modules, in_place._MODEL_MODULE, model_module)
    in_place.install()
    in_place.install()  # once per module
    assert model_module.get_model_provider_func is fake.get_model_provider_func
    actor = fake.get_model_provider_func(None, "actor")(vp_stage=None)
    trainable = {name for name, p in actor.named_parameters() if p.requires_grad}
    local = [e - 4 for e in selection.trainable[0] if e >= 4]
    assert {n for n in trainable if n.startswith("decoder.layers.0.")} == {
        f"decoder.layers.0.mlp.experts.linear_fc{b}.weight{i}" for b in (1, 2) for i in local}
    critic = fake.get_model_provider_func(None, role="critic")()
    assert all(p.requires_grad for p in critic.parameters())


def test_router_bias_must_stay_fixed():
    model = types.SimpleNamespace(config=types.SimpleNamespace(moe_router_enable_expert_bias=True,
                                                               moe_router_bias_update_rate=1e-3))
    with pytest.raises(in_place.PartialExpertsError, match="bias-update-rate 0"):
        in_place._check_router_bias(model)
    model.config.moe_router_bias_update_rate = 0.0
    in_place._check_router_bias(model)


@pytest.fixture(scope="module")
def syncer_binary() -> Path:
    if not Path(CARGO).exists():
        pytest.skip("needs cargo to build the syncer")
    subprocess.run([CARGO, "build", "-q"], cwd=ROOT / "syncer", check=True)
    return ROOT / "syncer/target/debug/yeto-syncer"


@pytest.mark.timeout(90)
@pytest.mark.parametrize("name", sorted(STRUCTURES))
def test_two_islands_merge_only_the_selected_experts_through_the_real_syncer(syncer_binary, tmp_path, name):
    structure = STRUCTURES[name]
    selection = _selection(structure)
    specs = _island_specs(structure, selection)
    check_expert_partition(specs, subset=in_place.expert_subset(selection))
    lay = ParameterLayout.create(algorithm="grpo", components=(ComponentIdentity("actor", "a" * 40, "b" * 64),),
                                 specs=specs, num_fragments=structure["ep"] * 2, fragment_strategy="owner_affine",
                                 matrix_merge="avg")
    num = lay.fragments.num_fragments
    rng = np.random.default_rng(3)
    init = {s.wire_name: rng.standard_normal(s.numel).astype(np.float32) for s in specs}
    deltas = [{s.wire_name: (0.01 * rng.standard_normal(s.numel)).astype(np.float32) for s in specs}
              for _ in range(2)]
    layout_names = {tensor for fragment in lay.fragments.fragments for tensor, _ in fragment.tensors}
    assert layout_names == set(init)  # the frozen experts and every dense tensor never reach a fragment

    def fragment_bytes(values, fid):
        return np.concatenate([values[n] for n, _ in lay.fragments.fragments[fid].tensors]).tobytes()

    port = free_port()
    proc = group_popen(
        [str(syncer_binary), "--port", str(port), "--learners", "2", "--quorum", "2", "--grace-ms", "20",
         "--quorum-timeout-s", "10", "--total-steps", str(num), "--pipeline", "1", "--sync-interval-steps", "0",
         "--delta-correction", "none", "--outer-lr", "1", "--outer-momentum", "0", "--learner-weight", "equal",
         "--final-state", str(tmp_path / "final.bin")],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    clients = [SyncerClient(("127.0.0.1", port), island, lay.fragments, dtype=DTYPE_F32, num_streams=1,
                            connect_timeout=10, max_reconnects=0, finalization_timeout=30) for island in range(2)]
    try:
        for client in clients:
            client.start()
        for fid in range(num):
            clients[0].send_init(fid, fragment_bytes(init, fid))
        pushed = [set(), set()]
        deadline = time.monotonic() + 45
        while any(len(p) < num for p in pushed):
            assert time.monotonic() < deadline, "syncer did not permit every fragment"
            for island, client in enumerate(clients):
                client.check_health()
                for permit in client.drain_pulls():
                    client.push_fragment(permit.fragment_id, permit.global_step, permit.round_attempt, 0, 1, 1, 10,
                                         fragment_bytes(deltas[island], permit.fragment_id))
                    pushed[island].add(permit.fragment_id)
            time.sleep(0.01)
        _, finals = clients[0].wait_for_final_fragments(timeout=30)
        assert {f.fragment_id for f in finals} == set(range(num))
        for final in finals:
            got = np.frombuffer(bytes(final.data), dtype=np.float32)
            offset = 0
            for tensor, numel in lay.fragments.fragments[final.fragment_id].tensors:
                expected = init[tensor] + (deltas[0][tensor] + deltas[1][tensor]) / 2
                np.testing.assert_allclose(got[offset:offset + numel], expected, rtol=1e-5, atol=1e-6)
                offset += numel
    finally:
        for client in clients:
            client.close()
        kill_group(proc)
        proc.wait(timeout=10)


# -- launcher: generic partial experts on the full-parameter preset's async rounds ---------------

from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: E402,F401


def _launch(tmp_path, structure, selection, *extra):
    import json

    from yeto.launcher import _prepare_rl_args, dry_run_plan

    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text(json.dumps(structure["config"]))
    path = tmp_path / "selection.json"
    selection.write(path)
    args = _launcher_args("ports", (
        "--model", str(model), "--rl-sync-preset", "full-parameter", "--rl-sync-full-parameter-rounds", "async",
        "--tuning", "full",
        "--rl-expert-selection", str(path), "--expert-parallel", str(structure["ep"]),
        "--rl-placement", "fixed-partition", "--rl-rollout-gpus", "4", "--rl-parameter-layout-sha256", "e" * 64,
        "--fragments", "4", "--local-rl-rounds-per-sync", "4", *extra), gpu="modal:8xh100,modal:8xh100")
    args.controller = "local"
    _prepare_rl_args(args)
    return args, dry_run_plan(args)


def test_generic_partial_experts_launch(tmp_path, monkeypatch):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "partial-experts-test-key")
    structure = dict(STRUCTURES["qwen3_moe"], ep=4)
    selection = sel.ExpertSelection.in_place("uniform", model="qwen3_moe", num_experts=8, num_trainable=4,
                                             ep_size=4, layers=(0, 1))
    args, plan = _launch(tmp_path, structure, selection)
    assert args.rl_sync_preset == "full-parameter" and args.rl_sync_full_parameter_rounds == "async"
    assert args._partial_experts["sha256"] == selection.sha256
    command = plan["island_requests"][0]["learner_command"]
    assert "--rl-expert-selection" in command and "--parameter-mode full" in command


@pytest.mark.parametrize("change, match", [
    (lambda s: sel.ExpertSelection.synthetic("uniform", model="kimi_k31", num_experts=8, num_trainable=4,
                                             ep_size=4, layers=(0, 1)), "layout 'slot_order'"),
    (lambda s: s.with_ep_size(2), "built for EP 2"),
    (lambda s: sel.ExpertSelection.in_place("uniform", model="m", num_experts=8, num_trainable=4, ep_size=4,
                                            layers=(1,)), "does not match the checkpoint"),
])
def test_generic_partial_experts_launch_refusals(tmp_path, monkeypatch, change, match):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "partial-experts-test-key")
    structure = dict(STRUCTURES["qwen3_moe"], ep=4)
    selection = sel.ExpertSelection.in_place("uniform", model="qwen3_moe", num_experts=8, num_trainable=4,
                                             ep_size=4, layers=(0, 1))
    with pytest.raises(ValueError, match=match):
        _launch(tmp_path, structure, change(selection))


def test_island_configures_the_runtime_from_the_bridge_provider(tmp_path, monkeypatch):
    from yeto.rl.adapters.miles import island_entry

    selection = _selection(STRUCTURES["deepseek_v3"])
    path = tmp_path / "selection.json"
    selection.write(path)
    for name in (in_place.ENV_SELECTION, in_place.GROUPED_SINGLE_PARAM_ENV):
        # setenv first so monkeypatch restores the absence: the island writes os.environ itself
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    installed = []
    monkeypatch.setattr(in_place, "install", lambda: installed.append(True))
    # the Megatron-Bridge DeepSeek provider: layer 0 dense as a moe_layer_freq mask
    provider = types.SimpleNamespace(num_layers=3, num_moe_experts=8, moe_router_topk=2, moe_layer_freq=[0, 1, 1])
    args = types.SimpleNamespace(rl_expert_selection=str(path), expert_parallel=4, rl_model_recipe="generic")
    argv = island_entry._configure_generic_partial_experts(args, provider)
    assert argv == in_place.BIAS_UPDATE_ARGV and installed == [True]
    import os

    assert os.environ[in_place.ENV_SELECTION] == str(path)
    assert os.environ[in_place.GROUPED_SINGLE_PARAM_ENV] == "0"
    assert island_entry._partial_experts_subset(args) == in_place.expert_subset(selection)
    assert island_entry._partial_experts_subset(types.SimpleNamespace(rl_model_recipe="kimi_k31",
                                                                      rl_expert_selection=str(path))) is None
    dense_first_two = types.SimpleNamespace(num_layers=3, num_moe_experts=8, moe_router_topk=2,
                                            moe_layer_freq=[0, 0, 1])
    with pytest.raises(in_place.PartialExpertsError, match="MoE layers"):
        island_entry._configure_generic_partial_experts(args, dense_first_two)


def test_kimi_k31_recipe_refuses_an_in_place_selection(tmp_path):
    from yeto.launch.k31_checks import check_expert_selection

    path = tmp_path / "selection.json"
    sel.ExpertSelection.in_place("uniform", model="kimi_k31", num_experts=16, num_trainable=8, ep_size=8,
                                 layers=(1, 2)).write(path)
    with pytest.raises(ValueError, match="needs a slot_order selection"):
        check_expert_selection(types.SimpleNamespace(rl_expert_selection=str(path), expert_parallel=8,
                                                     model="org/model"))


def test_generic_partial_experts_through_decoupled_equal_the_explicit_async_launch(tmp_path, monkeypatch, capsys):
    """--rl-sync-preset decoupled + --tuning full + an in_place selection is routed to the full-parameter
    preset's async rounds: the same plan as the explicit launch."""
    import json

    from yeto.launcher import _prepare_rl_args, dry_run_plan

    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "partial-experts-test-key")
    structure = dict(STRUCTURES["qwen3_moe"], ep=4)
    selection = sel.ExpertSelection.in_place("uniform", model="qwen3_moe", num_experts=8, num_trainable=4,
                                             ep_size=4, layers=(0, 1))
    _, explicit = _launch(tmp_path, structure, selection)
    model, path = tmp_path / "model", tmp_path / "selection.json"
    args = _launcher_args("ports", (
        "--model", str(model), "--rl-sync-preset", "decoupled", "--tuning", "full",
        "--rl-expert-selection", str(path), "--expert-parallel", "4",
        "--rl-placement", "fixed-partition", "--rl-rollout-gpus", "4", "--rl-parameter-layout-sha256", "e" * 64,
        "--fragments", "4", "--local-rl-rounds-per-sync", "4"), gpu="modal:8xh100,modal:8xh100")
    args.controller = "local"
    _prepare_rl_args(args)
    assert "[sync-route] decoupled -> full-parameter (async rounds)" in capsys.readouterr().out
    routed = dry_run_plan(args)
    assert json.dumps(routed, sort_keys=True, default=str) == json.dumps(explicit, sort_keys=True, default=str)


class _Embedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.word_embeddings = torch.nn.Module()
        self.word_embeddings.weight = _param(32, HIDDEN)

    def forward(self, tokens):
        return torch.nn.functional.embedding(tokens, self.word_embeddings.weight)


def _recompute_forward(chunk, tokens, n_local):
    """Embedding, then every layer's experts under a reentrant checkpoint (Megatron full recompute).

    The experts are SwiGLU (``silu(gate) * up``, as Megatron's MoE experts): every weight gets a nonzero
    gradient. A ReLU of half of fc1 left an expert with no active unit on the 4 tokens for some random
    weights (about 1 in 400 RNG states), so its fc1 and fc2 gradients were exactly zero."""

    hidden = chunk.embedding(tokens)
    for block in chunk.decoder.layers:
        experts = block.mlp.experts

        def layer(h, experts=experts):
            out = h
            for i in range(n_local):
                gate_up = h @ getattr(experts.linear_fc1, f"weight{i}").t()
                act = torch.nn.functional.silu(gate_up[..., :FFN]) * gate_up[..., FFN:]
                out = out + act @ getattr(experts.linear_fc2, f"weight{i}").t()
            return out

        hidden = torch.utils.checkpoint.checkpoint(layer, hidden, use_reentrant=True)
    return hidden


def test_selected_experts_get_gradients_under_full_recompute(monkeypatch):
    # its own RNG stream: the weights neither depend on what earlier tests drew nor shift later tests
    with torch.random.fork_rng():
        torch.manual_seed(0)
        _selected_experts_get_gradients_under_full_recompute(monkeypatch)


def _selected_experts_get_gradients_under_full_recompute(monkeypatch):
    structure = STRUCTURES["qwen3_moe"]
    selection = _selection(structure)
    n_local = selection.num_experts // structure["ep"]
    monkeypatch.setattr(in_place, "_rank_coordinates", lambda model, vp_stage: in_place.RankCoordinates(0, 0, 2))
    monkeypatch.setattr(in_place, "_check_router_bias", lambda model: None)

    def build():
        chunk = _chunk(structure, 0)
        chunk.embedding = _Embedding()
        return chunk

    tokens = torch.tensor([[1, 2, 3, 4]])
    bare = build()  # the failure mode: frozen embedding, nothing in the checkpoint's inputs needs grad
    in_place.freeze_chunk(bare, selection=selection, coords=in_place.RankCoordinates(0, 0, 2))
    with pytest.warns(UserWarning, match="requires_grad"):
        assert not _recompute_forward(bare, tokens, n_local).requires_grad

    chunk = in_place.wrap_model_provider(lambda **kw: build(), selection)(vp_stage=None)
    chunk.train()
    _recompute_forward(chunk, tokens, n_local).sum().backward()
    trainable = {name: p for name, p in chunk.named_parameters() if p.requires_grad}
    assert trainable and all(p.grad is not None and p.grad.abs().sum() > 0 for p in trainable.values())
    assert chunk.embedding.word_embeddings.weight.grad is None, "the embedding itself stays frozen"


def test_without_a_generic_selection_the_model_provider_is_untouched(monkeypatch):
    """kimi_k31 (and every run without an in_place selection) never sets ENV_SELECTION: the generic
    hook leaves Miles' provider as it is, so neither the freeze nor the embedding hook reaches it."""

    monkeypatch.delenv(in_place.ENV_SELECTION, raising=False)
    in_place._cached_selection.cache_clear()

    def get_model_provider_func(args, role="actor"):
        return lambda **kw: None

    fake = types.ModuleType(in_place._PROVIDER_MODULE)
    fake.get_model_provider_func = get_model_provider_func
    in_place._patch(fake)
    assert fake.get_model_provider_func is get_model_provider_func

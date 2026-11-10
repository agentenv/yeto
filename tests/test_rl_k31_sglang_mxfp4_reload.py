"""In-place MXFP4 expert reload on SGLang's SM100 trtllm-gen path
(``sglang_mxfp4_reload``), on CPU fakes of ``FusedMoE`` / ``Mxfp4MoEMethod``.

The fake method's cold transform is the whole-layer form of the pinned
``process_weights_after_loading`` (stacked over experts); the reload path must
give the same bytes as a cold start on the new checkpoint, keep every expert it
was not sent, never transform twice, and refuse a partly sent expert.
"""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from yeto.rl.adapters.sampler_engines import sglang_mxfp4_reload as reload  # noqa: E402
from yeto.rl.adapters.sampler_engines.sglang_server import SGLangServerEngine  # noqa: E402

E_GLOBAL, EP, INTER, HIDDEN = 6, 2, 64, 128


def _permute(x, tile, num_elts_per_sf=None):  # a fixed, shape-only row permutation
    n = x.shape[0]
    return torch.arange(n - 1, -1, -1) if num_elts_per_sf is None else torch.roll(torch.arange(n), 3)


def _interleave(x):  # a byte permutation of the scale block
    return x.flip(-1).contiguous()


MX = types.SimpleNamespace(_get_flashinfer_mxfp4_device_permute_indices=_permute,
                           nvfp4_block_scale_interleave=_interleave)


class FakeMethod:
    use_flashinfer, _fi_kernel, use_marlin, use_mega_moe, use_deep_gemm = True, "trtllm_sm100", False, False, False

    def __init__(self) -> None:
        self.intermediate_size_per_partition, self.hidden_size = INTER, HIDDEN
        self.processed = 0

    def process_weights_after_loading(self, layer):  # whole-layer cold transform (pinned form)
        self.processed += 1
        w13, s13 = layer.w13_weight.data, layer.w13_weight_scale.data
        w2, s2 = layer.w2_weight.data, layer.w2_weight_scale.data
        half = w13.shape[-2] // 2
        pair_idx = torch.empty(2 * half, dtype=torch.long)
        pair_idx[0::2] = torch.arange(half) + half
        pair_idx[1::2] = torch.arange(half)
        w13, s13 = w13[..., pair_idx, :].contiguous(), s13[..., pair_idx, :].contiguous()
        iw13, is13 = _permute(w13[0], 128), _permute(s13[0], 128, num_elts_per_sf=16)
        iw2, is2 = _permute(w2[0], 128), _permute(s2[0], 128, num_elts_per_sf=16)
        layer.w13_weight = torch.nn.Parameter(torch.stack([w13[i][iw13] for i in range(len(w13))]),
                                              requires_grad=False)
        layer.w13_weight_scale = torch.nn.Parameter(
            torch.stack([_interleave(s13[i][is13]) for i in range(len(s13))]).view(torch.float8_e4m3fn),
            requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(torch.stack([w2[i][iw2] for i in range(len(w2))]), requires_grad=False)
        layer.w2_weight_scale = torch.nn.Parameter(
            torch.stack([_interleave(s2[i][is2]) for i in range(len(s2))]).view(torch.float8_e4m3fn),
            requires_grad=False)


class FakeMoE(torch.nn.Module):
    """``FusedMoE``'s loader contract: ``weight_loader`` maps global -> local
    (EP), ``_weight_loader_impl`` copies one shard into ``param.data[local]``."""

    def __init__(self, ep_rank: int) -> None:
        super().__init__()
        self.layer_id, self.ep_rank, self.num_local = 3, ep_rank, E_GLOBAL // EP
        self.moe_runner_config = types.SimpleNamespace(gate_up_interleaved=False)
        n = self.num_local
        for name, shape, fill in (("w13_weight", (n, 2 * INTER, HIDDEN // 2), 0),
                                  ("w13_weight_scale", (n, 2 * INTER, HIDDEN // 32), 127),
                                  ("w2_weight", (n, HIDDEN, INTER // 2), 0),
                                  ("w2_weight_scale", (n, HIDDEN, INTER // 32), 127)):
            p = torch.nn.Parameter(torch.full(shape, fill, dtype=torch.uint8), requires_grad=False)
            p.weight_loader = self.weight_loader
            self.register_parameter(name, p)

    def weight_loader(self, param, loaded_weight, weight_name, shard_id, expert_id):
        local = expert_id - self.ep_rank * self.num_local
        if 0 <= local < self.num_local:
            self._weight_loader_impl(param=param, loaded_weight=loaded_weight, weight_name=weight_name,
                                     shard_id=shard_id, expert_id=local)

    def _weight_loader_impl(self, param, loaded_weight, weight_name, shard_id, expert_id):
        data = param.data[expert_id]
        if shard_id == "w2":
            data.copy_(loaded_weight)
        else:
            half = data.shape[0] // 2
            data[(0 if shard_id == "w1" else half):][:half].copy_(loaded_weight)


def checkpoint(seed: int) -> dict[tuple[int, str, str], torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    out = {}
    for e in range(E_GLOBAL):
        for proj, kind, shape in (("w1", "weight", (INTER, HIDDEN // 2)), ("w3", "weight", (INTER, HIDDEN // 2)),
                                  ("w2", "weight", (HIDDEN, INTER // 2)), ("w1", "scale", (INTER, HIDDEN // 32)),
                                  ("w3", "scale", (INTER, HIDDEN // 32)), ("w2", "scale", (HIDDEN, INTER // 32))):
            out[(e, proj, kind)] = torch.randint(0, 256, shape, generator=g, dtype=torch.uint8)
    return out


def load(layer, ckpt, experts=None):
    for (e, proj, kind), tensor in ckpt.items():
        if experts is not None and e not in experts:
            continue
        name = ("w2" if proj == "w2" else "w13") + ("_weight_scale" if kind == "scale" else "_weight")
        param = getattr(layer, name)
        param.weight_loader(param, tensor, f"model.layers.3.mlp.experts.{name}", shard_id=proj, expert_id=e)


def served(ep_rank: int, ckpt) -> tuple[FakeMoE, FakeMethod]:
    """A cold start: load everything, then the (patched) post-load step."""
    layer, method = FakeMoE(ep_rank), FakeMethod()
    load(layer, ckpt)
    patched(method, layer)
    return layer, method


def patched(method, layer):
    """``Mxfp4MoEMethod.process_weights_after_loading`` after ``install_on_module``."""
    module = types.SimpleNamespace(Mxfp4MoEMethod=type(method),
                                   _get_flashinfer_mxfp4_device_permute_indices=_permute,
                                   nvfp4_block_scale_interleave=_interleave)
    if not getattr(type(method), "_yeto_mxfp4_inplace", False):
        reload.install_on_module(module)  # patches FakeMethod in place (once per test session)
    return type(method).process_weights_after_loading(method, layer)


def raw(layer) -> dict[str, torch.Tensor]:
    return {n: getattr(layer, n).data.view(torch.uint8).clone() for n in reload.PARAMS}


@pytest.mark.parametrize("ep_rank", [0, 1])
def test_partial_reload_equals_cold_start_of_the_new_checkpoint(ep_rank):
    old, new = checkpoint(1), checkpoint(2)
    changed = {1, 4}
    target = {k: (new[k] if k[0] in changed else v) for k, v in old.items()}
    layer, method = served(ep_rank, old)
    assert layer._yeto_mxfp4_armed and method.processed == 1
    before = raw(layer)
    load(layer, new, experts=changed)
    patched(method, layer)  # end_weight_update / the disk path: no second transform
    assert method.processed == 1
    cold, _ = served(ep_rank, target)
    after, want = raw(layer), raw(cold)
    for name in reload.PARAMS:
        assert torch.equal(after[name], want[name]), name
    local_changed = {e - ep_rank * layer.num_local for e in changed} & set(range(layer.num_local))
    for slot in set(range(layer.num_local)) - local_changed:
        for name in reload.PARAMS:
            assert torch.equal(after[name][slot], before[name][slot])
    assert all(not torch.equal(after["w13_weight"][s], before["w13_weight"][s]) for s in local_changed)


def test_full_reload_is_idempotent_and_storage_stays():
    ckpt = checkpoint(5)
    layer, method = served(0, ckpt)
    pointers = {n: getattr(layer, n).data.data_ptr() for n in reload.PARAMS}
    before = raw(layer)
    load(layer, ckpt)
    patched(method, layer)
    assert method.processed == 1
    assert {n: getattr(layer, n).data.data_ptr() for n in reload.PARAMS} == pointers
    assert all(torch.equal(raw(layer)[n], before[n]) for n in reload.PARAMS)


def test_partly_sent_expert_is_refused_and_not_written():
    old, new = checkpoint(1), checkpoint(3)
    layer, method = served(0, old)
    before = raw(layer)
    half = {k: v for k, v in new.items() if k[0] == 0 and k[1] != "w2"}
    load(layer, half)
    with pytest.raises(reload.PartialExpertUpdate):
        patched(method, layer)
    assert all(torch.equal(raw(layer)[n], before[n]) for n in reload.PARAMS)
    assert layer._yeto_mxfp4_pending == {}


def test_bias_or_foreign_param_is_refused():
    layer, method = served(0, checkpoint(1))
    bias = torch.nn.Parameter(torch.zeros(layer.num_local, 2 * INTER), requires_grad=False)
    with pytest.raises(RuntimeError, match="cannot be reloaded in place"):
        layer.weight_loader(bias, torch.zeros(INTER), "experts.w13_weight_bias", "w1", 0)


def test_other_backends_are_not_armed():
    marlin = FakeMethod()
    marlin.use_marlin = True
    assert not reload.is_trtllm_sm100(marlin)
    cutlass = FakeMethod()
    cutlass._fi_kernel = "cutlass_sm90"
    assert not reload.is_trtllm_sm100(cutlass)
    assert reload.is_trtllm_sm100(FakeMethod())


def test_interleaved_layout_matches_the_pinned_swap():
    x = torch.arange(2 * 4 * 3, dtype=torch.uint8).reshape(2, 4, 3)
    swapped = reload._swap_every_two_rows(x, -2)
    assert torch.equal(swapped[:, 0], x[:, 1]) and torch.equal(swapped[:, 1], x[:, 0])
    assert torch.equal(swapped[:, 2], x[:, 3])


def _engine(**server):
    from yeto.rl.adapters.sampler_engines import from_spec

    spec = {"factory": "yeto.rl.adapters.sampler_engines.from_spec:build", "adapter": "sglang-server",
            "server": {"base_dir": "/m", "port": 30000, "python": "python", "card_model": "B300",
                       "device_family": "nvidia", "engine_version": "sglang-miles-kimi-k31@35708f76",
                       "precision": "mxfp4", "quant_scheme": "MXFP4_EXPERTS_RTN", "topology": "tp8",
                       "max_model_len": 4096, **server},
            "options": {"tp": 8}}
    engine = from_spec.build(spec)
    assert isinstance(engine, SGLangServerEngine)
    return engine


def test_adapter_sets_the_hook_env_for_in_place_mxfp4_on_cuda():
    assert _engine(in_place_reload=True).env()[reload.ENV] == "1"
    assert reload.ENV not in _engine(in_place_reload=False).env()
    assert _engine(in_place_reload=True, extra_env={reload.ENV: "0"}).env()[reload.ENV] == "0"


def _hot_engine():
    from yeto.rl.adapters.sampler_engines import from_spec

    spec = {"factory": "yeto.rl.adapters.sampler_engines.from_spec:build", "adapter": "sglang-server",
            "server": {"base_dir": "/m", "port": 30000, "python": "python", "card_model": "B300",
                       "device_family": "nvidia", "engine_version": "sglang-miles-kimi-k31@35708f76",
                       "precision": "mxfp4", "quant_scheme": "MXFP4_EXPERTS_RTN", "topology": "tp8",
                       "max_model_len": 4096, "in_place_reload": True},
            "options": {"tp": 8, "expert_file_update": "auto"}}
    return from_spec.build(spec)


def test_hot_mxfp4_server_registers_the_rank_local_file_loader():
    argv = _hot_engine().argv(__import__("pathlib").Path("/m"))
    assert argv[argv.index("--custom-weight-loader") + 1] == reload.FILE_LOADER


def test_expert_update_sends_file_names_not_tensors(tmp_path):
    from yeto.rl.adapters.sampler_engines import k31_release

    engine = _hot_engine()
    calls = []
    engine.call = lambda method, path, body=None, **kw: calls.append((path, body)) or {"success": True}
    engine._serialize = lambda pairs: pairs  # inspect the payload
    files = tuple(f"model-{i:05d}.safetensors" for i in range(10))
    plan = k31_release.ExpertUpdatePlan(release_dir=tmp_path, files=files, tensors={f: ("t",) for f in files})
    record = engine._send_tensors(plan, version=7)
    sent = [b for p, b in calls if p == "/update_weights_from_tensor"]
    assert [p for p, _ in calls][0] == "/begin_weight_update" and calls[-1][0] == "/end_weight_update"
    assert len(sent) == 2 and record["rank_local_files"] and sent[-1]["weight_version"] == "7"
    assert all(b["load_format"] == reload.FILE_LOADER and len(b["serialized_named_tensors"]) == 8 for b in sent)
    names = [reload.parse_files_command(b["serialized_named_tensors"][0][0][0]) for b in sent]
    assert names[0] == {"dir": str(tmp_path), "files": list(files[:8])} and names[1]["files"] == list(files[8:])


def test_load_files_feeds_only_expert_tensors_to_load_weights(tmp_path):
    save_file = pytest.importorskip("safetensors.torch").save_file
    expert = "language_model.model.layers.3.block_sparse_moe.experts.5.w1.weight_packed"
    save_file({expert: torch.ones(2, 2, dtype=torch.uint8), "lm_head.weight": torch.zeros(2)},
              str(tmp_path / "a.safetensors"))
    seen = []
    model = types.SimpleNamespace(load_weights=lambda it: seen.extend(n for n, _ in it))
    reload.load_files(model, [(reload.files_command_name(str(tmp_path), ["a.safetensors"]), 0)])
    assert seen == [expert]
    with pytest.raises(ValueError):
        reload.load_files(model, [("yeto-sparse.x", 0)])


def test_rank_reads_only_its_own_experts(tmp_path):
    save_file = pytest.importorskip("safetensors.torch").save_file
    name = "language_model.model.layers.3.block_sparse_moe.experts.{}.w2.weight_scale"
    save_file({name.format(e): torch.ones(1, dtype=torch.uint8) for e in range(E_GLOBAL)}, str(tmp_path / "a.safetensors"))
    model = torch.nn.Module()
    model.moe = FakeMoE(ep_rank=1)
    model.moe._map_global_expert_id_to_local_expert_id = lambda e: e - 3 if 3 <= e < 6 else -1
    keep = reload.local_expert_filter(model)
    got = [n for n, _ in reload.expert_tensors_of_files(str(tmp_path), ["a.safetensors"], keep)]
    assert got == [name.format(e) for e in (3, 4, 5)]


def _write_ckpt(directory, weight_map):
    import json

    directory.mkdir(parents=True, exist_ok=True)
    for file in set(weight_map.values()):
        (directory / file).write_bytes(b"x")
    (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


def test_overlay_of_a_partial_release_is_planned_from_its_release_files(tmp_path):
    from yeto.rl.adapters.sampler_engines import k31_release
    from yeto.rl.engine.release_overlay import overlay_release

    e = "language_model.model.layers.{}.block_sparse_moe.experts.{}.w1.weight_packed"
    base, r1, r2, other = tmp_path / "base", tmp_path / "r1", tmp_path / "r2", tmp_path / "r3"
    _write_ckpt(base, {e.format(1, 0): "model-1.safetensors", e.format(1, 1): "model-1.safetensors",
                       "lm_head.weight": "model-9.safetensors"})
    _write_ckpt(r1, {e.format(1, 0): "model-00001.safetensors"})
    _write_ckpt(r2, {e.format(1, 0): "model-00001.safetensors"})
    _write_ckpt(other, {e.format(1, 1): "model-00001.safetensors"})
    o1 = overlay_release(base, r1, tmp_path / "o1")
    o2 = overlay_release(base, r2, tmp_path / "o2")
    o3 = overlay_release(base, other, tmp_path / "o3")
    plan = k31_release.overlay_update_plan(o1, base, base)
    assert plan.files == ("release-model-00001.safetensors",) and plan.experts == {1: (0,)}
    assert k31_release.overlay_update_plan(o2, o1, base).files == plan.files  # same selection over an overlay
    with pytest.raises(k31_release.FileSparseUnavailable, match="overlay_selection_changed"):
        k31_release.overlay_update_plan(o3, o1, base)
    assert k31_release.overlay_update_plan(base, base, base) is None


def test_overlay_plan_over_a_base_that_is_a_view_of_symlinks(tmp_path):
    """The base is a Modal Volume HF view (relative symlinks into the real
    checkpoint): the overlay still counts as laid over it, release after release."""
    import os

    from yeto.rl.adapters.sampler_engines import k31_release
    from yeto.rl.engine.release_overlay import overlay_release

    e = "language_model.model.layers.{}.block_sparse_moe.experts.{}.w1.weight_packed"
    real, view = tmp_path / "Kimi" , tmp_path / "hf" / "Kimi" / "rev"
    _write_ckpt(real, {e.format(1, 0): "model-1.safetensors", "lm_head.weight": "model-9.safetensors"})
    view.mkdir(parents=True)
    for child in real.iterdir():
        os.symlink(os.path.relpath(child, view), view / child.name)
    r1, r2 = tmp_path / "r1", tmp_path / "r2"
    _write_ckpt(r1, {e.format(1, 0): "model-00001.safetensors"})
    _write_ckpt(r2, {e.format(1, 0): "model-00001.safetensors"})
    o1 = overlay_release(view, r1, tmp_path / "o1")
    o2 = overlay_release(view, r2, tmp_path / "o2")
    assert k31_release.overlay_update_plan(o1, view, view).files == ("release-model-00001.safetensors",)
    assert k31_release.overlay_update_plan(o2, o1, view).files == ("release-model-00001.safetensors",)

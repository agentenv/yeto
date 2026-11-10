"""Kimi-K3.1 ``--tuning lora`` next to the unchanged ``--tuning full`` form (CPU).

1. tuning rules (``yeto.rl.engine.kimi_k31_lora``) and their refusals;
2. the Miles argv of both forms: LoRA flags, shared-outer expert LoRA, no selection,
   no base CPU backup, R3 and full recompute in both; the full argv unchanged;
3. launcher -> island command -> run config -> Miles argv for a LoRA trainer island;
4. the LoRA release manifest and its record round trip;
5. the sampler side: a LoRA release merged into a BF16 + MXFP4 base for an engine
   without adapter support, and every refusal before the engine is touched.
"""

from __future__ import annotations

import dataclasses
import json
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from test_rl_k31_e2e_cpu import _hf_config, _k31_argv, _k31_config, _value
from test_rl_k31_synced_cpu import _k31, _launcher_args, _no_modal_listing, _fake_sky  # noqa: F401
from yeto.rl.adapters.miles import config as mc
from yeto.rl.adapters.miles import island_entry
from yeto.rl.adapters.miles.models.kimi_k31 import ports_recipe as k31_recipe
from yeto.rl.adapters.miles.models.kimi_k31 import provider as k31_provider
from yeto.rl.algos import score_centering as sc
from yeto.rl.engine import kimi_k31_lora as kl
from yeto.rl.engine import run_config as rc
from yeto.rl.engine.lora_release_merge import LoraMergeError, merge_lora_release, plan_merge
from yeto.rl.engine.policy_release import LORA, PolicyRelease, ReleaseNotLoaded
from yeto.rl.engine.sampler_checks import PortReleaseLoader, PostLoadChecks, overlaid_release_scheme


@pytest.fixture(autouse=True)
def _island_hmac_key(monkeypatch):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "k31-lora-test-key")


# ------------------------------------------------------------------ 1. rules
def test_tuning_rules_and_refusals():
    assert kl.check_tuning("full") == "full"
    assert kl.check_tuning("full", expert_selection="/s.json", routed_expert_qat="mxfp4") == "full"
    assert kl.check_tuning("lora", lora_targets="attention") == "lora"
    assert kl.target_modules("attention") == ("attn",)
    assert kl.target_modules("all-linear") == ("all-linear",)
    assert kl.target_modules("auto") == kl.target_modules("attention")  # as --help states
    assert not kl.adapts_routed_experts("auto") and kl.extra_lora_argv("auto") == [kl.SHARED_OUTER_FLAG]
    kl.check_tuning("lora", lora_targets="auto", policy_sync=True)
    # Miles native kimi_k3 LoRA asserts the shared-outer layout for any targets (GPU run 2026-11-10)
    assert kl.extra_lora_argv("attention") == kl.extra_lora_argv("all-linear") == [kl.SHARED_OUTER_FLAG]
    for kwargs, match in (({"expert_selection": "/s.json"}, "rl-expert-selection"),
                          ({"routed_expert_qat": "mxfp4"}, "routed-expert-qat"),
                          ({"lora_targets": "attention-routed-experts"}, "not a Kimi-K3.1 LoRA preset"),
                          ({"policy_sync": True, "lora_targets": "all-linear"}, "--lora-targets attention")):
        with pytest.raises(kl.KimiK31LoraError, match=match):
            kl.check_tuning("lora", **{"lora_targets": "attention", **kwargs})
    with pytest.raises(kl.KimiK31LoraError, match="trains"):
        kl.check_tuning("qlora")


# ------------------------------------------------------------------ 2. Miles argv
def _lora(cfg, targets: str, rank: int = 32):
    return dataclasses.replace(cfg, trainable=rc.TrainableConfig(
        parameter_mode="lora", lora_rank=rank, lora_targets=targets,
        target_modules=kl.target_modules(targets), expert_full_count=0))


def test_both_tunings_render_their_miles_argv():
    full = _k31_argv(_k31_config("/run/expert_selection.json"), sc.score_centering_spec())
    assert _value(full, "--partial-experts-selection") == "/run/expert_selection.json"
    assert "--lora-rank" not in full and kl.SHARED_OUTER_FLAG not in full
    for targets, shared_outer in (("attention", True), ("all-linear", True)):
        argv = _k31_argv(_lora(_k31_config(None), targets), sc.score_centering_spec())
        assert _value(argv, "--lora-rank") == "32" and _value(argv, "--lora-alpha") == "32"
        assert _value(argv, "--lora-type") == "canonical_lora"  # native LoRA requires it
        assert _value(argv, "--target-modules") == ",".join(kl.target_modules(targets))
        assert (kl.SHARED_OUTER_FLAG in argv) is shared_outer
        assert "--partial-experts-selection" not in argv and "--routed-expert-qat" not in argv
        assert "--lora-base-cpu-backup" not in argv
        # shared by both forms: raw bridge, R3, frozen router bias, full recompute for MXFP4 experts
        assert _value(argv, "--megatron-to-hf-mode") == "raw" and "--use-rollout-routing-replay" in argv
        assert _value(argv, "--moe-router-bias-update-rate") == "0"
        assert _value(argv, "--recompute-granularity") == "full"
        # the model flags of both forms are the same
        for flag in ("--num-experts", "--num-layers", "--moe-layer-freq", "--model-name"):
            assert _value(argv, flag) == _value(full, flag), flag
        assert [t for t in argv if t.startswith("--spec")] == [t for t in full if t.startswith("--spec")]


def test_colocated_lora_drops_the_base_cpu_backup_and_refusals_reach_the_argv():
    cfg = _lora(_k31_config(None), "all-linear")
    colocated = dataclasses.replace(cfg, parallel=dataclasses.replace(cfg.parallel, dedicated_rollout_gpus=0,
                                                                      visible_gpus_per_node=8))
    try:
        argv = _k31_argv(colocated, sc.score_centering_spec())
    except mc.MilesConfigError:
        argv = None  # the layout helper may not build a colocated placement here
    if argv is not None:
        assert "--lora-base-cpu-backup" not in argv
    with pytest.raises(mc.MilesConfigError, match="rl-expert-selection belongs"):
        _k31_argv(_lora(_k31_config("/run/s.json"), "attention"), sc.score_centering_spec())
    with pytest.raises(k31_recipe.KimiK31RecipeError, match="LoRA form"):
        k31_recipe.recipe_argv(selection_path=None, seq_length=4096, routed_expert_qat="mxfp4")
    lora_recipe = k31_recipe.recipe_argv(selection_path=None, seq_length=4096, include_model_argv=False)
    assert list(lora_recipe) == list(k31_recipe.RECIPE_ARGV)
    full_recipe = k31_recipe.recipe_argv(selection_path="s.json", seq_length=4096, include_model_argv=False)
    assert full_recipe == [*k31_recipe.RECIPE_ARGV, k31_recipe.SELECTION_FLAG, "s.json"]


# ------------------------------------------------------------------ 3. launcher -> island -> argv
def _lora_launch(tmp_path: Path, targets: str = "attention", *, trainer_role: bool = False) -> tuple[str, ...]:
    """The K3.1 launch with --tuning lora: no selection. ``trainer_role``: the A-card form
    (samples from the store); else one colocated island that samples itself (SGLang)."""
    full = list(_k31(tmp_path, synced=False))
    i = full.index("--rl-expert-selection")
    del full[i:i + 2]
    full[full.index("--tuning") + 1] = "lora"
    if not trainer_role:
        for flag in ("--rl-sample-role", "--rl-max-policy-age"):
            i = full.index(flag)
            del full[i:i + 2]
    return (*full, "--lora-r", "32", "--lora-targets", targets)


def _island(tmp_path: Path, launch: tuple[str, ...]):
    from yeto.launcher import _prepare_rl_args, dry_run_plan

    args = _launcher_args("ports", launch, gpu="modal:1x8xb300")
    args.controller = "local"
    _prepare_rl_args(args)
    plan = dry_run_plan(args)
    (island,) = plan["island_requests"]
    return args, island["learner_command"]


def test_launcher_island_and_run_config_of_a_lora_trainer(tmp_path):
    args, command = _island(tmp_path, _lora_launch(tmp_path))
    assert args.tuning == "lora"
    import re
    assert "--parameter-mode full" not in command and "--lora-targets attention" in command, re.findall(
        r"--(?:parameter-mode|lora-[a-z]+|tuning|rl-model-recipe|rl-expert-selection) \S+", command)
    assert "--rl-expert-selection" not in command and "--rl-model-recipe kimi_k31" in command
    tail = command.split("-m yeto.rl.adapters.miles.island_entry", 1)[1]
    tokens = [{"$LEARNER_ID": "0", "$SYNCER_ADDR": "10.0.0.1:29400"}.get(t, t) for t in shlex.split(tail)]
    # the launcher mounts the spec file at ~/yeto-rl/; read the local copy
    tokens = [str(tmp_path / "algorithm_spec.json") if t.endswith("/algorithm_spec.json") else t for t in tokens]
    iargs = island_entry.parse_args(tokens)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(_hf_config()))
    prompts = tmp_path / "p.jsonl"
    prompts.write_text('{"problem": "q", "answer": "a"}\n')
    provider = k31_provider.kimi_k31_provider(model)
    config = rc.resolve_rl_run_config(iargs, model_path=model, prompt_path=prompts, provider=provider,
                                      target_modules=list(kl.target_modules(iargs.lora_targets)),
                                      yeto_policy_sync=False, verify_ref_load=False)
    assert config.trainable.parameter_mode == "lora" and config.model_recipe.expert_selection is None
    argv = list(island_entry.build_ports_launch(iargs, config).argv)
    assert _value(argv, "--target-modules") == "attn" and "--partial-experts-selection" not in argv
    assert "--use-rollout-routing-replay" in argv and "--colocate" in argv
    assert "--lora-base-cpu-backup" not in argv
    # the island's manifest of a LoRA release
    manifest = island_entry._kimi_k31_release_manifest(iargs, provider)
    assert manifest == kl.manifest_fields(rank=32, lora_targets="attention")
    # refusals at the launcher: a selection with LoRA, a non-K3.1 target preset
    with pytest.raises(ValueError, match="rl-expert-selection belongs"):
        _island(tmp_path, (*_lora_launch(tmp_path), "--rl-expert-selection", str(tmp_path / "expert_selection.json")))
    with pytest.raises(ValueError, match="not a Kimi-K3.1 LoRA preset"):
        _island(tmp_path, _lora_launch(tmp_path, "attention-routed-experts"))


def test_routed_expert_lora_is_refused_for_samplers_that_cannot_serve_it(tmp_path):
    for engine in (None, "vllm"):
        with pytest.raises(kl.KimiK31LoraError, match="lora-targets attention"):
            kl.check_sampler_engine("all-linear", sample_role="trainer", sampler_engine=engine)
    kl.check_sampler_engine("all-linear", sample_role="trainer", sampler_engine="sglang")
    kl.check_sampler_engine("all-linear", sample_role="both", sampler_engine=None)  # colocated SGLang
    for targets in ("attention", "auto"):
        kl.check_sampler_engine(targets, sample_role="trainer", sampler_engine="vllm")
    launch = (*_lora_launch(tmp_path, "all-linear", trainer_role=True), "--rl-sampler-engine", "vllm")
    with pytest.raises(ValueError, match="MXFP4"):
        _island(tmp_path, launch)


def test_the_full_form_still_refuses_a_missing_selection(tmp_path):
    full = list(_k31(tmp_path, synced=False))
    i = full.index("--rl-expert-selection")
    del full[i:i + 2]
    with pytest.raises(ValueError, match="needs --rl-expert-selection"):
        _island(tmp_path, tuple(full))


# ------------------------------------------------------------------ 4. release manifest
def test_lora_release_manifest_round_trips_through_the_record():
    manifest = kl.manifest_fields(rank=16, lora_targets="all-linear")
    block = kl.lora_manifest(manifest)
    assert block["routed_experts"] and block["experts_shared_outer"] and block["target_modules"] == ["all-linear"]
    record = dataclasses.asdict(PolicyRelease(seq=1, version=1, policy_hash="h", path="r1-v1", weights=LORA,
                                              manifest=manifest))
    back = PolicyRelease.from_record(json.loads(json.dumps(record)))
    assert back.weights == LORA and kl.lora_manifest(back.manifest) == block
    with pytest.raises(kl.KimiK31LoraError):
        kl.manifest_fields(rank=0, lora_targets="attention")


# ------------------------------------------------------------------ 5. sampler: merge into the base
Q = "language_model.model.layers.0.self_attn.q_a_proj"
O = "language_model.model.layers.1.self_attn.o_proj"
EXPERT = "language_model.model.layers.1.block_sparse_moe.experts.3.w1"


def _base(tmp_path: Path) -> Path:
    base = tmp_path / "base"
    base.mkdir()
    torch.manual_seed(0)
    dense = {f"{Q}.weight": torch.randn(6, 8).to(torch.bfloat16), f"{O}.weight": torch.randn(8, 4).to(torch.bfloat16)}
    experts = {f"{EXPERT}.weight_packed": torch.randint(0, 255, (6, 4), dtype=torch.uint8),
               f"{EXPERT}.weight_scale": torch.randint(0, 255, (6, 1), dtype=torch.uint8)}
    save_file(dense, str(base / "model-00001.safetensors"))
    save_file(experts, str(base / "model-00002.safetensors"))
    weight_map = {**{k: "model-00001.safetensors" for k in dense}, **{k: "model-00002.safetensors" for k in experts}}
    (base / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    (base / "config.json").write_text(json.dumps({"text_config": {"quantization_config": {
        "quant_method": "compressed-tensors", "format": "mxfp4-pack-quantized"}}}))
    return base


def _adapter(path: Path, tensors: dict, *, rank: int = 2, alpha: int = 4) -> Path:
    path.mkdir(parents=True)
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path / "adapter_model.safetensors"))
    (path / "adapter_config.json").write_text(json.dumps({"r": rank, "lora_alpha": alpha, "peft_type": "LORA"}))
    return path


def _attention_adapter(path: Path) -> tuple[Path, dict]:
    torch.manual_seed(1)
    t = {f"base_model.model.{Q}.lora_A.weight": torch.randn(2, 8), f"base_model.model.{Q}.lora_B.weight": torch.randn(6, 2),
         f"base_model.model.{O}.lora_A.weight": torch.randn(2, 4), f"base_model.model.{O}.lora_B.weight": torch.randn(8, 2)}
    return _adapter(path, {k: v.to(torch.bfloat16) for k, v in t.items()}), t


def test_attention_adapter_merges_exactly_and_keeps_the_mxfp4_experts(tmp_path):
    base = _base(tmp_path)
    adapter, t = _attention_adapter(tmp_path / "release")
    out = merge_lora_release(base, adapter, tmp_path / "release.merged")
    index = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    assert index[f"{Q}.weight"] == index[f"{O}.weight"] == "release-lora-merged.safetensors"
    assert index[f"{EXPERT}.weight_packed"] == "model-00002.safetensors"
    # the expert file is the base's own bytes (a symlink), the config is the base's
    assert (out / "model-00002.safetensors").resolve() == (base / "model-00002.safetensors").resolve()
    assert json.loads((out / "config.json").read_text()) == json.loads((base / "config.json").read_text())
    merged = load_file(str(out / "release-lora-merged.safetensors"))
    original = load_file(str(base / "model-00001.safetensors"))
    for module in (Q, O):
        a = t[f"base_model.model.{module}.lora_A.weight"].to(torch.bfloat16).float()
        b = t[f"base_model.model.{module}.lora_B.weight"].to(torch.bfloat16).float()
        want = (original[f"{module}.weight"].float() + 2.0 * (b @ a)).to(torch.bfloat16)
        assert merged[f"{module}.weight"].dtype == torch.bfloat16
        assert torch.equal(merged[f"{module}.weight"], want)
    # rerunning over an existing output replaces it
    assert merge_lora_release(base, adapter, tmp_path / "release.merged") == out


def test_merge_refusals(tmp_path):
    base = _base(tmp_path)
    cases = {
        "quantized_base": {f"{EXPERT}.lora_A.weight": torch.zeros(2, 4), f"{EXPERT}.lora_B.weight": torch.zeros(6, 2)},
        "stacked": {"language_model.model.layers.1.block_sparse_moe.experts.w1.lora_A.weight": torch.zeros(1, 2, 4),
                    "language_model.model.layers.1.block_sparse_moe.experts.w1.lora_B.weight": torch.zeros(4, 6, 2)},
        "unknown_module": {"model.layers.9.mlp.up_proj.lora_A.weight": torch.zeros(2, 8),
                           "model.layers.9.mlp.up_proj.lora_B.weight": torch.zeros(6, 2)},
        "unpaired": {f"{Q}.lora_A.weight": torch.zeros(2, 8)},
        "shape": {f"{Q}.lora_A.weight": torch.zeros(2, 8), f"{Q}.lora_B.weight": torch.zeros(6, 3)},
    }
    for reason, tensors in cases.items():
        with pytest.raises(LoraMergeError) as error:
            plan_merge(base, tensors)
        assert error.value.reason == reason, (reason, error.value)
    wrong = _adapter(tmp_path / "wrong", {f"{Q}.lora_A.weight": torch.zeros(2, 4), f"{Q}.lora_B.weight": torch.zeros(6, 2)})
    with pytest.raises(LoraMergeError, match="shape"):
        merge_lora_release(base, wrong, tmp_path / "wrong.merged")


def _engine(base: Path):
    return SimpleNamespace(config=SimpleNamespace(base_dir=base), served_version=lambda: 0,
                           identity=SimpleNamespace(quant_scheme="MXFP4_EXPERTS_RTN"))


def test_port_loader_serves_a_merged_lora_release_in_the_base_scheme(tmp_path):
    base = _base(tmp_path)
    adapter, _ = _attention_adapter(tmp_path / "r1-v1")
    release = PolicyRelease(seq=1, version=1, policy_hash="h", path="r1-v1", weights=LORA,
                            manifest=kl.manifest_fields(rank=2, lora_targets="attention"))
    assert overlaid_release_scheme(release, adapter, base, "MXFP4_EXPERTS_RTN") == "MXFP4_EXPERTS_RTN"
    loader = PortReleaseLoader(_engine(base), PostLoadChecks(), lambda *a, **k: None)
    served = []
    loader._serve = lambda release, directory, version, first: served.append(Path(directory))
    loader(release, adapter)
    assert served == [tmp_path / "r1-v1.merged"]
    assert (served[0] / "release-lora-merged.safetensors").is_file()


def test_port_loader_refuses_lora_releases_it_cannot_merge(tmp_path):
    base = _base(tmp_path)
    adapter, _ = _attention_adapter(tmp_path / "r1-v1")
    events = []
    served = []

    def loader_for(engine):
        loader = PortReleaseLoader(engine, PostLoadChecks(), lambda e, **f: events.append((e, f)))
        loader._serve = lambda *a, **k: served.append(a)
        return loader

    expert_release = PolicyRelease(seq=1, version=1, policy_hash="h", path="r1-v1", weights=LORA,
                                   manifest=kl.manifest_fields(rank=2, lora_targets="all-linear"))
    with pytest.raises(ReleaseNotLoaded, match="routed experts") as error:
        loader_for(_engine(base))(expert_release, adapter)
    assert error.value.reason == "release_refused:lora_merge"
    no_base = _engine(base)
    no_base.config.base_dir = None
    plain = dataclasses.replace(expert_release, manifest=None)
    with pytest.raises(ReleaseNotLoaded, match="(base_dir|scheme)"):
        loader_for(no_base)(plain, adapter)
    bad = _adapter(tmp_path / "r2-v2", {f"{EXPERT}.lora_A.weight": torch.zeros(2, 4),
                                        f"{EXPERT}.lora_B.weight": torch.zeros(6, 2)})
    with pytest.raises(ReleaseNotLoaded, match="quantized_base"):
        loader_for(_engine(base))(dataclasses.replace(plain, seq=2, version=2, path="r2-v2"), bad)
    assert served == []  # the engine was never touched


def test_a_multi_gpu_lora_buffer_trainer_runs_trainer_only_without_offload(tmp_path):
    """The A-card form (trainer island, samples from the store, engines on NC): a
    trainer-only fixed partition, never colocated, never offloaded, so the LoRA export
    reads the adapter from the resident trainer (release_plugin.write_lora_adapter).
    It used to fall through to the colocated placement and its one-GPU guard."""
    launch = _lora_launch(tmp_path, trainer_role=True)
    args, command = _island(tmp_path, launch)
    tail = command.split("-m yeto.rl.adapters.miles.island_entry", 1)[1]
    tokens = [{"$LEARNER_ID": "0", "$SYNCER_ADDR": "10.0.0.1:29400"}.get(t, t) for t in shlex.split(tail)]
    tokens = [str(tmp_path / "algorithm_spec.json") if t.endswith("/algorithm_spec.json") else t for t in tokens]
    iargs = island_entry.parse_args(tokens)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(_hf_config()))
    prompts = tmp_path / "p.jsonl"
    prompts.write_text('{"problem": "q", "answer": "a"}\n')
    config = rc.resolve_rl_run_config(iargs, model_path=model, prompt_path=prompts,
                                      provider=k31_provider.kimi_k31_provider(model),
                                      target_modules=["attn"], yeto_policy_sync=False, verify_ref_load=False)
    assert not config.parallel.colocated and config.parallel.dedicated_rollout_gpus == 0
    assert mc.placement_request(config).trainer_only
    argv = list(island_entry.build_ports_launch(iargs, config).argv)
    assert "--debug-train-only" in argv and "--no-offload-train" in argv
    for flag in ("--colocate", "--offload-train", "--lora-base-cpu-backup", "--rollout-num-gpus",
                 "--update-weight-transfer-mode", "--partial-experts-selection"):
        assert flag not in argv, flag
    assert _value(argv, "--target-modules") == "attn" and "--use-rollout-routing-replay" in argv


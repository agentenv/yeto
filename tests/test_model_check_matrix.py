"""`yeto model check` against the launcher and the island run config, cell by cell.

Every fixture model x trainable form x sync preset x backend: the check's
``launcher`` verdict must equal what ``yeto launch`` (``_prepare_rl_args``)
does with the check's own ``launch_argv``, and its run-config verdict must equal
what ``resolve_rl_run_config`` does with the island's arguments and a provider
built from the same config.json. A rule that changes on one side fails here.
CPU only; no network, no weights.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import types
from functools import lru_cache
from pathlib import Path

import pytest

from yeto.rl.model_registry.check import (
    BACKENDS, FAIL, FORMS, ISLAND, PENDING, PRESETS, CheckRequest, actor_gpus, check_model, render_text,
    report_dict,
)

DATA = Path(__file__).parent / "data" / "model_registry"
MODELS = {path.stem: path for path in sorted((DATA / "configs").glob("*.json"))}
MODELS["kimi_k31"] = DATA / "kimi_k31.json"
# the families the onboarding brief names, each present at least once
REQUIRED = {"qwen3_8b", "qwen3_30b_a3b", "llama3_1_8b", "mistral_7b", "mixtral_8x7b", "deepseek_v3",
            "deepseek_v4_flash", "glm4_9b", "glm4_5_air", "olmoe_1b_7b", "gemma3_12b", "kimi_k31"}
HEX = "e" * 64


def _config(name: str) -> dict:
    return json.loads(MODELS[name].read_text())


def test_the_fixtures_cover_the_onboarding_families():
    assert REQUIRED <= set(MODELS)


# ----------------------------------------------------------------------------- launcher layer
@pytest.fixture(scope="module")
def launcher(tmp_path_factory):
    """``_prepare_rl_args`` on a full ``yeto launch`` command line (fake sky, no Modal listing)."""

    sys.path.insert(0, str(Path(__file__).parent))
    from test_rl_launcher import _Resources, _Storage, _StorageMode, _Task

    patch = pytest.MonkeyPatch()
    patch.setitem(sys.modules, "sky", types.SimpleNamespace(
        Task=_Task, Resources=_Resources, Storage=_Storage, StorageMode=_StorageMode))
    patch.setattr("yeto.launch.modal_islands._list_modal_apps", lambda: [])
    patch.setenv("YETO_RL_COMPAT_GROUP", "nvidia-h100")
    from yeto.cli import parse_args
    from yeto.launcher import _prepare_rl_args
    from yeto.rl.adapters.miles.models.partial_experts.selection import ExpertSelection

    folder = tmp_path_factory.mktemp("sel")

    @lru_cache(maxsize=None)
    def selection(recipe: str, ep: int) -> str:
        """The cell's selection, built at the launch's EP as the user builds it: K3.1 slot_order,
        or an in_place selection for the generic recipe (the launcher reads no config.json for
        ``org/model``, so its shape is not checked against the fixture model here)."""

        path = folder / f"expert_selection.{recipe}.ep{ep}.json"
        try:
            if recipe == "kimi_k31":
                ExpertSelection.synthetic("uniform", model="kimi_k31", num_experts=896, num_trainable=128,
                                          ep_size=ep, layers=list(range(1, 93))).write(str(path))
            else:
                ExpertSelection.in_place("uniform", model="generic", num_experts=64, num_trainable=8,
                                         ep_size=ep, layers=list(range(4))).write(str(path))
        except ValueError:  # an EP the selection cannot split (the check refuses it on its own)
            return selection(recipe, 8)
        return str(path)

    fill = {"<64-hex>": HEX, "<store>": "modal-volume://v/run", "<verl plan hash>": HEX}

    @lru_cache(maxsize=None)
    def run(argv: tuple[str, ...], gpu: str) -> str | None:
        ep = int(argv[argv.index("--expert-parallel") + 1]) if "--expert-parallel" in argv else 8
        fill["<expert_selection.json>"] = selection(argv[argv.index("--rl-model-recipe") + 1], ep)
        args = parse_args([
            "--gpu", gpu, "--model", "org/model", "--data", "org/data", "--training-mode", "rl",
            "--total-steps", "3", "--rollout-batch-size", "8", "--n-samples-per-prompt", "2",
            "--rollout-max-response-len", "128", "--reward-function", "pkg.reward:score",
            "--trust-remote-code", "--inner-lr", "1e-5", *[fill.get(a, a) for a in argv]])
        args.model_revision, args.data_revision = "a" * 40, "b" * 40
        args.source_sha256, args.reward_sha256 = "c" * 64, "d" * 64
        args.controller = "local"
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                _prepare_rl_args(args)
        except (ValueError, SystemExit) as error:
            return str(error) or type(error).__name__
        return None

    yield run
    patch.undo()


def _cells():
    for model in MODELS:
        for form in FORMS:
            for preset in PRESETS:
                for backend in BACKENDS:
                    yield model, CheckRequest(form=form, preset=preset, backend=backend)


# shape and target variations on top of the recommended cells (the refusals they reach)
VARIATIONS = (
    {"lora_targets": "all-linear"},
    {"lora_targets": "all-linear", "expert_parallel": 2},
    {"expert_parallel": 3},
    {"tensor_parallel": 2},
    {"tensor_parallel": 2, "pipeline_parallel": 2},
)
VARIED = ("qwen3_8b", "qwen3_30b_a3b", "mixtral_8x7b", "kimi_k31", "deepseek_v3")


def _varied_cells():
    for model in VARIED:
        for variation in VARIATIONS:
            for form in FORMS:
                for preset in PRESETS:
                    for backend in BACKENDS:
                        yield model, CheckRequest(form=form, preset=preset, backend=backend, **variation)


def _launcher_disagreements(run, model: str, cells) -> list[str]:
    from yeto.rl.model_registry.check import gpu_spec

    out = []
    for cell_model, request in cells:
        if cell_model != model:
            continue
        report = check_model(_config(model), request, model=model)
        refusal = run(tuple(report.launch_argv), gpu_spec(report.request, report.recipe))
        if report.launcher_accepts != (refusal is None):
            out.append(f"{request}: launcher {'accepts' if refusal is None else 'refuses: ' + refusal}\n"
                       + render_text(report))
    return out


@pytest.mark.parametrize("model", sorted(MODELS))
def test_launcher_verdicts_match_yeto_launch(launcher, model):
    assert not (bad := _launcher_disagreements(launcher, model, _cells())), "\n\n".join(bad)


@pytest.mark.parametrize("model", VARIED)
def test_launcher_verdicts_match_yeto_launch_on_other_shapes(launcher, model):
    assert not (bad := _launcher_disagreements(launcher, model, _varied_cells())), "\n\n".join(bad)


# ----------------------------------------------------------------------------- island run config
def _provider(cfg: dict) -> argparse.Namespace:
    """The Megatron-Bridge provider fields ``resolve_rl_run_config`` reads, from config.json."""

    from yeto.rl.moe_geometry import from_hf_config, text_config

    t = text_config(cfg)
    heads, layers = int(t["num_attention_heads"]), int(t["num_hidden_layers"])
    fields = dict(
        hidden_size=t["hidden_size"], num_attention_heads=heads, num_layers=layers,
        ffn_hidden_size=t.get("intermediate_size") or t.get("moe_intermediate_size"),
        num_query_groups=t.get("num_key_value_heads") or heads,
        kv_channels=t.get("head_dim") or t["hidden_size"] // heads,
        seq_length=t.get("max_position_embeddings", 4096), vocab_size=t["vocab_size"],
        layernorm_epsilon=t.get("rms_norm_eps", 1e-6), position_embedding_type="rope",
        rotary_base=int(t.get("rope_theta", 10000)), rotary_percent=1.0)
    geometry = from_hf_config(cfg)
    if geometry is not None:
        fields["num_moe_experts"] = geometry.num_experts
        fields["moe_layer_freq"] = [1 if i in geometry.moe_layers else 0 for i in range(layers)]
        fields["moe_ffn_hidden_size"] = t.get("moe_intermediate_size") or t.get("intermediate_size")
        fields["moe_router_topk"] = geometry.router_topk
        fields["moe_grouped_gemm"] = True  # the Megatron-Bridge MoE bridges set it (qwen3_moe_bridge.py)
    return argparse.Namespace(**fields)


def _island_refusal(cfg: dict, report) -> str | None:
    sys.path.insert(0, str(Path(__file__).parent))
    from test_rl_argv_snapshot import _captured_args

    from yeto.rl.engine.run_config import resolve_rl_run_config

    (base,), kwargs = _captured_args()
    req = report.request
    full = req.form != "lora"
    namespace = argparse.Namespace(**{
        **vars(base), "seq_len": req.seq_len, "lora_r": req.lora_rank,
        "lora_targets": req.lora_targets,
        "actor_num_gpus_per_node": actor_gpus(req, report.recipe), "tensor_parallel": req.tensor_parallel,
        "pipeline_parallel": req.pipeline_parallel, "expert_parallel": req.expert_parallel,
        "parameter_mode": "full" if full else "lora", "rl_model_recipe": report.recipe,
        "rl_single_island_no_sync": req.preset == "single-island"})
    targets = [] if full else list(report.profile.lora_target_modules.get(
        "all-linear" if req.lora_targets == "all-linear" else "attention", ()))
    try:
        resolve_rl_run_config(namespace, **{**kwargs, "provider": _provider(cfg), "target_modules": targets},
                              yeto_policy_sync=req.preset != "single-island", verify_ref_load=False)
    except ValueError as error:
        return str(error)
    return None


def _island_cells():
    # generic-recipe Miles islands (the K3.1 / DSV4 recipes build their own providers);
    # full parameters on one island (two-island full-parameter runs add the placement rules
    # the launcher layer covers)
    for model in MODELS:
        if model in ("kimi_k31", "deepseek_v4_flash"):
            continue
        for extra in ({}, {"lora_targets": "all-linear"}, {"expert_parallel": 2, "lora_targets": "all-linear"},
                      {"expert_parallel": 2}, {"tensor_parallel": 3}, {"seq_len": 10 ** 7}):
            yield model, CheckRequest(form="lora", preset="strict-avg", **extra)
            yield model, CheckRequest(form="full", preset="single-island", **extra)


@pytest.mark.parametrize("model", sorted(set(m for m, _ in _island_cells())))
def test_run_config_verdicts_match_the_island(model):
    cfg = _config(model)
    for cell_model, request in _island_cells():
        if cell_model != model:
            continue
        report = check_model(cfg, request, model=model)
        expected = not any(i.status == FAIL and i.source == ISLAND and "run_config" in i.reason for i in report.items)
        refusal = _island_refusal(cfg, report)
        assert expected == (refusal is None), (
            f"{model} {request}: island {'accepts' if refusal is None else 'refuses: ' + refusal}\n"
            + render_text(report))


# ----------------------------------------------------------------------------- report shape
def test_three_onboarding_examples():
    qwen = check_model(_config("qwen3_0p6b"), CheckRequest(form="full", preset="full-parameter"))
    assert qwen.runnable and qwen.recommended["sync"]["fragments"] == 4  # the Qwen3-0.6B layout probe ran P=4
    assert "--rl-parameter-layout-sha256" in qwen.launch_argv and "auto" in qwen.launch_argv

    mixtral = check_model(_config("mixtral_8x7b"), CheckRequest(form="full", preset="full-parameter"))
    assert not mixtral.runnable and mixtral.launcher_accepts
    reasons = " ".join(i.reason for i in mixtral.items if i.status == FAIL)
    assert "no bridge for MixtralForCausalLM" in reasons and "converter for MixtralConfig" in reasons
    assert len(mixtral.templates) >= 2
    on_verl = check_model(_config("mixtral_8x7b"), CheckRequest(form="full", preset="full-parameter",
                                                                backend="verl"))
    assert on_verl.runnable

    moe = check_model(_config("qwen3_30b_a3b"), CheckRequest(form="partial-moe", preset="decoupled"))
    assert moe.runnable and moe.launch_argv[moe.launch_argv.index("--rl-sync-preset") + 1] == "decoupled"
    assert moe.recommended["memory"]["fits"] and "peak_gib_per_gpu" in moe.recommended["memory"]
    assert any("--layout in_place" in t for t in moe.templates)

    k31 = check_model(_config("kimi_k31"), CheckRequest(form="partial-moe", preset="full-parameter"))
    assert k31.recipe == "kimi_k31" and k31.launcher_accepts and k31.request.expert_parallel == 8
    k31_decoupled = check_model(_config("kimi_k31"), CheckRequest(form="lora", preset="decoupled"))
    assert not k31_decoupled.launcher_accepts and k31_decoupled.pending == ["decoupled/kimi_k31-lora"]


def test_report_dict_is_json_and_the_cli_exit_code_follows_the_verdict(capsys):
    from yeto.cli import main

    report = check_model(_config("qwen3_8b"), CheckRequest())
    json.dumps(report_dict(report), default=str)
    assert main(["model", "check", str(MODELS["qwen3_8b"]), "--no-fetch"]) == 0
    assert "verdict  RUNNABLE" in capsys.readouterr().out
    assert main(["model", "check", str(MODELS["mixtral_8x7b"]), "--no-fetch", "--form", "full",
                 "--preset", "full-parameter", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["runnable"] is False
    assert main(["model", "check", str(MODELS["qwen3_8b"]), "--no-fetch", "--recipe", "nope"]) == 2
    assert "not a --rl-model-recipe" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["model", "check", str(MODELS["qwen3_8b"]), "--tensor-parallel", "0"])


def test_the_index_verifies_expert_names():
    cfg = _config("qwen3_30b_a3b")
    good = [f"model.layers.{layer}.mlp.experts.{e}.{leaf}.weight" for layer in range(48) for e in range(128)
            for leaf in ("gate_proj", "up_proj", "down_proj")]
    ok = check_model(cfg, CheckRequest(), weight_names=good)
    assert next(i for i in ok.items if i.name == "expert-naming").status == "ok"
    short = [n for n in good if ".experts.127." not in n]
    bad = check_model(cfg, CheckRequest(), weight_names=short)
    item = next(i for i in bad.items if i.name == "expert-naming")
    assert item.status == FAIL and "expert count" in item.reason


def test_check_module_is_import_light():
    import subprocess

    code = ("import sys; import yeto.rl.model_registry.check; "
            "heavy = [m for m in ('torch', 'transformers', 'peft', 'megatron', 'yeto.launcher') if m in sys.modules]; "
            "print(heavy)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         cwd=Path(__file__).parents[1])
    assert out.stdout.strip() == "[]"


def test_the_matrix_reaches_both_verdicts(launcher):
    from yeto.rl.model_registry.check import gpu_spec

    seen = {True: 0, False: 0}
    refusals = set()
    for model, request in list(_cells()) + list(_varied_cells()):
        report = check_model(_config(model), request, model=model)
        refusal = launcher(tuple(report.launch_argv), gpu_spec(report.request, report.recipe))
        seen[refusal is None] += 1
        if refusal:
            refusals.add(refusal.split(":")[0][:60])
    assert seen[True] > 100 and seen[False] > 100 and len(refusals) >= 5


def test_every_pending_entry_still_reaches_a_failing_cell():
    """A change that turns a pending cell into a run removes its PENDING entry too;
    the printed report names the open work, never a pull request number."""

    import re

    reached = set()
    for model, request in list(_cells()) + list(_varied_cells()):
        report = check_model(_config(model), request, model=model)
        reached.update(report.pending)
        if report.pending:
            assert not re.search(r"#\d", render_text(report))
    assert reached == set(PENDING)


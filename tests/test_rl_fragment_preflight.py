"""``--fragments`` preflight of a Miles full-parameter run. No Ray, no GPU, no network.

P5 (Qwen3-4B full parameters, TP2): ``--fragments 4`` passed every dry run and
both trainer islands refused their 10-fragment layout after a six-minute boot.
The offline layout reproduces the measured island layout, and the launcher
refuses the mismatch before anything is ordered.
"""

from __future__ import annotations

import json

import pytest

from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from yeto.rl.adapters.miles.models import dense_gpt_layout as L

# Qwen/Qwen3-4B-Instruct-2507 config.json, the fields that shape parameters.
QWEN3_4B = {
    "architectures": ["Qwen3ForCausalLM"], "attention_bias": False, "head_dim": 128,
    "hidden_size": 2560, "intermediate_size": 9728, "model_type": "qwen3",
    "num_attention_heads": 32, "num_hidden_layers": 36, "num_key_value_heads": 8,
    "tie_word_embeddings": True, "vocab_size": 151936,
}
QWEN3_4B_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
QWEN3_4B_CONFIG_SHA256 = "5beea1a4a34c62782bfb2f911c606741a3bab8f92d80a118fa053c28af12e8ba"
# The island's own layout probe record (P5, island refusal at --fragments 4).
P5_LAYOUT_SHA256 = "d6bae71baf070eef2bbf156a046328fc1b4caf50fa1c691a7ede94308edc7aed"
P5_RANK_FRAGMENT_BYTES = (1816861184, 1816851456, 1816851456, 1816851968, 777912320)


def test_the_offline_qwen3_4b_tp2_layout_is_the_measured_island_layout():
    geometry = L.DenseGeometry.from_hf_config(QWEN3_4B)
    specs = L.rank_specs(geometry, tp=2)
    for rank in ("tp0-of-2", "tp1-of-2"):
        mine = [s for s in specs if s.shard_id.startswith(rank)]
        assert len(mine) == 290 and sum(s.numel for s in mine) == 2_011_332_096
    layout = L.offline_layout(QWEN3_4B, tp=2, minimum_fragments=4, matrix_merge="rda",
                              model_revision=QWEN3_4B_REVISION, config_sha256=QWEN3_4B_CONFIG_SHA256)
    assert layout.fragments == 10 and layout.exact
    assert layout.fragment_bytes == P5_RANK_FRAGMENT_BYTES * 2
    assert layout.layout_hash == P5_LAYOUT_SHA256


def test_the_fragment_count_follows_the_byte_bound_merge_and_requested_count():
    def count(**kw):
        return L.offline_layout(QWEN3_4B, tp=2, **{"minimum_fragments": 2, "matrix_merge": "rda", **kw}).fragments

    assert count() == 10
    assert count(minimum_fragments=10) == 10 and count(minimum_fragments=12) == 12
    assert count(matrix_merge="avg") == 8  # the vocabulary table shares a fragment
    assert count(max_fragment_bytes=1 << 30) > 10
    assert L.offline_layout(QWEN3_4B, tp=2, pp=2, minimum_fragments=2).exact is False


def test_models_outside_the_offline_layout_are_refused_by_it():
    with pytest.raises(L.DenseLayoutError, match="qwen3_moe"):
        L.DenseGeometry.from_hf_config({**QWEN3_4B, "model_type": "qwen3_moe"})
    with pytest.raises(L.DenseLayoutError, match="vocabulary"):
        L.rank_specs(L.DenseGeometry.from_hf_config({**QWEN3_4B, "vocab_size": 151935}), tp=2)


def test_untied_embeddings_add_the_output_layer():
    untied = L.DenseGeometry.from_hf_config({**QWEN3_4B, "tie_word_embeddings": False})
    names = {s.name for s in L.rank_specs(untied, tp=2)}
    assert "module.module.output_layer.weight" in names
    tied = {s.name for s in L.rank_specs(L.DenseGeometry.from_hf_config(QWEN3_4B), tp=2)}
    assert names - tied == {"module.module.output_layer.weight"}


def test_the_config_comes_from_a_local_directory_or_the_local_hf_cache(tmp_path, monkeypatch):
    from yeto.launch.fragment_preflight import local_hf_config

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(QWEN3_4B))
    assert json.loads(local_hf_config(str(model), None)) == QWEN3_4B
    cache = tmp_path / "hub"
    snapshot = cache / "models--Qwen--Qwen3-4B-Instruct-2507" / "snapshots" / QWEN3_4B_REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(json.dumps(QWEN3_4B))
    import huggingface_hub.constants

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(cache))
    assert json.loads(local_hf_config("Qwen/Qwen3-4B-Instruct-2507", QWEN3_4B_REVISION)) == QWEN3_4B
    assert local_hf_config("Qwen/Qwen3-4B-Instruct-2507", "f" * 40) is None  # not cached: no download
    assert local_hf_config("Qwen/not-cached", None) is None
    assert local_hf_config("s3://bucket/model", None) is None


# The P5 trainer argv shape: decoupled, full parameters, TP2 synced trainers.
P5_FLAGS = ("--rl-sync-preset", "decoupled", "--tuning", "full", "--tensor-parallel", "2",
            "--rl-placement", "fixed-partition", "--rl-rollout-gpus", "2",
            "--rl-parameter-layout-sha256", "auto", "--local-rl-rounds-per-sync", "4")
TWO_ISLANDS = "modal:4xh100,modal:4xh100"  # 2 trainer GPUs each: TP2, DP1


def _prepare(fragments, monkeypatch, *, config=QWEN3_4B, extra=(), gpu=TWO_ISLANDS):
    from yeto.launcher import _prepare_rl_args

    seen = []

    def config_bytes(model, revision):
        seen.append((model, revision))
        return None if config is None else json.dumps(config).encode()

    monkeypatch.setattr("yeto.launch.fragment_preflight.local_hf_config", config_bytes)
    args = _launcher_args("ports", (*P5_FLAGS, "--fragments", str(fragments), *extra), gpu=gpu)
    args.controller = "local"
    _prepare_rl_args(args)
    return args, seen


def test_the_launcher_refuses_a_fragment_count_the_islands_would_refuse(monkeypatch):
    with pytest.raises(ValueError, match=r"--fragments 4 does not match .* has 10 fragments over 2 "
                                         r"trainer rank\(s\).*Set --fragments 10$"):
        _prepare(4, monkeypatch)


def test_a_matching_fragment_count_launches_exactly_as_before(monkeypatch, capsys):
    from yeto.launcher import dry_run_plan

    def launch(config):
        args, seen = _prepare(10, monkeypatch, config=config)
        plan = json.dumps(dry_run_plan(args), sort_keys=True, default=str)
        return seen, plan, capsys.readouterr()

    seen, *checked = launch(QWEN3_4B)
    assert seen == [("org/model", "a" * 40)]
    _, *unchecked = launch(None)  # no config.json: not checked
    assert checked == unchecked  # the plan and every printed line


def test_a_model_outside_the_offline_layout_is_not_checked(monkeypatch):
    args, seen = _prepare(4, monkeypatch, config={**QWEN3_4B, "model_type": "llama"})
    assert seen and args.fragments == 4


def test_lora_runs_do_not_read_the_model_config(monkeypatch):
    seen = []
    monkeypatch.setattr("yeto.launch.fragment_preflight.local_hf_config",
                        lambda *a: seen.append(a))
    from yeto.launcher import _prepare_rl_args

    args = _launcher_args("ports", ("--rl-sync-preset", "decoupled", "--fragments", "4",
                                    "--local-rl-rounds-per-sync", "2"), gpu=TWO_ISLANDS)
    args.controller = "local"
    _prepare_rl_args(args)
    assert seen == []


def test_an_estimated_layout_warns_instead_of_refusing(monkeypatch, capsys):
    # PP2: Megatron's stage split is estimated, so a mismatch is a warning only.
    _prepare(4, monkeypatch, extra=("--pipeline-parallel", "2", "--rl-rollout-gpus", "4"),
             gpu="modal:8xh100,modal:8xh100")
    err = capsys.readouterr().err
    assert "[launcher] WARNING: --fragments 4 may not match: by estimate" in err
    assert "TP2 x PP2" in err

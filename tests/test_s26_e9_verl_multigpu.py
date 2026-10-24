"""S26 E9 G3: a verl sync island uses every card of its container; a multi-card
verl island on Modal runs NCCL with the shared-memory transport off."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from yeto.gpu_spec import parse_gpu_spec
from yeto.launch import modal_islands
from yeto.rl.adapters.verl import config as vconf
from yeto.rl.adapters.verl.island_entry import island_gpus


def _run(**kw) -> vconf.VerlRunConfig:
    return vconf.VerlRunConfig(model_path="/m", train_file="/t", val_file="/v", out_dir="/o", **kw)


def test_two_card_sync_island_trains_on_both_ranks():
    overrides = vconf.build_overrides(_run(n_gpus=2))
    assert "trainer.n_gpus_per_node=2" in overrides and "trainer.nnodes=1" in overrides


def test_zero_cards_is_refused():
    with pytest.raises(vconf.VerlConfigError, match="n_gpus"):
        vconf.validate(_run(n_gpus=0))


@pytest.mark.parametrize("info, want", [
    ({"before": {"CPU": 32.0, "GPU": 2.0}}, 2),
    ({"before": {"CPU": 8.0, "GPU": 2.0}, "after": {"CPU": 32.0, "GPU": 2.0}}, 2),
    ({"before": {"CPU": 32.0}}, 1),          # no GPU resource reported
    ({"skipped": "no RAY_ADDRESS"}, 1),      # no island-local Ray
])
def test_island_gpus_reads_the_island_ray(info, want):
    assert island_gpus(info) == want


@pytest.mark.parametrize("backend, gpu, want", [
    ("verl", "modal:2xa10g", True),
    ("verl", "modal:1xa10g", False),
    ("miles", "modal:2xa10g", False),
    (None, "modal:2xa10g", False),           # default backend is Miles
])
def test_nccl_shm_off_only_for_multi_card_verl(backend, gpu, want):
    (spec,) = parse_gpu_spec(gpu)
    assert modal_islands.verl_needs_nccl_shm_off(SimpleNamespace(rl_backend=backend), spec) is want


@pytest.mark.parametrize("extra, want", [
    ((), "1"),
    (("--modal-env", "NCCL_SHM_DISABLE=0"), "0"),  # an explicit --modal-env wins
])
def test_modal_island_config_carries_the_default(monkeypatch, tmp_path, extra, want):
    from yeto import cli

    monkeypatch.setenv("HOME", str(tmp_path))
    args = cli.parse_args(["--training-mode", "rl", "--rl-backend", "verl", "--gpu", "modal:2xa10g",
                           "--cluster-prefix", "e9g3", "--model", "Qwen/Qwen3-0.6B", "--data", "d", *extra])
    args.rl_image = "verl-build:" + "a" * 40  # what resolve_default_rl_image sets for verl
    (spec,) = parse_gpu_spec(args.gpu)
    task = SimpleNamespace(envs={}, secrets={}, file_mounts={}, run="true", setup="")
    cfg = modal_islands.build_modal_island_config(args, spec, 0, task, "none")
    assert cfg.envs["NCCL_SHM_DISABLE"] == want


def _write_readbacks(root: str, n: int) -> None:
    import os

    from yeto.rl.adapters.verl import vllm_readback as rb

    os.environ[rb.READBACK_ENV] = root
    for _ in range(n):
        rb.after_add_lora(SimpleNamespace(), 1)  # a broken manager: the error record is written


def test_readback_from_several_vllm_replicas_at_once(tmp_path):
    """e9b-g3-a: two vLLM replicas (one per card) wrote readback-v0.json through one
    shared tmp name; the second os.replace raised FileNotFoundError inside vLLM."""
    import json
    import multiprocessing

    (tmp_path / "expected_version").write_text("0")
    ctx = multiprocessing.get_context("fork")
    procs = [ctx.Process(target=_write_readbacks, args=(str(tmp_path), 200)) for _ in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
    assert [p.exitcode for p in procs] == [0, 0]
    assert json.loads((tmp_path / "readback-v0.json").read_text())["version"] == "0"
    assert not list(tmp_path.glob("*.tmp"))

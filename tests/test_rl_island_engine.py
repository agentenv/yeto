"""Per-island / per-pool engine image and backend (hetero-1114 A22): the pipe
from config to island task, Modal config, manifest and sampler pools. Without
the new keys nothing changes."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from yeto import launcher
from yeto.cli import parse_args as parse_cli
from yeto.gpu_spec import parse_gpu_spec
from yeto.launch import engine_images
from yeto.launch.nc import cli as nc_cli
from yeto.launch.nc import ssh_sampler, tasks
from yeto.launch.topology import rl_island_args
from yeto.launcher import _prepare_rl_args
from yeto.rl import sampler_pins
from yeto.rl.adapters.miles import pins as miles_pins
from yeto.rl.adapters.verl import pins as verl_pins
from yeto.rl.engine import island_engine

OTHER = "docker:ghcr.io/example/other@sha256:" + "a" * 64


def _cli(extra=()):
    return parse_cli([
        "--gpu", "aws:1xa100@us-east-1,aws:1xa100@us-west-2",
        "--model", "org/model", "--data", "org/data",
        "--training-mode", "rl", "--total-steps", "3",
        "--rollout-batch-size", "4", "--n-samples-per-prompt", "2",
        "--rollout-max-response-len", "128", "--local-rl-rounds-per-sync", "1",
        "--reward-function", "pkg.reward:score", "--trust-remote-code", *extra,
    ])


# --- default table ------------------------------------------------------------
def test_default_table_resolves_to_the_pins_constants():
    assert engine_images.default_image("cuda", "trainer", "miles") == miles_pins.MILES_NEXT_IMAGE
    assert engine_images.default_image("nvidia", "trainer", "verl") == verl_pins.VERL_IMAGE
    assert engine_images.default_image("rocm", "sampler", "vllm") == sampler_pins.VLLM_SAMPLER_IMAGE_ROCM
    assert engine_images.default_image("rocm", "sampler", "sglang") == sampler_pins.SGLANG_SAMPLER_IMAGE_ROCM
    assert engine_images.default_image("cuda", "sampler", "vllm") == sampler_pins.VLLM_SAMPLER_IMAGE_CUDA
    for row in engine_images.DEFAULT_IMAGES:  # every ref resolves; no copied digest
        assert row.device_family in engine_images.DEVICE_FAMILIES and row.role in engine_images.ROLES
        if row.ref is not None:
            assert engine_images.resolve_ref(row.ref)
        else:
            assert row.reason


def test_npu_rows_exist_and_ask_for_an_explicit_image():
    assert engine_images.default_image_entry("npu", "trainer", "verl").device_family == "ascend"
    assert engine_images.default_image_entry("ascend", "sampler", "vllm-ascend").ref is None
    with pytest.raises(engine_images.NoDefaultImage, match="site image"):
        engine_images.default_image("ascend", "trainer", "verl")
    with pytest.raises(ValueError, match="unknown"):
        engine_images.canonical_device_family("tpu")


# --- trainer islands ------------------------------------------------------------
def test_without_the_flag_every_island_sees_args_itself():
    args = _cli()
    _prepare_rl_args(args)
    assert rl_island_args(args, 0) is args and rl_island_args(args, 1) is args
    assert launcher._island_engine_args(args, 1) is None
    assert launcher._island_engine_rows(args, 1) == []
    island_engine.check(args, 2)  # no-op


def test_island_image_and_backend_reach_only_that_island():
    args = _cli(("--rl-island-engine", f"1:image={OTHER}"))
    _prepare_rl_args(args)
    assert rl_island_args(args, 0).rl_image == args.rl_image
    assert rl_island_args(args, 1).rl_image == OTHER
    assert args.rl_image != OTHER
    assert launcher._island_engine_rows(args, 1) == [
        {"island": 1, "backend": "miles", "image": OTHER}]


def test_backend_change_without_image_gets_the_backend_default():
    args = SimpleNamespace(gpu="aws:1xa100,aws:1xa100", training_mode="rl", rl_backend="miles",
                           rl_image=miles_pins.MILES_NEXT_IMAGE, rl_sync_layout="shape-free",
                           rl_island_engine=["1:backend=verl"])
    iargs = island_engine.island_args(args, 1, 2)
    assert (iargs.rl_backend, iargs.rl_image) == ("verl", verl_pins.VERL_IMAGE)
    island_engine.check(args, 2)
    args.rl_sync_layout = "shape-bound"
    with pytest.raises(ValueError, match="shape-free"):
        island_engine.check(args, 2)


def test_platform_picks_the_family_trainer_image_or_refuses():
    args = SimpleNamespace(gpu="aws:1xa100,aws:1xa100", training_mode="rl", rl_backend="verl",
                           rl_image=verl_pins.VERL_IMAGE, rl_island_engine=["0:platform=cuda"])
    assert island_engine.island_args(args, 0, 2).rl_image == verl_pins.VERL_IMAGE
    args.rl_island_engine = ["0:platform=npu"]
    with pytest.raises(engine_images.NoDefaultImage):
        island_engine.check(args, 2)
    args.rl_island_engine = ["0:platform=npu", f"0:image={OTHER}"]
    island_engine.check(args, 2)
    assert island_engine.records(args, 2) == [
        {"island": 0, "backend": "verl", "image": OTHER, "platform": "ascend"}]


@pytest.mark.parametrize("item, match", [
    ("2:image=x", "not in 0..1"), ("0:tag=x", "unknown key"), ("0image=x", "ISLAND:KEY=VALUE"),
    ("0:backend=torchtitan", "backend must be one of"),
])
def test_bad_items_are_refused(item, match):
    with pytest.raises(ValueError, match=match):
        island_engine.parse_overrides([item], 2)


def test_unpinned_island_image_is_refused():
    args = SimpleNamespace(training_mode="rl", rl_backend="miles", rl_image=OTHER,
                           rl_island_engine=["0:image=ghcr.io/x/y:latest"])
    with pytest.raises(ValueError, match="sha256"):
        island_engine.check(args, 2)


def test_modal_island_config_takes_the_island_image():
    task = SimpleNamespace(run="true", envs={}, setup="true")
    args = _cli(("--gpu", "modal:1xh100,modal:1xh100", "--cluster-prefix", "run",
                 "--rl-island-engine", f"1:image={OTHER}"))
    _prepare_rl_args(args)
    specs = parse_gpu_spec(args.gpu)
    base = launcher.build_modal_island_config(args, specs[0], 0, task, "1.2.3.4:5000",
                                              engine_args=launcher._island_engine_args(args, 0))
    own = launcher.build_modal_island_config(args, specs[1], 1, task, "1.2.3.4:5000",
                                             engine_args=launcher._island_engine_args(args, 1))
    assert base.image_ref == args.rl_image.removeprefix("docker:")
    assert own.image_ref == OTHER.removeprefix("docker:")


# --- sampler pools ------------------------------------------------------------------
POOL = {"target": 1, "accounts": [7], "shard_base": "nc-x", "run": "run-x", "spec": "s.json",
        "run_prefix": "run-x", "r2_env_file": "store.env"}


def test_pool_without_engine_keys_keeps_its_old_image():
    assert nc_cli.pool_sampler_image(POOL) is None
    assert nc_cli.pool_sampler_image({**POOL, "image": "x@sha256:1"}) is None
    nc_cli.pool_config(POOL)
    assert ssh_sampler.IMAGE == tasks.SERVING_IMAGE  # the two routes' old default


def test_pool_engine_platform_select_the_default_row():
    cfg = {**POOL, "engine": "sglang", "platform": "rocm"}
    nc_cli.pool_config(cfg)
    assert nc_cli.pool_sampler_image(cfg) == sampler_pins.SGLANG_SAMPLER_IMAGE_ROCM.removeprefix("docker:")
    assert nc_cli.pool_sampler_image({**POOL, "platform": "cuda"}) == \
        sampler_pins.VLLM_SAMPLER_IMAGE_CUDA.removeprefix("docker:")
    assert nc_cli.pool_sampler_image({**cfg, "image": "site/img@sha256:1"}) == "site/img@sha256:1"


def test_pool_engine_checks():
    with pytest.raises(ValueError, match="engine 'trt'"):
        nc_cli.pool_config({**POOL, "engine": "trt"})
    with pytest.raises(ValueError, match="set 'image'"):
        nc_cli.pool_config({**POOL, "engine": "vllm-ascend", "platform": "ascend"})
    nc_cli.pool_config({**POOL, "engine": "vllm-ascend", "platform": "ascend", "image": "site/va@sha256:1"})
    with pytest.raises(ValueError, match="engine.adapter"):
        nc_cli.check_spec_engine({"engine": "sglang"}, {"engine": {"adapter": "vllm-server"}})
    nc_cli.check_spec_engine({"engine": "vllm"}, {"engine": {"adapter": "vllm-server"}})
    nc_cli.check_spec_engine({}, {"engine": {"adapter": "sglang-server"}})


_NC_FREE = {k: v for k, v in POOL.items() if k != "accounts"}


def test_daytona_and_modal_providers_take_the_pool_default_image():
    from yeto.launch.nc import daytona_provider, modal_provider

    seen = {}

    class _Daytona:
        def __init__(self, run_id, **kw):
            seen.update(kw)

    orig = daytona_provider.DaytonaProvider
    daytona_provider.DaytonaProvider = _Daytona
    try:
        daytona_provider.from_config({**_NC_FREE, "slots": [0], "platform": "rocm",
                                      "provider": {"kind": "daytona"}}, "r")
        assert seen["image"] == sampler_pins.VLLM_SAMPLER_IMAGE_ROCM.removeprefix("docker:")
        seen.clear()
        daytona_provider.from_config({**_NC_FREE, "slots": [0],
                                      "provider": {"kind": "daytona", "snapshot": "s"}}, "r")
        assert "image" not in seen  # no engine keys: the section as written
    finally:
        daytona_provider.DaytonaProvider = orig
    m = modal_provider.from_config({**_NC_FREE, "slots": [0], "engine": "vllm",
                                    "platform": "cuda", "provider": {"kind": "modal"}}, "r",
                                   client=object())
    assert m.config["image"] == sampler_pins.VLLM_SAMPLER_IMAGE_CUDA.removeprefix("docker:")
    m = modal_provider.from_config({**_NC_FREE, "slots": [0],
                                    "provider": {"kind": "modal"}}, "r", client=object())
    assert "image" not in m.config


def test_sky_island_task_runs_the_island_image(monkeypatch):
    import sys
    import types

    from tests.test_rl_launcher import _Resources, _Storage, _StorageMode, _Task

    monkeypatch.setitem(sys.modules, "sky", types.SimpleNamespace(
        Task=_Task, Resources=_Resources, Storage=_Storage, StorageMode=_StorageMode))
    args = _cli(("--rl-island-engine", f"1:image={OTHER}", "--inner-lr", "1e-6"))
    _prepare_rl_args(args)
    args.model_revision, args.data_revision = "a" * 40, "b" * 40
    args.source_sha256, args.reward_sha256 = "c" * 64, "d" * 64
    specs = parse_gpu_spec(args.gpu)
    images = [launcher._island_task(launcher.make_miles_island_task, args, spec, m, 2,
                                    "127.0.0.1:29400", {}).resources.image_id
              for m, spec in enumerate(specs)]
    assert images == [args.rl_image, OTHER]


def test_island_override_under_release_code_pack_is_refused():
    args = SimpleNamespace(training_mode="rl", rl_backend="miles", rl_image=miles_pins.MILES_NEXT_IMAGE,
                           rl_code_pack="release", rl_island_engine=[f"1:image={OTHER}"])
    with pytest.raises(ValueError, match="code-pack"):
        island_engine.check(args, 2)
    args.rl_code_pack = "git-patch"
    island_engine.check(args, 2)
    # an override that keeps the run-wide image and backend passes under release
    args.rl_code_pack = "release"
    args.rl_island_engine = [f"1:image={miles_pins.MILES_NEXT_IMAGE}"]
    island_engine.check(args, 2)


def test_island_backend_change_under_engine_cluster_is_refused():
    args = SimpleNamespace(training_mode="rl", rl_backend="miles", rl_image=miles_pins.MILES_NEXT_IMAGE,
                           rl_sync_layout="shape-free", rl_engine_cluster="aws:1xa100",
                           rl_island_engine=["1:backend=verl"])
    with pytest.raises(ValueError, match="engine-cluster"):
        island_engine.check(args, 2)
    args.rl_engine_cluster = None
    island_engine.check(args, 2)


@pytest.mark.parametrize("cfg", [{"engine": "vllm-ascend", "image": "x"},
                                 {"platform": "ascend", "image": "x"}])
def test_pool_with_explicit_image_is_accepted(cfg):
    nc_cli.check_sampler_engine(cfg)
    assert nc_cli.pool_sampler_image(cfg) == "x"


def test_ascend_pool_defaults_to_vllm_ascend():
    assert nc_cli._sampler_engine({"platform": "ascend"}, "ascend") == "vllm-ascend"
    assert nc_cli._sampler_engine({"platform": "cuda"}, "cuda") == "vllm"
    with pytest.raises(ValueError, match="set 'image'"):
        nc_cli.check_sampler_engine({"platform": "ascend"})


# --- fix1115 F08 ----------------------------------------------------------------
def test_platform_cuda_takes_the_recipe_default_image():
    args = SimpleNamespace(training_mode="rl", rl_backend="miles", rl_model_recipe="kimi_k31",
                           tuning="full", rl_image=miles_pins.MILES_K31_FULL_IMAGE,
                           rl_island_engine=["1:platform=cuda"])
    assert island_engine.island_args(args, 1, 2).rl_image == miles_pins.MILES_K31_FULL_IMAGE
    args.rl_model_recipe, args.tuning = None, None
    assert island_engine.island_args(args, 1, 2).rl_image == miles_pins.default_rl_image()


def test_platform_without_a_pinned_row_asks_for_an_image():
    args = SimpleNamespace(training_mode="rl", rl_backend="miles", rl_image=miles_pins.MILES_NEXT_IMAGE,
                           rl_island_engine=["1:platform=rocm"])
    with pytest.raises(engine_images.NoDefaultImage, match="image="):
        island_engine.island_args(args, 1, 2)


def test_platform_only_keeping_the_run_image_passes_release():
    args = SimpleNamespace(training_mode="rl", rl_backend="miles", rl_model_recipe="kimi_k31",
                           tuning="full", rl_image=miles_pins.MILES_K31_FULL_IMAGE,
                           rl_code_pack="release", rl_island_engine=["1:platform=cuda"])
    island_engine.check(args, 2)
    # a platform default that differs from the run-wide image is still refused
    args.rl_image = OTHER
    with pytest.raises(ValueError, match="code-pack"):
        island_engine.check(args, 2)


def test_infer_backend_from_the_image():
    assert island_engine.infer_backend(miles_pins.MILES_K31_FULL_IMAGE) == "miles"
    assert island_engine.infer_backend(miles_pins.MILES_NEXT_IMAGE) == "miles"
    assert island_engine.infer_backend(verl_pins.VERL_IMAGE) == "verl"
    assert island_engine.infer_backend(verl_pins.VERL_MEGATRON_IMAGE) == "verl"
    assert island_engine.infer_backend(OTHER) is None


def test_image_of_another_backend_without_backend_key_is_refused(monkeypatch):
    args = SimpleNamespace(training_mode="rl", rl_backend="verl", rl_image=verl_pins.VERL_IMAGE,
                           rl_sync_layout="shape-free",
                           rl_island_engine=[f"1:image={miles_pins.MILES_NEXT_IMAGE}"])
    with pytest.raises(ValueError, match="backend=miles"):
        island_engine.check(args, 2)
    args.rl_island_engine = [f"1:image={miles_pins.MILES_NEXT_IMAGE}", "1:backend=miles"]
    island_engine.check(args, 2)
    args.rl_island_engine = [f"1:image={miles_pins.MILES_NEXT_IMAGE}", "1:backend=verl"]
    with pytest.raises(ValueError, match="is a 'miles' image"):
        island_engine.check(args, 2)


def test_unknown_image_without_backend_key_warns(capsys):
    args = SimpleNamespace(training_mode="rl", rl_backend="miles", rl_image=miles_pins.MILES_NEXT_IMAGE,
                           rl_island_engine=[f"1:image={OTHER}"])
    island_engine.check(args, 2)
    assert "WARNING" in capsys.readouterr().err
    args.rl_island_engine = [f"1:image={miles_pins.MILES_K31_FULL_IMAGE}"]
    island_engine.check(args, 2)
    assert "WARNING" not in capsys.readouterr().err

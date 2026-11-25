"""bug 26: a trainer island on a sky VM pulls the overlay-safe repack of a pinned image
whose layer docker cannot unpack, the plan shows the swap, and an overlay-unsafe image
without a repack is refused before anything is ordered (CPU, no cloud)."""

from __future__ import annotations

import io
from types import SimpleNamespace

import pytest

from yeto.gpu_spec import parse_gpu_spec
from yeto.launch import trainer_image_check as tic
from yeto.rl.adapters.miles import pins

REPACK = pins.MILES_K31_OVERLAYFIX_IMAGE
FULL_D = "docker:ghcr.io/agentenv/yeto-miles-k31@sha256:63ad0d2a9e70a139221050f51214f073a5bc53f66c0c57090ae926012e439c04"
FULL_D_SAFE = "docker:ghcr.io/agentenv/yeto-miles-k31@sha256:32626906f3fd14ab4d888c874c2ae27a727c4119b94b38aa8b60a6c5da17c084"
OLD_FULL = pins.MILES_K31_FULL_IMAGE  # fa70fe28: the FULL_E yeto layer, no repack


def test_sky_vm_pulls_the_repack_and_modal_keeps_the_pin():
    assert tic.sky_pull_image(pins.MILES_K31_IMAGE) == REPACK
    assert tic.sky_pull_image(pins.MILES_K31_IMAGE.removeprefix("docker:")) == REPACK.removeprefix("docker:")
    for image in (pins.MILES_K31_NEXT9_IMAGE, REPACK, FULL_D, "docker:ghcr.io/o/i:tag", None,
                  {"eu-north1": "computeimage-x", "docker": pins.MILES_K31_IMAGE}):
        assert tic.sky_pull_image(image) == image
    note, refused = tic.sky_pull_problem(pins.MILES_K31_IMAGE, "nebius")
    assert not refused and REPACK in note and "overlay-safe repack" in note
    for cloud in ("modal", "ssh", "MODAL"):
        assert tic.sky_pull_problem(pins.MILES_K31_IMAGE, cloud) is None
        assert tic.sky_pull_problem(FULL_D, cloud) is None
    assert tic.sky_pull_problem(pins.MILES_K31_NEXT9_IMAGE, "nebius") is None  # the customer K3.1 image
    message, refused = tic.sky_pull_problem(FULL_D, "aws")
    assert refused and "no overlay-safe repack" in message and f"use {FULL_D_SAFE} instead" in message
    message, refused = tic.sky_pull_problem(OLD_FULL, "nebius")
    assert refused and f"use {pins.MILES_K31_FULL_OVERLAYFIX_IMAGE} instead" in message
    # an unknown image is nobody's business here
    assert tic.vm_pull_refusal("docker:ghcr.io/o/i@sha256:" + "a" * 64, "docker") is None


def _args(image, gpu):
    return SimpleNamespace(training_mode="rl", rl_backend="miles", rl_image=image, gpu=gpu,
                           rl_training_attention_backend="auto", rl_model_recipe="kimi_k31",
                           model="moonshotai/Kimi-K3.1", model_revision=None)


@pytest.fixture
def no_pin_checks(monkeypatch):
    monkeypatch.setattr(tic, "image_pin_problems", lambda args, n=None: [])
    from yeto.rl.engine import sample_roles

    monkeypatch.setattr(sample_roles, "sampler_islands", lambda args, n: [])


def test_launch_preflight_notes_the_swap_and_refuses_an_unrepacked_image(no_pin_checks):
    gpu = "modal:1x1xH100,nebius:1x1xH200@us-central1"
    out = io.StringIO()
    tic.check_trainer_images(_args(pins.MILES_K31_IMAGE, gpu), parse_gpu_spec(gpu), out=out)
    lines = out.getvalue().splitlines()
    assert len(lines) == 1 and lines[0].startswith("[launcher] --gpu island 1 (nebius:") and REPACK in lines[0]
    with pytest.raises(ValueError, match=r"--gpu island 1 \(nebius:.*no overlay-safe repack"):
        tic.check_trainer_images(_args(FULL_D, gpu), parse_gpu_spec(gpu), out=io.StringIO())
    # Modal only: the unrepacked image is Modal's business (its builder unpacks it)
    out = io.StringIO()
    tic.check_trainer_images(_args(FULL_D, "modal:1x1xH100"), parse_gpu_spec("modal:1x1xH100"), out=out)
    assert out.getvalue() == ""
    out = io.StringIO()
    tic.check_trainer_images(_args(pins.MILES_K31_NEXT9_IMAGE, gpu), parse_gpu_spec(gpu), out=out)
    assert out.getvalue() == ""


def test_island_resources_pull_the_repack_on_a_sky_vm():
    from yeto.launch.rl_island_task import island_sky_resources

    def resources(image, gpu):
        spec = parse_gpu_spec(gpu)[0]
        args = SimpleNamespace(rl_image=image, learner_instance_type=None, spot=False, disk_size=None,
                               rl_island_network_tier="auto", rl_island_cpus=None, cpus=None)
        return island_sky_resources(args, spec, in_vm_docker=False)

    assert resources(pins.MILES_K31_IMAGE, "nebius:1x1xH200@us-central1")["image_id"] == REPACK
    assert resources(pins.MILES_K31_IMAGE, "aws:1x1xH100")["image_id"] == REPACK
    assert resources(pins.MILES_K31_NEXT9_IMAGE, "nebius:1x1xH200")["image_id"] == pins.MILES_K31_NEXT9_IMAGE
    assert resources(pins.MILES_K31_IMAGE, "modal:1x1xH100")["image_id"] == pins.MILES_K31_IMAGE


def test_plan_entry_names_the_pulled_repack():
    from yeto.launcher import _sky_overlay_swap

    swap = _sky_overlay_swap(SimpleNamespace(rl_image=pins.MILES_K31_IMAGE), "nebius")
    assert swap["pinned"] == pins.MILES_K31_IMAGE and swap["pulled"] == REPACK
    assert _sky_overlay_swap(SimpleNamespace(rl_image=pins.MILES_K31_IMAGE), "modal") is None
    assert _sky_overlay_swap(SimpleNamespace(rl_image=pins.MILES_K31_NEXT9_IMAGE), "nebius") is None
    assert _sky_overlay_swap(SimpleNamespace(rl_image=None), "aws") is None


# --- the launcher with fake sky / Modal backends (test_trainer_launch_render): no cloud ---
from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: E402,F401  (fixtures)


def _launch_args(gpu, image):
    args = _launcher_args("ports", ("--controller", "local", "--rl-single-island-no-sync", "--rl-image", image),
                          gpu=gpu)
    args.keep, args.recover_timeout, args.controller_poll = False, 0.0, 30.0
    return args


def test_the_dry_run_plan_names_the_repack_a_sky_vm_pulls(monkeypatch, capsys):
    import yeto.launcher as launcher

    monkeypatch.setenv("YETO_PREFLIGHT_MODEL", "off")
    monkeypatch.setenv("YETO_SKY_SERVER_CHECK", "off")
    args = _launch_args("aws:1xh100@us-west-2", pins.MILES_K31_IMAGE)
    launcher._prepare_rl_args(args)
    (entry,) = launcher.dry_run_plan(args)["island_requests"]
    assert entry["overlay_safe_image"]["pinned"] == pins.MILES_K31_IMAGE
    assert entry["overlay_safe_image"]["pulled"] == REPACK
    args = _launch_args("modal:1xh100", pins.MILES_K31_IMAGE)
    launcher._prepare_rl_args(args)
    (entry,) = launcher.dry_run_plan(args)["island_requests"]
    assert "overlay_safe_image" not in entry  # a Modal island's plan is unchanged
    args = _launch_args("aws:1xh100@us-west-2", pins.MILES_K31_NEXT9_IMAGE)
    launcher._prepare_rl_args(args)
    (entry,) = launcher.dry_run_plan(args)["island_requests"]
    assert "overlay_safe_image" not in entry  # an overlay-safe pin pulls as it is


def test_the_launched_sky_task_pulls_the_repack(monkeypatch, tmp_path):
    import sys

    import yeto.launcher as launcher
    from test_trainer_launch_render import _Provisioned, _setup

    monkeypatch.setenv("YETO_PREFLIGHT_MODEL", "off")
    monkeypatch.setenv("YETO_SKY_SERVER_CHECK", "off")
    _setup(monkeypatch, tmp_path, [])
    tasks = []

    def launch(task, **kw):
        tasks.append(task)
        return f"rid-{kw.get('cluster_name')}"

    sys.modules["sky"].launch = launch
    try:
        launcher.run(_launch_args("aws:1xh100@us-west-2", pins.MILES_K31_IMAGE))
    except _Provisioned:
        pass
    (task,) = tasks
    (res,) = list(task.resources) if isinstance(task.resources, (set, list, tuple)) else [task.resources]
    image = getattr(res, "image_id", None) or getattr(res, "kwargs", {}).get("image_id")
    assert REPACK.removeprefix("docker:") in str(image) and pins.MILES_K31_IMAGE.rpartition("@")[2] not in str(image)


def test_sampler_and_engine_islands_on_a_sky_vm_are_checked_with_their_own_image(no_pin_checks, monkeypatch):
    """REV follow-up: a sampler island (or an island with its own --rl-island-engine image)
    on a sky VM pins its own image; an overlay-unsafe one with no repack is refused before
    anything is ordered, naming the digest to use; a repackable one prints the swap."""
    from yeto.rl.engine import island_engine, sample_roles

    gpu = "modal:1x1xH100,nebius:1x1xH200@us-central1"
    images = {0: pins.MILES_K31_NEXT9_IMAGE, 1: FULL_D}

    def island_args(args, m, count):
        return SimpleNamespace(**{**vars(args), "rl_image": images[m]})

    monkeypatch.setattr(island_engine, "island_args", island_args)
    monkeypatch.setattr(sample_roles, "sampler_islands", lambda args, n: [1])  # island 1 samples
    with pytest.raises(ValueError, match=r"--gpu island 1 \(nebius:.*use .*32626906.* instead"):
        tic.check_trainer_images(_args(pins.MILES_K31_NEXT9_IMAGE, gpu), parse_gpu_spec(gpu), out=io.StringIO())
    images[1] = pins.MILES_K31_IMAGE
    out = io.StringIO()
    tic.check_trainer_images(_args(pins.MILES_K31_NEXT9_IMAGE, gpu), parse_gpu_spec(gpu), out=out)
    assert "--gpu island 1 (nebius:" in out.getvalue() and REPACK in out.getvalue()
    # the same unsafe image on the Modal island: Modal's builder unpacks it
    images[0], images[1] = FULL_D, pins.MILES_K31_NEXT9_IMAGE
    out = io.StringIO()
    tic.check_trainer_images(_args(pins.MILES_K31_NEXT9_IMAGE, gpu), parse_gpu_spec(gpu), out=out)
    assert out.getvalue() == ""


def _sky_pool(image_id):
    return {"shard_base": "p", "run": "r", "slots": [0],
            "provider": {"kind": "sky", "offers": [{"key": "o1", "price_usd_h": 1.0,
                                                    "resources": {"infra": "nebius/us-central1",
                                                                  "accelerators": "L40S:1",
                                                                  "image_id": image_id}}]}}


def test_a_sky_pool_offer_pinning_an_unrepacked_unsafe_image_is_refused_before_any_order(monkeypatch):
    import sys

    from yeto.launch.pool import sky_provider as sp

    monkeypatch.delitem(sys.modules, "sky", raising=False)  # a stub without __spec__ from another test

    provider = sp.from_config(_sky_pool(FULL_D), "r", sky=object())
    (problem,) = sp.offer_docker_problems(provider)
    assert "sky offer o1" in problem and f"use {FULL_D_SAFE} instead" in problem
    provider = sp.from_config(_sky_pool({"us-central1": "computeimage-x", "docker": OLD_FULL}), "r", sky=object())
    assert pins.MILES_K31_FULL_OVERLAYFIX_IMAGE in sp.offer_docker_problems(provider)[0]
    for ok in (pins.MILES_K31_NEXT9_IMAGE, "docker:vllm/vllm-openai@sha256:" + "b" * 64, "ubuntu-2204", None):
        assert sp.offer_docker_problems(sp.from_config(_sky_pool(ok), "r", sky=object())) == []
    assert any("32626906" in p for p in sp.start_problems(_sky_pool(FULL_D)))
    assert any("32626906" in p for p in sp.offline_problems(_sky_pool(FULL_D)))

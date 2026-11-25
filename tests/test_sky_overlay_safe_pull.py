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
FULL_D = "docker:ghcr.io/agentenv/yeto-miles-k31@sha256:" + pins.OVERLAY_UNSAFE_UNREPACKED[0]


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
    assert refused and "no overlay-safe repack" in message and "repack-whiteouts" in message


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

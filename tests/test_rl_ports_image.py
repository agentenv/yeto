"""MILES_NEXT_IMAGE (private ghcr ports image) pins, registry-credential
wiring for SkyPilot and Modal, and the image-aware ports source setup.
Legacy must be unchanged."""

import hashlib
import json
import re
import subprocess
from pathlib import Path

import pytest

from tests.test_modal_runner import _cfg, fake_modal
from tests.test_rl_engine_selection import _cli, _island_task
from yeto import launcher
from yeto import modal_runner as mr
from yeto import rl
from yeto.launcher import _prepare_rl_args, build_modal_island_config

REPO = Path(__file__).resolve().parents[1]
LOGIN = {
    "SKYPILOT_DOCKER_USERNAME": "user",
    "SKYPILOT_DOCKER_PASSWORD": "not-a-real-token",
    "SKYPILOT_DOCKER_SERVER": "ghcr.io",
}


@pytest.fixture
def no_login(monkeypatch):
    for key in mr.DOCKER_LOGIN_ENV_VARS:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def login(monkeypatch):
    for key, value in LOGIN.items():
        monkeypatch.setenv(key, value)


# ------------------------------------------------------------------ pins


def test_ports_image_is_the_private_digest_pinned_fork_image():
    assert rl.default_rl_image("ports") == rl.MILES_NEXT_IMAGE
    assert re.fullmatch(
        r"docker:ghcr\.io/michaellchung/yeto-miles-ports@sha256:[0-9a-f]{64}",
        rl.MILES_NEXT_IMAGE,
    )
    assert rl.MILES_NEXT_BASE_IMAGE.startswith("docker:docker.io/radixark/miles@sha256:")
    assert rl.MILES_NEXT_IMAGE_MANIFEST == "/opt/yeto/image-manifest.json"
    # The tag the build script pushes names the pinned commits.
    tag = f"{rl.MILES_NEXT_COMMIT[:7]}-{rl.SGLANG_NEXT_COMMIT[:7]}"
    assert f"Tag {tag};" in (REPO / "yeto/rl/__init__.py").read_text()


def test_legacy_image_default_is_unchanged():
    assert rl.default_rl_image("legacy") == rl.MILES_IMAGE == (
        "docker:ghcr.io/agentenv/miles@sha256:"
        "80c20538b63f76defde06ad5d4cfa564ae6f261110696eb1864470cb835e1590"
    )


def test_build_inputs_match_the_pins():
    dockerfile = (REPO / "docker/miles-ports/Dockerfile").read_text()
    base = rl.MILES_NEXT_BASE_IMAGE.split("@")[1]
    assert f"index {base}" in dockerfile.replace("\n# ", " ")
    for value in (rl.MILES_NEXT_COMMIT, rl.SGLANG_NEXT_COMMIT,
                  rl.MILES_NEXT_REPOSITORY, rl.SGLANG_NEXT_REPOSITORY):
        assert value in dockerfile
    script = (REPO / "scripts/build_miles_ports_image.sh").read_text()
    assert f"BASE_INDEX_DIGEST={base}" in script
    assert "yeto/rl/__init__.py" in script  # commits are read from the pins
    subprocess.run(["bash", "-n", str(REPO / "scripts/build_miles_ports_image.sh")], check=True)


def test_commits_and_digest_are_pinned_together_with_the_build_record():
    """Commit pins, tag, Dockerfile ARGs and the image digest move together:
    the latest build record must name exactly the pinned commits and digest."""
    record_dir = REPO / "openspec/changes/rl-infra-spec/evidence/ports-image/2026-10-01-lora-checksum"
    record = json.loads((record_dir / "build-record.json").read_text())
    assert record["miles_commit"] == rl.MILES_NEXT_COMMIT
    assert record["sglang_commit"] == rl.SGLANG_NEXT_COMMIT
    assert record["digest"] == rl.MILES_NEXT_IMAGE.split("@")[1]
    short = f"{rl.MILES_NEXT_COMMIT[:7]}-{rl.SGLANG_NEXT_COMMIT[:7]}"
    assert record["tag"].endswith(f":{short}")
    manifest = json.loads((record_dir / "image-manifest.json").read_text())
    assert manifest["miles"]["commit"] == rl.MILES_NEXT_COMMIT
    assert manifest["sglang"]["commit"] == rl.SGLANG_NEXT_COMMIT
    dockerfile = (REPO / "docker/miles-ports/Dockerfile").read_text()
    assert f"ARG SGLANG_VERSION={record['sglang_version']}" in dockerfile


# --------------------------------------------------------- source setup

LEGACY_SETUP_SHA256 = "1166134dbe978365d349f5e8f1851be7ccf7599c62dd28b01e88a64901425985"


def test_legacy_source_setup_is_byte_identical():
    miles, sglang = launcher._miles_source_setup("legacy")
    digest = hashlib.sha256((miles + "\0" + sglang).encode()).hexdigest()
    assert digest == LEGACY_SETUP_SHA256


def test_ports_setup_reuses_the_image_forks_and_still_verifies():
    miles, sglang = launcher._miles_source_setup("ports")
    subprocess.run(["bash", "-n"], input=miles + "\n" + sglang, text=True, check=True)
    c = rl.MILES_NEXT_COMMIT
    # Miles: the image's /root/miles (= ~/miles) is already at the pin, so
    # the fetch is conditional; identity checks stay unconditional.
    assert f'if [ "$(git -C ~/miles rev-parse HEAD 2>/dev/null)" != {c} ]; then\n' in miles
    guard = miles.index(f"!= {c} ]; then\n")
    fetch = miles.index(f"git -C ~/miles fetch --depth 1 origin {c}")
    end = miles.index("\nfi\n", guard)
    assert guard < fetch < end
    assert end < miles.index(f'test "$(git -C ~/miles rev-parse HEAD)" = {c}')
    assert "status --porcelain --untracked-files=all" in miles
    # SGLang: reuse /sgl-workspace/sglang only if it is exactly the pinned,
    # clean fork checkout that `import sglang` resolves to; else clone.
    s = rl.SGLANG_NEXT_COMMIT
    guard, rest = sglang.split("; then\n", 1)
    assert guard.startswith("if ")
    for needle in (
        f'[ "$(git -C /sgl-workspace/sglang rev-parse HEAD 2>/dev/null)" = {s} ]',
        f'= {rl.SGLANG_NEXT_REPOSITORY} ]',
        "git -C /sgl-workspace/sglang status --porcelain --untracked-files=all",
        "import os, sys, sglang",
        "[ ! -e ~/sglang ]",
    ):
        assert needle in guard
    reuse, fallback = rest.split("\nelse\n", 1)
    assert "ln -sfn /sgl-workspace/sglang ~/sglang" in reuse
    assert f"git -C ~/sglang fetch --depth 1 origin {s}" in fallback
    assert fallback.endswith("pip install -q --no-deps -e ~/sglang/python\nfi")


# ------------------------------------------------------ registry login


def test_registry_credentials_match_the_image_registry():
    ghcr = rl.MILES_NEXT_IMAGE
    assert mr.registry_host(ghcr) == "ghcr.io"
    assert mr.registry_host("docker:radixark/miles@sha256:" + "a" * 64) == "docker.io"
    assert mr.registry_credentials(ghcr, {}) is None
    assert mr.registry_credentials(ghcr, LOGIN) == LOGIN
    assert mr.registry_credentials(ghcr, {**LOGIN, "SKYPILOT_DOCKER_SERVER": "https://ghcr.io/"}) is not None
    # A login for another registry is not sent to this one.
    assert mr.registry_credentials(rl.MILES_NEXT_BASE_IMAGE, LOGIN) is None
    assert mr.registry_credentials(None, LOGIN) is None
    partial = {k: v for k, v in LOGIN.items() if k != "SKYPILOT_DOCKER_PASSWORD"}
    with pytest.raises(ValueError, match="SKYPILOT_DOCKER_PASSWORD"):
        mr.registry_credentials(ghcr, partial)
    # ... but a partial login only matters for its own registry.
    assert mr.registry_credentials(rl.MILES_NEXT_BASE_IMAGE, partial) is None
    assert mr.registry_credentials(ghcr, {"SKYPILOT_DOCKER_USERNAME": "u"}) is None


def _fake_login_config(monkeypatch):
    monkeypatch.setattr(launcher, "_sky_docker_login_config", lambda login: ("login", dict(login)))


def test_sky_ports_task_logs_in_via_task_secrets(monkeypatch, login):
    """sky 0.13 re-derives the DockerLoginConfig from the task secrets on every
    load; a DockerLoginConfig in Resources breaks its YAML round trip (B1 nsmoke)."""
    args = _cli()
    _prepare_rl_args(args)
    task = _island_task(args, monkeypatch)
    assert task.secrets == LOGIN
    assert not hasattr(task.resources, "_docker_login_config")
    assert not set(LOGIN) & set(task.envs)
    # SkyPilot exports secrets into setup/run: both scripts drop them first
    assert task.setup.startswith(launcher.DOCKER_LOGIN_UNSET)
    assert task.run.startswith(launcher.DOCKER_LOGIN_UNSET)
    assert "not-a-real-token" not in json.dumps(task.envs) + task.setup + task.run
    assert task.resources.image_id == rl.MILES_NEXT_IMAGE


@pytest.mark.parametrize("engine", ["ports", "legacy"])
def test_no_login_means_no_docker_login(monkeypatch, no_login, engine):
    args = _cli(("--rl-engine", engine))
    _prepare_rl_args(args)
    task = _island_task(args, monkeypatch)
    assert not hasattr(task, "secrets")
    assert not hasattr(task.resources, "_docker_login_config")


def _legacy_task_blob(monkeypatch):
    args = _cli(("--rl-engine", "legacy"))
    _prepare_rl_args(args)
    task = _island_task(args, monkeypatch)
    d = {k: v for k, v in vars(task).items() if k not in ("calls", "storage_mounts")}
    d["resources"] = vars(d["resources"])
    return json.dumps(d, sort_keys=True, default=str)


@pytest.mark.parametrize(
    "env",
    [LOGIN, {"SKYPILOT_DOCKER_SERVER": "ghcr.io"}, {**LOGIN, "SKYPILOT_DOCKER_SERVER": "docker.io"}],
)
def test_legacy_task_ignores_any_registry_login(monkeypatch, no_login, env):
    before = _legacy_task_blob(monkeypatch)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert _legacy_task_blob(monkeypatch) == before  # even for its ghcr.io image


def test_modal_pulls_the_private_image_with_the_login(monkeypatch, login):
    args = _cli()
    _prepare_rl_args(args)
    task = _island_task(args, monkeypatch)
    from yeto.gpu_spec import parse_gpu_spec

    args.cluster_prefix = "img-test"
    cfg = build_modal_island_config(args, parse_gpu_spec("modal:1xh100")[0], 0, task, "1.2.3.4:5")
    assert cfg.registry_login and cfg.image_ref == mr.image_ref_from_rl_image(rl.MILES_NEXT_IMAGE)
    assert not set(LOGIN) & set(cfg.envs)  # never inside the container
    assert "not-a-real-token" not in cfg.to_json()
    state = fake_modal(monkeypatch)
    mr.ModalOps("img-test").define(cfg)
    (img,) = state["images"]
    assert img.calls[0] == (
        "from_registry",
        (cfg.image_ref,),
        {"secret": ("secret", {"REGISTRY_USERNAME": "user", "REGISTRY_PASSWORD": "not-a-real-token"})},
    )


def test_modal_legacy_pull_ignores_the_login(monkeypatch, login):
    args = _cli(("--rl-engine", "legacy"))
    _prepare_rl_args(args)
    task = _island_task(args, monkeypatch)
    from yeto.gpu_spec import parse_gpu_spec

    args.cluster_prefix = "img-test"
    cfg = build_modal_island_config(args, parse_gpu_spec("modal:1xh100")[0], 0, task, "1.2.3.4:5")
    assert cfg.registry_login is False
    state = fake_modal(monkeypatch)
    mr.ModalOps("img-test").define(cfg)
    assert state["images"][0].calls[0] == ("from_registry", (cfg.image_ref,), {})


def test_modal_public_image_pull_is_unchanged(monkeypatch, no_login):
    state = fake_modal(monkeypatch)
    ref = "ghcr.io/x/miles@sha256:" + "c" * 64
    mr.ModalOps("yeto-run").define(_cfg(training_mode="rl", image_ref=ref, setup_script="true"))
    assert state["images"][0].calls[0] == ("from_registry", (ref,), {})


def test_sky_docker_login_config_is_skypilots_type():
    sky_utils = pytest.importorskip("sky.provision.docker_utils")
    config = launcher._sky_docker_login_config(LOGIN)
    assert isinstance(config, sky_utils.DockerLoginConfig) and config.server == "ghcr.io"


def test_pure_python_check_rejects_build_inputs(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("make_layer", REPO / "docker/miles-ports/make_layer.py")
    ml = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ml)
    repo = tmp_path / "r"
    run = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True, text=True).stdout.strip()
    repo.mkdir()
    run("init", "-q")
    run("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "base")
    base = run("rev-parse", "HEAD")
    pkgs = ("python/sglang/",)
    for path, ok in (
        ("python/sglang/srt/x.py", True),
        ("test/registered/t.py", True),
        ("README.md", True),
        ("python/pyproject.toml", False),
        ("python/setup.py", False),
        ("python/sglang/setup.cfg", False),
        ("python/requirements.txt", False),
        ("sgl-kernel/python/sgl_kernel/x.py", False),
        ("rust/src/lib.rs", False),
        ("python/sglang/srt/kernel.cu", False),
    ):
        run("checkout", "-q", "--detach", base)
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text("x\n")
        run("add", path)
        run("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", path)
        head = run("rev-parse", "HEAD")
        (repo / path).unlink()
        if ok:
            assert ml.check_pure_python(repo, base, head, pkgs) == [path]
        else:
            with pytest.raises(SystemExit, match="pure-Python"):
                ml.check_pure_python(repo, base, head, pkgs)


def test_gpu_exact_needs_modal_islands():
    from yeto.gpu_spec import parse_gpu_spec

    args = _cli(("--modal-gpu-exact",))
    with pytest.raises(ValueError, match="only to Modal"):
        launcher.require_modal_for_gpu_exact(args, parse_gpu_spec("modal:1xh100,aws:1xa100@us-east-1"))
    launcher.require_modal_for_gpu_exact(args, parse_gpu_spec("modal:1xh100"))
    launcher.require_modal_for_gpu_exact(_cli(), parse_gpu_spec("aws:1xa100@us-east-1"))


def test_image_manifest_schema_in_build_script():
    script = (REPO / "scripts/build_miles_ports_image.sh").read_text()
    for key in ('"miles"', '"sglang"', '"base"', '"commit"', '"index_digest"', '"manifest_digest"'):
        assert key in script
    assert json.loads('{"schema": 1}')  # manifest is plain JSON


# ------------------------------------------------------ exact Modal GPU


def test_default_gpu_request_is_unchanged():
    assert _cfg().gpu_request == "H100:8"  # Modal may still upgrade to H200
    assert _cfg(gpu="L40S", gpus_per_node=1).gpu_request == "L40S:1"


def test_exact_gpu_pins_h100_and_leaves_others_alone():
    assert _cfg(gpu_exact=True).gpu_request == "H100!:8"
    assert _cfg(gpu="L40S", gpus_per_node=1, gpu_exact=True).gpu_request == "L40S:1"
    cfg = _cfg(gpu_exact=True)
    assert mr.ModalIslandConfig.from_json(cfg.to_json()).gpu_request == "H100!:8"
    # Configs serialised before the field existed load as non-exact.
    old = json.loads(_cfg().to_json())
    old.pop("gpu_exact")
    assert mr.ModalIslandConfig.from_json(json.dumps(old)).gpu_exact is False


def test_check_gpu_names():
    h100 = "NVIDIA H100 80GB HBM3"
    mr.check_gpu_names("H100", [h100] * 2, 2)
    mr.check_gpu_names("L4", ["NVIDIA L4"], 1)
    mr.check_gpu_names("A100-80GB", ["NVIDIA A100-SXM4-80GB"], 1)
    for gpu, names, count in (
        ("H100", ["NVIDIA H200"], 1),
        ("H100", [h100, "NVIDIA H200"], 2),
        ("H100", [h100], 2),
        ("L4", ["NVIDIA L40S"], 1),
        ("A100-80GB", ["NVIDIA A100-SXM4-40GB"], 1),
    ):
        with pytest.raises(RuntimeError, match="exact"):
            mr.check_gpu_names(gpu, names, count)
    assert set(mr.MODAL_GPU_NAME_PATTERNS) == set(mr.MODAL_GPUS)


@pytest.mark.parametrize("exact", [False, True])
def test_island_main_asserts_the_gpu_only_when_exact(monkeypatch, exact):
    ran = []
    monkeypatch.setattr(mr, "visible_gpu_names", lambda: ["NVIDIA H200"])
    monkeypatch.setattr(mr.subprocess, "call", lambda *a, **k: ran.append(a) or 0)
    cfg = _cfg(gpus_per_node=1, gpu_exact=exact)
    if exact:
        with pytest.raises(RuntimeError, match="H200"):
            mr.island_main(cfg.to_json())
        assert ran == []  # nothing runs on the wrong GPU
    else:
        assert mr.island_main(cfg.to_json()) == 0 and ran


def test_launcher_flag_reaches_the_modal_config(monkeypatch, no_login):
    from yeto.gpu_spec import parse_gpu_spec

    spec = parse_gpu_spec("modal:1xh100")[0]
    for flag, expected in ((("--modal-gpu-exact",), "H100!:1"), ((), "H100:1")):
        args = _cli(flag)
        _prepare_rl_args(args)
        task = _island_task(args, monkeypatch)
        args.cluster_prefix = "img-test"
        cfg = build_modal_island_config(args, spec, 0, task, "1.2.3.4:5")
        assert cfg.gpu_request == expected


def _sky_roundtrip(task_kwargs, resources_kwargs):
    sky = pytest.importorskip("sky")
    from sky.utils import dag_utils

    t = sky.Task(name="x", run="echo", **task_kwargs)
    t.set_resources(sky.Resources(infra="nebius", accelerators="H100:8",
                                  image_id=rl.MILES_NEXT_IMAGE, **resources_kwargs))
    with sky.Dag() as dag:
        dag.add(t)
    text = dag_utils.dump_chain_dag_to_yaml_str(dag)  # client
    for _ in range(2):  # server load + dump, and once more
        dag = dag_utils.load_chain_dag_from_yaml_str(text)
        text = dag_utils.dump_chain_dag_to_yaml_str(dag)
    return next(iter(dag.tasks[0].resources))


def test_sky_resources_login_breaks_the_yaml_round_trip():
    """The B1 nsmoke failure, reproduced on the installed sky (skipped without sky)."""
    docker = pytest.importorskip("sky.provision.docker_utils")
    config = docker.DockerLoginConfig(username="user", password="not-a-real-token", server="ghcr.io")
    with pytest.raises(TypeError, match="asdict"):
        _sky_roundtrip({}, {"_docker_login_config": config})


def test_sky_secrets_login_survives_the_yaml_round_trip():
    docker = pytest.importorskip("sky.provision.docker_utils")
    r = _sky_roundtrip({"secrets": dict(LOGIN)}, {})
    assert isinstance(r._docker_login_config, docker.DockerLoginConfig)
    assert r._docker_login_config.server == "ghcr.io"


def test_on_demand_islands_for_acceptance_runs(monkeypatch, no_login):
    """Nebius acceptance runs: --on-demand gives use_spot=False and no spot
    checkpoint storage; the default (spot) is unchanged."""
    args = _cli(("--gpu", "nebius:1x8xH100@eu-north1", "--on-demand"))
    _prepare_rl_args(args)
    task = _island_task(args, monkeypatch)
    assert task.resources.use_spot is False
    assert "storage" not in task.calls
    default = _cli(("--gpu", "nebius:1x8xH100@eu-north1"))
    _prepare_rl_args(default)
    spot = _island_task(default, monkeypatch)
    assert spot.resources.use_spot is True and "storage" in spot.calls

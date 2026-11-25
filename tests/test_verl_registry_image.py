"""The verl FSDP2 registry image and the verl trainer cards (fleet-verl-trainer-islands 5.1, 5.2).

The registry image is the Modal image's recipe (``image.recipe_steps``) as a Dockerfile:
the checked-in ``docker/verl-fsdp/Dockerfile``, the pins' recipe hash and the Modal call
chain are kept equal here, so the two images cannot drift apart. A sky island pulls the
registry image in place of ``verl-build:<commit>``; an ssh: island still refuses
``verl-build`` and names the registry image. A verl FSDP2 trainer is allowed on
sm80 / sm89 / sm90 cards; a card without a recorded run gets an "unverified" warning."""

from __future__ import annotations

import io
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401  (fixtures)

from yeto.rl.adapters.verl import image, launch_flags, pins

ROOT = Path(__file__).resolve().parents[1]
DIGEST_RE = re.compile(r"docker:[^\s@]+/yeto-verl-fsdp@sha256:[0-9a-f]{64}")
REGISTRY = "registry.example/mirror"
IMG = f"docker:{REGISTRY}/yeto-verl-fsdp@sha256:{pins.VERL_REGISTRY_DIGEST}"


@pytest.fixture(autouse=True)
def _verl_registry(monkeypatch):
    """The operator's verl registry (there is no default); tests of the unset case delete it."""
    monkeypatch.setenv(pins.VERL_REGISTRY_ENV, REGISTRY)


class _Recorder:
    """A fake ``modal`` whose Image records every build call in order."""

    def __init__(self):
        self.calls = []
        outer = self

        class Image:
            @staticmethod
            def from_registry(ref, add_python=None):
                outer.calls.append(("from_registry", ref, add_python))
                return Image()

            def apt_install(self, *pkgs):
                outer.calls.append(("apt_install", pkgs))
                return self

            def pip_install(self, *reqs):
                outer.calls.append(("pip_install", reqs))
                return self

            def run_commands(self, *cmds):
                outer.calls.append(("run_commands", cmds))
                return self

            def add_local_file(self, src, dst, copy=False):
                outer.calls.append(("add_local_file", Path(src).name, dst, copy))
                return self

            def env(self, mapping):
                outer.calls.append(("env", dict(mapping)))
                return self

        self.Image = Image


def _modal_calls(**kw):
    rec = _Recorder()
    image.modal_image(rec, **kw)
    return rec.calls


# --- 5.1: one recipe, two images ----------------------------------------------------------------

def test_the_checked_in_dockerfile_is_the_recipe():
    assert (ROOT / "docker/verl-fsdp/Dockerfile").read_text() == image.dockerfile(), (
        "regenerate: python -m yeto.rl.adapters.verl.image --dockerfile > docker/verl-fsdp/Dockerfile")


def test_the_pinned_registry_image_was_built_from_this_recipe():
    # a recipe (or patch_verl.py) change without a rebuild fails here: rebuild with
    # docker/verl-fsdp/build.sh --push and pin the new tag, digest and recipe hash
    assert pins.VERL_REGISTRY_RECIPE_SHA256 == image.recipe_sha256()
    assert pins.VERL_REGISTRY_TAG == f"fsdp-{pins.VERL_COMMIT[:8]}-{image.recipe_sha256()[:8]}"
    assert f"ai.yeto.verl.commit={pins.VERL_COMMIT}" in image.dockerfile()
    assert f"ai.yeto.verl.recipe_sha256={image.recipe_sha256()}" in image.dockerfile()
    assert DIGEST_RE.fullmatch(IMG)
    assert IMG.endswith("@sha256:" + pins.VERL_REGISTRY_DIGEST)


def test_the_recipe_hash_follows_the_patch_file(monkeypatch, tmp_path):
    before = image.recipe_sha256()
    patched = tmp_path / "patch_verl.py"
    patched.write_text((ROOT / "yeto/rl/adapters/verl/patch_verl.py").read_text() + "\n# changed\n")
    real = Path.with_name
    monkeypatch.setattr(Path, "with_name",
                        lambda self, name: patched if name == "patch_verl.py" else real(self, name))
    assert image.recipe_sha256() != before


def test_the_modal_image_runs_the_same_steps_as_the_dockerfile():
    calls = _modal_calls()
    text = image.dockerfile()
    assert calls[0] == ("from_registry", pins.VERL_BASE_IMAGE, pins.VERL_PYTHON)
    assert f"FROM {pins.VERL_BASE_IMAGE}" in text
    pos = 0
    for call in calls[1:]:
        kind = call[0]
        if kind == "run_commands":
            needle = "RUN " + " && ".join(f"({c})" for c in call[1])
        elif kind == "apt_install":
            needle = "apt-get install -y " + " ".join(call[1])
        elif kind == "pip_install":
            needle = "pip install --break-system-packages " + " ".join(call[1])
        elif kind == "add_local_file":
            needle = f"COPY {call[1]} {call[2]}"
        else:
            needle = "ENV " + " ".join(f"{k}={v}" for k, v in call[1].items())
        found = text.find(needle, pos)
        assert found >= 0, f"{kind} step missing or out of order in the Dockerfile: {needle}"
        pos = found + len(needle)


def test_the_modal_call_chain_is_unchanged():
    # the Modal layer cache keys on this exact chain: the refactor onto recipe_steps keeps it
    patch = f"{image.VENV_BIN}/python /workspace/patch_verl.py {pins.VERL_ROOT}"
    base = [("from_registry", pins.VERL_BASE_IMAGE, pins.VERL_PYTHON),
            ("apt_install", ("git", "build-essential", "curl", "libnuma1")),
            ("pip_install", (f"uv=={pins.VERL_UV_VERSION}",)),
            ("run_commands", tuple(image.build_commands())),
            ("run_commands", tuple(image.store_client_commands()))]
    assert _modal_calls() == [*base, ("add_local_file", "patch_verl.py", "/workspace/patch_verl.py", True),
                              ("run_commands", (patch,)), ("env", image.image_env())]
    assert _modal_calls(patched=False) == [*base, ("env", image.image_env())]
    assert _modal_calls(variant="megatron") == [
        *base, ("run_commands", tuple(image.megatron_commands())), ("apt_install", image.RDMA_PACKAGES),
        ("add_local_file", "patch_verl.py", "/workspace/patch_verl.py", True),
        ("run_commands", (patch + " --profile megatron",)), ("env", image.image_env())]


def test_the_registry_image_is_on_the_registry_the_operator_names(monkeypatch):
    assert pins.verl_registry_image() == IMG
    monkeypatch.setenv(pins.VERL_REGISTRY_ENV, "111122223333.dkr.ecr.us-east-1.amazonaws.com/")
    assert pins.verl_registry_image() == (
        f"docker:111122223333.dkr.ecr.us-east-1.amazonaws.com/yeto-verl-fsdp@sha256:{pins.VERL_REGISTRY_DIGEST}")
    monkeypatch.delenv(pins.VERL_REGISTRY_ENV)
    monkeypatch.setenv("YETO_IMAGE_REGISTRY", "registry.example/miles-mirror")  # the Miles images' registry is not used
    assert pins.verl_registry_image() is None and pins.registry_image_for(pins.VERL_IMAGE) is None
    assert pins.is_verl_registry_image(IMG) and pins.is_verl_registry_image(IMG.removeprefix("docker:"))
    assert not pins.is_verl_registry_image(IMG.replace(pins.VERL_REGISTRY_DIGEST, "0" * 64))
    assert not pins.is_verl_registry_image("docker:x/other@sha256:" + pins.VERL_REGISTRY_DIGEST)


def test_without_a_registry_a_pulling_verl_island_is_refused_before_any_order(monkeypatch):
    from yeto.launch import ssh_trainer_island as sti
    from yeto.launch.rl_backend import pulled_rl_image
    from yeto.launch.trainer_image_check import check_trainer_images

    monkeypatch.delenv(pins.VERL_REGISTRY_ENV)
    pulled = pulled_rl_image(_rl(), "aws")
    assert pulled["image"] is None and pins.VERL_REGISTRY_ENV in pulled["problem"]
    assert "dkr.ecr" in pulled["problem"] and "ghcr.io/<owner>" in pulled["problem"]
    for check in ("on", "off"):  # a configuration problem, not a registry question: refused either way
        monkeypatch.setenv("YETO_IMAGE_MANIFEST_CHECK", check)
        with pytest.raises(ValueError, match=r"--gpu island 0 .*YETO_VERL_IMAGE_REGISTRY"):
            check_trainer_images(_verl_args(), _sky_specs("aws:1xh100@us-west-2"), out=io.StringIO())
    assert check_trainer_images(_verl_args(), _sky_specs("modal:1xh100"), out=io.StringIO()) == []  # Modal builds
    with pytest.raises(sti.SshIslandError, match="YETO_VERL_IMAGE_REGISTRY"):
        sti.run_image(_rl(), [SimpleNamespace(runtime="docker", image=None, name="n0")])


def test_the_dry_run_of_a_sky_verl_island_without_a_registry_fails(_sky_launch, monkeypatch):
    import yeto.launcher as launcher

    monkeypatch.delenv(pins.VERL_REGISTRY_ENV)
    args = _sky_launch("aws:1xa100@us-west-2")
    launcher._prepare_rl_args(args)
    with pytest.raises(ValueError, match="YETO_VERL_IMAGE_REGISTRY"):
        launcher.dry_run_plan(args)
    args = _sky_launch("modal:1xa100")
    launcher._prepare_rl_args(args)
    assert launcher.dry_run_plan(args)["island_requests"]  # a Modal island needs no registry


def test_the_registry_image_belongs_to_the_verl_backend():
    from yeto.rl.engine import island_engine

    assert island_engine.infer_backend(IMG) == "verl"
    assert island_engine.image_ok(SimpleNamespace(rl_image=IMG, rl_backend="verl"))


def _rl(backend="verl", image_ref=None):
    return SimpleNamespace(training_mode="rl", rl_backend=backend,
                           rl_image=image_ref if image_ref is not None else pins.VERL_IMAGE)


def test_a_sky_island_pulls_the_registry_image_in_place_of_verl_build():
    from yeto.launch.rl_backend import pulled_rl_image

    for cloud in ("aws", "nebius", "gcp"):
        assert pulled_rl_image(_rl(), cloud) == {"image": IMG, "tag": pins.VERL_REGISTRY_TAG,
                                                 "replaces": pins.VERL_IMAGE}
    assert pulled_rl_image(_rl(), "modal") is None  # Modal builds verl-build itself
    assert pulled_rl_image(_rl(), "ssh") is None  # ssh_trainer_island.run_image decides (refuses)
    assert pulled_rl_image(_rl(image_ref=pins.VERL_MEGATRON_IMAGE), "aws") is None  # no registry variant
    assert pulled_rl_image(_rl(image_ref=IMG), "aws") is None  # used as given
    from yeto.rl import default_rl_image

    assert pulled_rl_image(_rl("miles", default_rl_image()), "aws") is None  # Miles: unchanged


def test_an_ssh_island_still_refuses_verl_build_and_names_the_registry_image():
    from yeto.launch import ssh_trainer_island as sti

    nodes = [SimpleNamespace(runtime="docker", image=None, name="n0")]
    with pytest.raises(sti.SshIslandError, match="builds on Modal") as err:
        sti.run_image(_rl(), nodes)
    assert IMG in str(err.value) and pins.VERL_REGISTRY_TAG in str(err.value)
    assert sti.run_image(_rl(image_ref=IMG), nodes) == IMG.removeprefix(
        "docker:")


def test_the_launch_flags_offer_the_registry_image_for_the_pinned_build_only():
    assert launch_flags.pulled_image(pins.VERL_IMAGE)["image"] == IMG
    assert launch_flags.pulled_image("verl-build:" + "b" * 40) is None
    assert launch_flags.pulled_image(IMG) is None


# --- 5.1: the plan of a sky island --------------------------------------------------------------

@pytest.fixture
def _sky_launch(monkeypatch):
    monkeypatch.setenv("YETO_PREFLIGHT_MODEL", "off")
    monkeypatch.setenv("YETO_SKY_SERVER_CHECK", "off")
    return lambda gpu: _launcher_args("ports", ("--rl-backend", "verl", "--reward-function",
                                                "gsm8k_reward:score", "--controller", "local",
                                                "--rl-single-island-no-sync"), gpu=gpu)


def test_the_sky_island_plan_names_the_registry_image_tag(_sky_launch, capsys):
    import yeto.launcher as launcher

    args = _sky_launch("aws:1xa100@us-west-2")
    launcher._prepare_rl_args(args)
    assert args.rl_image == pins.VERL_IMAGE
    (entry,) = launcher.dry_run_plan(args)["island_requests"]
    assert entry["trainer_image"] == {"image": IMG, "tag": pins.VERL_REGISTRY_TAG,
                                      "replaces": pins.VERL_IMAGE}
    assert f"pulls {IMG} (tag {pins.VERL_REGISTRY_TAG})" in capsys.readouterr().err

    args = _sky_launch("modal:1xa100")
    launcher._prepare_rl_args(args)
    (entry,) = launcher.dry_run_plan(args)["island_requests"]
    assert "trainer_image" not in entry  # a Modal island's plan is unchanged


def test_the_sky_island_task_runs_in_the_registry_image(_sky_launch):
    import yeto.launcher as launcher
    from yeto.launch.rl_island_task import make_miles_island_task
    from yeto.gpu_spec import parse_gpu_spec

    args = _sky_launch("aws:1xa100@us-west-2")
    launcher._prepare_rl_args(args)
    (spec,) = parse_gpu_spec(args.gpu)
    task = launcher._island_task(make_miles_island_task, args, spec, 0, 1, "$SYNCER_ADDR", {})
    assert task.resources.image_id == IMG
    assert args.rl_image == pins.VERL_IMAGE  # the run's own args are untouched


def test_the_island_task_itself_refuses_a_pulling_verl_island_without_a_registry(_sky_launch, monkeypatch):
    # the launcher's own guard (launcher._pulled_image_args), behind the preflight: a path that builds an
    # island task without check_trainer_images (fleet island add / move, a relaunch) still cannot order
    # a sky verl island whose image has no registry
    import yeto.launcher as launcher
    from yeto.gpu_spec import parse_gpu_spec
    from yeto.launch.rl_island_task import make_miles_island_task

    args = _sky_launch("aws:1xa100@us-west-2")
    launcher._prepare_rl_args(args)
    (spec,) = parse_gpu_spec(args.gpu)
    monkeypatch.delenv(pins.VERL_REGISTRY_ENV)
    built = []
    factory = lambda *a, **kw: built.append(a) or make_miles_island_task(*a, **kw)  # noqa: E731
    with pytest.raises(ValueError, match=r"^aws trainer island: .*YETO_VERL_IMAGE_REGISTRY"):
        launcher._island_task(factory, args, spec, 0, 1, "$SYNCER_ADDR", {})
    assert built == []  # refused before the task factory runs
    with pytest.raises(ValueError, match="aws trainer island: .*dkr.ecr.*ghcr.io/<owner>"):
        launcher._pulled_image_args(_rl(), SimpleNamespace(cloud="aws"))
    for cloud in ("modal", "ssh", None):  # no pull: the args come back as they are
        iargs = _rl()
        assert launcher._pulled_image_args(iargs, SimpleNamespace(cloud=cloud)) is iargs
    miles = _rl("miles", "docker:x/miles@sha256:" + "0" * 64)
    assert launcher._pulled_image_args(miles, SimpleNamespace(cloud="aws")) is miles
    monkeypatch.setenv(pins.VERL_REGISTRY_ENV, REGISTRY)
    iargs = _rl()
    swapped = launcher._pulled_image_args(iargs, SimpleNamespace(cloud="aws"))
    assert swapped.rl_image == IMG and iargs.rl_image == pins.VERL_IMAGE  # a copy; the caller's args untouched


def test_a_private_registry_image_logs_in_through_the_sky_task_secrets_only(_sky_launch, monkeypatch):
    # the Miles route for a private --rl-image (credentials.registry_login_for): SKYPILOT_DOCKER_* in the
    # launching environment become sky task SECRETS for the island whose image is on that registry; the
    # scripts unset them first; a Modal island building verl-build gets no login at all
    import yeto.launcher as launcher
    from yeto.gpu_spec import parse_gpu_spec
    from yeto.launch.rl_island_task import make_miles_island_task
    password = "registry-password-SENTINEL-0123456789"
    monkeypatch.setenv("SKYPILOT_DOCKER_USERNAME", "puller")
    monkeypatch.setenv("SKYPILOT_DOCKER_PASSWORD", password)
    monkeypatch.setenv("SKYPILOT_DOCKER_SERVER", REGISTRY.split("/", 1)[0])
    args = _sky_launch("aws:1xa100@us-west-2")
    launcher._prepare_rl_args(args)
    (spec,) = parse_gpu_spec(args.gpu)
    task = launcher._island_task(make_miles_island_task, args, spec, 0, 1, "$SYNCER_ADDR", {})
    assert task.resources.image_id == IMG
    assert task.secrets["SKYPILOT_DOCKER_PASSWORD"] == password
    assert task.secrets["SKYPILOT_DOCKER_USERNAME"] == "puller"
    texts = "\n".join(str(getattr(task, a, "") or "") for a in ("setup", "run", "envs", "file_mounts"))
    assert password not in texts
    assert task.setup.startswith("unset SKYPILOT_DOCKER_USERNAME") and task.run.startswith("unset SKYPILOT_DOCKER")

    from yeto.launch.credentials import registry_login_for

    modal_args = _sky_launch("modal:1xa100")
    launcher._prepare_rl_args(modal_args)
    assert modal_args.rl_image == pins.VERL_IMAGE and registry_login_for(modal_args) is None


# --- 5.2: cards ---------------------------------------------------------------------------------

def _specs(*gpus):
    return [SimpleNamespace(gpu=g) for g in gpus]


def _verl_args():
    return SimpleNamespace(training_mode="rl", rl_backend="verl", rl_image=pins.VERL_IMAGE, tensor_parallel=1,
                           pipeline_parallel=1, expert_parallel=1, context_parallel=1)


def test_sm80_sm89_sm90_are_allowed_for_the_fsdp2_trainer():
    assert set(pins.VERL_FSDP_CARD_ARCHES) == {"sm80", "sm89", "sm90"}
    from yeto.launch.trainer_image_check import verl_card_problem

    assert verl_card_problem("H100") is None and verl_card_problem("L40S") is None  # verified (5.3 for L40S)
    for card in ("A100", "A100-80GB", "L4"):
        message, refused = verl_card_problem(card)
        assert not refused and message.startswith("UNVERIFIED") and "an allowed architecture" in message
    message, refused = verl_card_problem("B300")  # above the floor, outside the list: warned, not refused
    assert not refused and "outside the verified list" in message
    for card in ("T4", "V100"):
        message, refused = verl_card_problem(card)
        assert refused and "below sm80" in message
    assert verl_card_problem("910B") is None  # Ascend: not a CUDA card
    assert verl_card_problem("no-such-card") is None


def test_l40s_warns_until_it_is_in_the_verification_record(monkeypatch):
    from yeto.launch.trainer_image_check import check_trainer_images

    out = io.StringIO()
    record = dict(pins.VERL_FSDP_VERIFIED_CARDS)
    record.pop("L40S", None)
    monkeypatch.setattr(pins, "VERL_FSDP_VERIFIED_CARDS", record)
    (warning,) = check_trainer_images(_verl_args(), _specs("L40S"), out=out)
    assert "UNVERIFIED" in warning and "L40S" in warning and "UNVERIFIED" in out.getvalue()
    monkeypatch.setattr(pins, "VERL_FSDP_VERIFIED_CARDS", {**record, "L40S": "test evidence"})
    out = io.StringIO()
    assert check_trainer_images(_verl_args(), _specs("L40S"), out=out) == [] and out.getvalue() == ""


def test_a_card_below_the_floor_is_refused_before_any_order():
    from yeto.launch.trainer_image_check import check_trainer_images

    with pytest.raises(ValueError, match=r"--gpu island 0 .*below sm80"):
        check_trainer_images(_verl_args(), _specs("T4"), out=io.StringIO())


def test_the_megatron_engine_island_and_miles_are_not_checked_by_the_verl_rule():
    from yeto.launch.trainer_image_check import check_trainer_images

    megatron = _verl_args()
    megatron.tensor_parallel = 2
    assert check_trainer_images(megatron, _specs("T4"), out=io.StringIO()) == []
    miles = SimpleNamespace(training_mode="rl", rl_backend="miles", rl_image="")
    assert check_trainer_images(miles, _specs("T4"), out=io.StringIO()) == []


# --- 5.1: the registry is asked for the image before any order -----------------------------------

class _Resp:
    def __init__(self, status=200, body=b"{}"):
        self.status, self.headers, self._body = status, {}, body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _registry(manifests: dict[str, int], *, scheme="Bearer", password=None):
    """A fake registry opener: ``manifests`` maps an Authorization header kind
    ("anon" / "token" / "basic") to the manifest HEAD's HTTP status."""
    import urllib.error

    seen = []

    def urlopen(req, timeout=None):
        auth = req.headers.get("Authorization", "")
        seen.append((req.get_method(), req.full_url, auth))
        if "/token?" in req.full_url:
            if password is not None and auth != "Basic " + __import__("base64").b64encode(
                    f"puller:{password}".encode()).decode():
                raise urllib.error.HTTPError(req.full_url, 401, "no", {}, None)
            return _Resp(200, b'{"token": "t0k"}')
        kind = "token" if auth == "Bearer t0k" else "basic" if auth.startswith("Basic ") else "anon"
        code = manifests.get(kind, 401)
        if code == 200:
            return _Resp(200)
        headers = {}
        if code == 401 and kind == "anon":
            headers["WWW-Authenticate"] = (
                'Bearer realm="https://reg.example/token",service="reg.example",scope="repository:o/r:pull"'
                if scheme == "Bearer" else 'Basic realm="https://reg.example/",service="ecr.amazonaws.com"')
        raise urllib.error.HTTPError(req.full_url, code, "x", headers, None)

    return urlopen, seen


REF = "docker:reg.example/o/r@sha256:" + "a" * 64


def test_the_manifest_head_follows_the_registry_token_flow():
    from yeto.launch import image_manifest as im

    opener, seen = _registry({"token": 200})
    assert im.manifest_status(REF, urlopen=opener)[0] == im.PRESENT
    assert seen[0][0] == "HEAD" and seen[0][1] == "https://reg.example/v2/o/r/manifests/sha256:" + "a" * 64
    assert "service=reg.example" in seen[1][1] and "scope=repository%3Ao%2Fr%3Apull" in seen[1][1]
    assert im.manifest_status(REF, urlopen=_registry({"token": 404})[0])[0] == im.MISSING
    status, detail = im.manifest_status(REF, urlopen=_registry({"token": 403})[0])
    assert status == im.DENIED and "anonymously" in detail
    login = {"SKYPILOT_DOCKER_USERNAME": "puller", "SKYPILOT_DOCKER_PASSWORD": "pw-SENTINEL",
             "SKYPILOT_DOCKER_SERVER": "reg.example"}
    opener, seen = _registry({"token": 200}, password="pw-SENTINEL")
    assert im.manifest_status(REF, login, urlopen=opener)[0] == im.PRESENT
    assert seen[1][2].startswith("Basic ") and "pw-SENTINEL" not in seen[1][1]  # the login goes in a header only
    ecr, _ = _registry({"basic": 200}, scheme="Basic")
    assert im.manifest_status(REF, login, urlopen=ecr)[0] == im.PRESENT  # ECR: Basic, no token endpoint
    assert im.manifest_status(REF, urlopen=ecr)[0] == im.DENIED


def test_an_unreachable_registry_is_unknown_and_refs_are_split():
    import urllib.error

    from yeto.launch import image_manifest as im

    def down(req, timeout=None):
        raise urllib.error.URLError("no route")

    assert im.manifest_status(REF, urlopen=down)[0] == im.UNKNOWN
    assert im.split_ref("docker:ghcr.io/o/r@sha256:" + "b" * 64) == ("ghcr.io", "o/r", "sha256:" + "b" * 64)
    assert im.split_ref("docker:radixark/miles@sha256:" + "b" * 64)[:2] == ("registry-1.docker.io", "radixark/miles")
    assert im.split_ref("docker:ubuntu@sha256:" + "b" * 64)[:2] == ("registry-1.docker.io", "library/ubuntu")
    with pytest.raises(ValueError):
        im.split_ref("verl-build:" + "c" * 40)


def _redirecting(hops: dict[str, str], final: int = 200):
    """A fake opener: ``hops`` maps a URL to the Location it redirects to (HTTP 307);
    any other URL answers ``final``. Records (method, url, Authorization) per request."""
    import urllib.error

    seen = []

    def urlopen(req, timeout=None):
        seen.append((req.get_method(), req.full_url, req.headers.get("Authorization")))
        if req.full_url in hops:
            raise urllib.error.HTTPError(req.full_url, 307, "redirect", {"Location": hops[req.full_url]}, None)
        if final == 200:
            return _Resp(200)
        raise urllib.error.HTTPError(req.full_url, final, "x", {}, None)

    return urlopen, seen


MANIFEST_URL = "https://reg.example/v2/o/r/manifests/sha256:" + "a" * 64


def test_a_redirect_to_another_host_does_not_carry_the_registry_login():
    # ECR / Docker Hub answer a manifest or blob request with a redirect to a presigned storage URL on
    # another host: the Authorization header (a registry token or the Basic login) must stay behind
    from yeto.launch import image_manifest as im

    login = {"SKYPILOT_DOCKER_USERNAME": "puller", "SKYPILOT_DOCKER_PASSWORD": "pw-SENTINEL",
             "SKYPILOT_DOCKER_SERVER": "reg.example"}
    basic = "Basic " + __import__("base64").b64encode(b"puller:pw-SENTINEL").decode()
    hops = {MANIFEST_URL: "https://storage.example/presigned?sig=1"}
    plain, seen = _redirecting(hops)
    ecr, ecr_seen = _registry({"basic": 200}, scheme="Basic")

    def opener(req, timeout=None):  # anonymous HEAD -> Basic challenge; the Basic HEAD -> redirect
        if req.headers.get("Authorization") or "storage.example" in req.full_url:
            return plain(req, timeout)
        return ecr(req, timeout)

    assert im.manifest_status(REF, login, urlopen=opener)[0] == im.PRESENT
    assert seen[0] == ("HEAD", MANIFEST_URL, basic)
    assert seen[1] == ("HEAD", "https://storage.example/presigned?sig=1", None)  # login dropped

    # same host, https: the header follows (a registry moving a path under itself); a relative Location too
    same, seen = _redirecting({MANIFEST_URL: "/v2/o/r/moved"})
    with im._open(MANIFEST_URL, method="HEAD", headers={"Authorization": "Bearer t"}, timeout=1, urlopen=same):
        pass
    assert seen[1] == ("HEAD", "https://reg.example/v2/o/r/moved", "Bearer t")
    # a hop back to the registry after leaving it does not get the header back, nor does http on the same host
    away, seen = _redirecting({MANIFEST_URL: "https://cdn.example/x", "https://cdn.example/x": MANIFEST_URL + "?b"})
    with im._open(MANIFEST_URL, method="HEAD", headers={"Authorization": "Bearer t"}, timeout=1, urlopen=away):
        pass
    assert [a for _, _, a in seen] == ["Bearer t", None, None]
    down, seen = _redirecting({MANIFEST_URL: "http://reg.example/v2/o/r/plain"})
    with im._open(MANIFEST_URL, method="HEAD", headers={"Authorization": "Bearer t"}, timeout=1, urlopen=down):
        pass
    assert seen[1][2] is None
    # a redirect loop ends as "registry not reached", not a hang
    loop, _ = _redirecting({MANIFEST_URL: MANIFEST_URL})
    assert im.manifest_status(REF, urlopen=loop)[0] == im.UNKNOWN


def test_the_default_opener_does_not_follow_redirects_itself():
    # urllib's own redirect handler would copy the Authorization header to the new host: the module's
    # opener turns a redirect into an HTTPError for image_manifest._open to decide
    import http.server
    import threading
    import urllib.error

    from yeto.launch import image_manifest as im

    hits = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_HEAD(self):
            hits.append((self.path, self.headers.get("Authorization")))
            self.send_response(307 if self.path == "/a" else 200)
            self.send_header("Location", f"http://127.0.0.1:{self.server.server_port}/b")
            self.end_headers()

        def log_message(self, *a):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        req = __import__("urllib.request").request.Request(
            f"http://127.0.0.1:{server.server_port}/a", method="HEAD", headers={"Authorization": "Bearer t"})
        with pytest.raises(urllib.error.HTTPError) as err:
            im._OPENER.open(req, timeout=5)
        assert err.value.code == 307
        assert hits == [("/a", "Bearer t")]  # /b was never asked
    finally:
        server.shutdown()
        server.server_close()


def _sky_specs(*gpus):
    from yeto.gpu_spec import parse_gpu_spec

    return [s for g in gpus for s in parse_gpu_spec(g)]


def test_a_sky_verl_island_refuses_an_image_its_registry_does_not_have(monkeypatch):
    from yeto.launch import image_manifest as im
    from yeto.launch.trainer_image_check import check_trainer_images

    monkeypatch.setenv(im.CHECK_ENV, "on")
    asked = []

    def status(ref, login=None, **kw):
        asked.append((ref, login))
        return answers[ref]

    monkeypatch.setattr(im, "manifest_status", status)
    answers = {IMG: (im.MISSING, "reg has no manifest")}
    with pytest.raises(ValueError, match=r"--gpu island 0 .*trainer image .*no manifest.*YETO_VERL_IMAGE_REGISTRY"):
        check_trainer_images(_verl_args(), _sky_specs("aws:1xh100@us-west-2"), out=io.StringIO())
    answers[IMG] = (im.DENIED, "private")
    with pytest.raises(ValueError, match="SKYPILOT_DOCKER_USERNAME"):
        check_trainer_images(_verl_args(), _sky_specs("aws:1xh100@us-west-2"), out=io.StringIO())
    answers[IMG] = (im.UNKNOWN, "not reached")
    (warning,) = check_trainer_images(_verl_args(), _sky_specs("aws:1xh100@us-west-2"), out=io.StringIO())
    assert "not checked: not reached" in warning
    answers[IMG] = (im.PRESENT, "ok")
    assert check_trainer_images(_verl_args(), _sky_specs("aws:1xh100@us-west-2"), out=io.StringIO()) == []
    asked.clear()
    assert check_trainer_images(_verl_args(), _sky_specs("modal:1xh100"), out=io.StringIO()) == []
    assert asked == []  # a Modal island builds verl-build: no registry asked
    monkeypatch.setenv(im.CHECK_ENV, "off")
    answers[IMG] = (im.MISSING, "x")
    assert check_trainer_images(_verl_args(), _sky_specs("aws:1xh100@us-west-2"), out=io.StringIO()) == []


def test_the_manifest_check_uses_the_login_the_island_pulls_with(monkeypatch):
    from yeto.launch import image_manifest as im
    from yeto.launch.trainer_image_check import check_trainer_images
    monkeypatch.setenv(im.CHECK_ENV, "on")
    monkeypatch.setenv("SKYPILOT_DOCKER_USERNAME", "puller")
    monkeypatch.setenv("SKYPILOT_DOCKER_PASSWORD", "pw")
    monkeypatch.setenv("SKYPILOT_DOCKER_SERVER", REGISTRY.split("/", 1)[0])
    asked = []
    monkeypatch.setattr(im, "manifest_status", lambda ref, login=None, **kw: asked.append(login) or (im.PRESENT, ""))
    check_trainer_images(_verl_args(), _sky_specs("aws:1xh100@us-west-2", "nebius:1xh100"), out=io.StringIO())
    assert len(asked) == 1 and asked[0]["SKYPILOT_DOCKER_USERNAME"] == "puller"  # one ask per image

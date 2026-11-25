"""A sky island with an ``s3://`` store on a custom endpoint (Cloudflare R2, MinIO).

sky Storage takes an ``s3://`` source for an AWS S3 bucket: a store on R2 is looked up
at AWS and refused ("Attempted to use a non-existent bucket as a source": a
Modal island moved to AWS never started). Such a sky island now reaches the store like a
Modal or ``ssh:`` island and a sky sampler node: by its object API from a local stage,
the URL, endpoint and credentials in its environment (the secret as a task secret), no
sky Storage mount. The dry run and the launch preflight say so before any order.

The launcher runs with fake sky / Modal backends (``test_trainer_launch_render``); no
cloud resource is started."""

from __future__ import annotations

import sys
import types

import pytest
from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401  (fixtures)
from test_trainer_launch_render import IMAGE, _Provisioned, _setup

import yeto.launcher as launcher
from yeto.cloud import s3_store
from yeto.launch import storage
from yeto.rl.engine.resume import STORE_IMPL_ENV

STORE = "s3://yeto-r2-test/runs/p9"
ENDPOINT = "https://0123456789abcdef.r2.cloudflarestorage.com"
KEY_ID = "R2KEYID-SENTINEL-0001"
SECRET = "r2-secret-SENTINEL-0123456789"


@pytest.fixture
def _r2(monkeypatch):
    monkeypatch.setenv(s3_store.KEY_ID_ENV, KEY_ID)
    monkeypatch.setenv(s3_store.SECRET_ENV, SECRET)
    monkeypatch.setenv(s3_store.ENDPOINT_ENV, ENDPOINT)
    for name in ("AWS_ENDPOINT_URL", "AWS_ENDPOINT_URL_S3", s3_store.PROFILE_ENV):
        monkeypatch.delenv(name, raising=False)


def _args(gpu, store=STORE):
    args = _launcher_args("ports", ("--controller", "local", "--rl-single-island-no-sync", "--rl-image", IMAGE,
                                    "--rl-checkpoint-store", store), gpu=gpu)
    args.keep, args.recover_timeout, args.controller_poll = False, 0.0, 30.0
    return args


def _launched(monkeypatch, tmp_path, gpu, store=STORE):
    """The sky tasks (or Modal island configs) ``launcher.run`` orders for ``gpu``."""
    monkeypatch.setenv("YETO_PREFLIGHT_MODEL", "off")
    monkeypatch.setenv("YETO_SKY_SERVER_CHECK", "off")
    _setup(monkeypatch, tmp_path, [])
    tasks = []

    def launch(task, **kw):
        tasks.append(task)
        return f"rid-{kw.get('cluster_name')}"

    sys.modules["sky"].launch = launch
    try:
        launcher.run(_args(gpu, store))
    except _Provisioned:
        pass
    return tasks


def _texts(task) -> str:
    return "\n".join(str(getattr(task, a, "") or "") for a in ("setup", "run", "envs", "file_mounts"))


def _store_mounts(task) -> list:
    """Every sky Storage / file mount of the store's bucket on the task."""
    mounts = dict(getattr(task, "storage_mounts", None) or {})
    found = [path for path, m in mounts.items()
             if path == storage.ELASTIC_CHECKPOINT_STORE_MOUNT or "yeto-r2-test" in str(getattr(m, "source", ""))]
    found += [path for path, src in dict(getattr(task, "file_mounts", None) or {}).items() if "yeto-r2-test" in str(src)]
    return found


@pytest.mark.usefixtures("_r2")
@pytest.mark.parametrize("gpu", ["aws:1xh100@us-west-2", "nebius:1x8xh100@eu-north1"])
def test_a_sky_trainer_island_reaches_an_r2_store_by_object_api_not_sky_storage(gpu, tmp_path, monkeypatch, capsys):
    (task,) = _launched(monkeypatch, tmp_path, gpu)
    assert _store_mounts(task) == []  # sky Storage would look the R2 bucket up at AWS
    envs = dict(task.envs)
    assert {k: envs.get(k) for k in (s3_store.URI_ENV, s3_store.ENDPOINT_ENV, s3_store.KEY_ID_ENV, STORE_IMPL_ENV,
                                     s3_store.STAGE_ENV)} == {
        s3_store.URI_ENV: STORE, s3_store.ENDPOINT_ENV: ENDPOINT, s3_store.KEY_ID_ENV: KEY_ID,
        STORE_IMPL_ENV: s3_store.STORE_IMPL, s3_store.STAGE_ENV: s3_store.DEFAULT_STAGE}
    # the secret is a sky task secret (SkyPilot keeps it out of the task's envs and YAML): never a
    # plain env, never in the scripts
    assert task.secrets[s3_store.SECRET_ENV] == SECRET and s3_store.SECRET_ENV not in envs
    assert SECRET not in _texts(task)
    assert "import boto3" in task.setup  # the object API's client
    err = capsys.readouterr().err
    assert f"--rl-checkpoint-store {STORE}: its S3 endpoint is not AWS S3" in err and "no sky Storage mount" in err
    assert ENDPOINT not in err  # the endpoint (an R2 account id) is not printed


@pytest.mark.usefixtures("_r2")
def test_an_aws_s3_store_is_still_mounted(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv(s3_store.ENDPOINT_ENV)
    (task,) = _launched(monkeypatch, tmp_path, "aws:1xh100@us-west-2", store="s3://yeto-aws-test/runs/p9")
    mount = task.storage_mounts[storage.ELASTIC_CHECKPOINT_STORE_MOUNT]
    assert mount.source == "s3://yeto-aws-test" and mount.mode == "mount"
    assert s3_store.URI_ENV not in dict(task.envs)  # unchanged: the mount, no object API
    assert "no sky Storage mount" not in capsys.readouterr().err


@pytest.mark.usefixtures("_r2")
def test_an_explicit_aws_endpoint_is_still_mounted():
    args = types.SimpleNamespace(rl_checkpoint_store=STORE, rl_object_store_endpoint="https://s3.us-west-2.amazonaws.com",
                                 rl_object_store_profile=None)
    import os

    os.environ.pop(s3_store.ENDPOINT_ENV)
    assert storage.bucket_unmountable_reason(args, "aws") is None and storage.mounts_store_bucket(args, "aws")


@pytest.mark.usefixtures("_r2")
def test_only_sky_islands_take_the_object_api_route():
    args = types.SimpleNamespace(rl_checkpoint_store=STORE, rl_object_store_endpoint=None, rl_object_store_profile=None)
    assert storage.bucket_unmountable_reason(args, "aws") is not None
    assert storage.bucket_unmountable_reason(args, "nebius") is not None
    assert storage.bucket_unmountable_reason(args, "modal") is None  # Modal: no mount, already by object API
    assert storage.bucket_unmountable_reason(args, "ssh") is None  # ssh: no mount, already by object API
    args.rl_checkpoint_store = "/mnt/shared/run"
    assert storage.bucket_unmountable_reason(args, "aws") is None


@pytest.mark.usefixtures("_r2")
def test_the_dry_run_shows_the_object_api_route_before_any_order(monkeypatch, capsys):
    monkeypatch.setenv("YETO_PREFLIGHT_MODEL", "off")
    monkeypatch.setenv("YETO_SKY_SERVER_CHECK", "off")
    args = _args("aws:1xh100@us-west-2")
    launcher._prepare_rl_args(args)
    (entry,) = launcher.dry_run_plan(args)["island_requests"]
    assert entry["checkpoint_store_access"] == {"route": "object API", "sky_storage_mount": False,
                                                "why": "its S3 endpoint is not AWS S3 (R2 / MinIO): sky Storage "
                                                       "cannot mount it"}
    assert "island 0 (aws): --rl-checkpoint-store" in capsys.readouterr().err
    args = _args("modal:1xh100")
    launcher._prepare_rl_args(args)
    (entry,) = launcher.dry_run_plan(args)["island_requests"]
    assert "checkpoint_store_access" not in entry  # a Modal island's plan is unchanged


@pytest.mark.usefixtures("_r2")
def test_a_sky_island_without_store_credentials_is_refused_in_the_dry_run(monkeypatch):
    monkeypatch.delenv(s3_store.KEY_ID_ENV)
    monkeypatch.delenv(s3_store.SECRET_ENV)
    with pytest.raises(ValueError, match="needs store credentials"):
        launcher._prepare_rl_args(_args("aws:1xh100@us-west-2"))


@pytest.mark.usefixtures("_r2")
def test_the_modal_island_task_is_unchanged_by_the_r2_route(tmp_path, monkeypatch):
    """A Modal trainer island with the same R2 store: its island config is what main renders
    (the object API from a local stage, as before)."""
    from yeto.launch.storage import store_object_envs

    args = _args("modal:1xh100")
    launcher._prepare_rl_args(args)
    envs = store_object_envs(args, "modal")
    assert envs[s3_store.STAGE_ENV] == s3_store.DEFAULT_STAGE and envs[s3_store.ENDPOINT_ENV] == ENDPOINT


@pytest.mark.usefixtures("_r2")
def test_a_multi_node_sky_island_stages_the_r2_store_on_every_node(tmp_path, monkeypatch):
    """No shared mount to fall back on: each node stages the store, and each rank fetches its
    own round-cut shards (as on a clustered Modal or an ``ssh:`` island)."""
    (task,) = _launched(monkeypatch, tmp_path, "nebius:2x2xh100@eu-north1")
    assert _store_mounts(task) == []
    envs = dict(task.envs)
    assert envs[s3_store.STAGE_ENV] == s3_store.DEFAULT_STAGE and envs[s3_store.LAZY_CUT_SHARDS_ENV] == "1"
    assert envs[s3_store.ENDPOINT_ENV] == ENDPOINT and task.secrets[s3_store.SECRET_ENV] == SECRET

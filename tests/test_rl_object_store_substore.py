"""A store path below the island's store path on an object store (s3, oss) is a
sub-store (``s3_store.sub_store_uri``). CPU, fake S3 client, no cloud resource.

Regression: a synced group whose samplers follow their own trainer
(``--rl-sampler-follow trainer``) gives each trainer the sub-store
``trainer<k>`` (``--rl-sample-store <store path>/trainer<k>``), while the
island's store URI (``YETO_RL_STORE_S3_URI``) names the run's store. The store
hook built the object API (and the staged root, which ignores the path) from the
URI alone: the trainer read the buffer and wrote its releases under the run's
root, its samplers under ``trainer<k>/``, and the trainer waited for groups that
never came."""

from __future__ import annotations

import shlex
from pathlib import Path

import pytest

from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from tests.test_rl_object_store import FakeS3
from yeto.cloud import oss_store, s3_store
from yeto.cloud.s3_store import sub_store_uri
from yeto.rl.adapters.miles import island_entry
from yeto.rl.engine import sample_roles
from yeto.rl.engine.resume import STORE_IMPL_ENV, store_for

S3_URI = "s3://bkt/runs/a"
MOUNT = "~/yeto-checkpoint-store/runs/a"


@pytest.mark.parametrize(("path", "expected"), [
    (MOUNT, S3_URI),
    (MOUNT + "/", S3_URI),
    (MOUNT + "/trainer1", S3_URI + "/trainer1"),
    (str(Path(MOUNT).expanduser() / "trainer0"), S3_URI + "/trainer0"),  # expanded home
    (MOUNT + "/island-cuts/island2", S3_URI + "/island-cuts/island2"),
    ("/elsewhere/store", S3_URI),  # not below the store path: the run's store, as before
    ("~/yeto-checkpoint-store/runs/b/trainer0", S3_URI),
])
def test_sub_store_uri(path, expected):
    assert sub_store_uri(S3_URI, path, MOUNT) == expected


def test_sub_store_uri_of_a_bucket_root_store():
    assert sub_store_uri("s3://bkt", "~/yeto-checkpoint-store/trainer0",
                         "~/yeto-checkpoint-store") == "s3://bkt/trainer0"


def _s3_env(stage=None):
    return {STORE_IMPL_ENV: s3_store.STORE_IMPL, s3_store.URI_ENV: S3_URI,
            **({s3_store.STAGE_ENV: str(stage)} if stage else {})}


def test_a_staged_s3_sub_store_has_its_own_root_and_prefix(tmp_path):
    client = FakeS3()
    env = _s3_env(tmp_path / "stage")
    root = store_for(MOUNT, environ=env, client=client)
    sub = store_for(f"{MOUNT}/trainer0", environ=env, client=client)
    assert root.uri == S3_URI and sub.uri == S3_URI + "/trainer0"
    assert Path(root.root) == tmp_path / "stage" / "bkt" / "runs" / "a"
    assert Path(sub.root) == tmp_path / "stage" / "bkt" / "runs" / "a" / "trainer0"
    sub.objects.write("sample-buffer/index.jsonl", b"{}")
    assert ("bkt", "runs/a/trainer0/sample-buffer/index.jsonl") in client.objects


def test_a_mounted_s3_sub_store_keeps_its_path():
    # a bucket mount (no stage): the path is the root, the URI the same directory
    path = Path(s3_store.ISLAND_MOUNT).expanduser() / "runs" / "a" / "trainer1"
    store = store_for(str(path), environ=_s3_env(), client=FakeS3())
    assert store.uri == S3_URI + "/trainer1" and Path(store.root) == path


def test_an_oss_sub_store_has_its_own_root_and_prefix(tmp_path):
    env = {STORE_IMPL_ENV: oss_store.STORE_IMPL, oss_store.URI_ENV: "oss://bkt/runs/a",
           oss_store.STAGE_ENV: str(tmp_path / "stage")}
    sub = store_for(f"{oss_store.DEFAULT_STAGE}/bkt/runs/a/trainer0", environ=env, client=FakeS3())
    assert sub.uri == "oss://bkt/runs/a/trainer0"
    assert Path(sub.root) == tmp_path / "stage" / "bkt" / "runs" / "a" / "trainer0"
    root = store_for(f"{oss_store.DEFAULT_STAGE}/bkt/runs/a", environ=env, client=FakeS3())
    assert root.uri == "oss://bkt/runs/a"


# ------------------------------------------- the launch: a follow-trainer group on R2
GROUP_GPU = "modal:1xh100,modal:1xh100,modal:1xl40s,modal:1xl40s"
GROUP = ("--rl-sync-preset", "full-parameter", "--rl-sync-full-parameter-rounds", "async",
         "--rl-parameter-layout-sha256", "e" * 64, "--fragments", "4",
         "--local-rl-rounds-per-sync", "2",
         "--rl-sample-role", "trainer", "--rl-island-sample-role", "2:sampler",
         "--rl-island-sample-role", "3:sampler", "--rl-max-policy-age", "2",
         "--rl-sampler-follow", "trainer", "--rl-sample-wait-s", "900", "--rl-stall-timeout", "900",
         "--rl-checkpoint-store", S3_URI,
         "--rl-object-store-endpoint", "https://r2.example.invalid",
         "--rl-algorithm-spec", "examples/rl_algorithms/decoupled-ppo-c2.json",
         "--cluster-prefix", "substore-group")


@pytest.fixture(autouse=True)
def _island_hmac_key(monkeypatch):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "substore-test-key")


def test_each_trainer_and_its_sampler_resolve_one_s3_sub_store(monkeypatch, tmp_path):
    from yeto.launch.storage import store_object_envs
    from yeto.launcher import _prepare_rl_args, dry_run_plan

    monkeypatch.setenv(s3_store.KEY_ID_ENV, "AKIDTEST")
    monkeypatch.setenv(s3_store.SECRET_ENV, "test-secret")
    args = _launcher_args("ports", GROUP, gpu=GROUP_GPU)
    args.controller = "local"
    _prepare_rl_args(args)
    plan = dry_run_plan(args)
    trainers = [i["learner_id"] for i in plan["island_requests"] if i["sample_role"] == "trainer"]
    resolved = {}
    for m, island in enumerate(plan["island_requests"]):
        learner = trainers.index(island["learner_id"]) if island["learner_id"] in trainers else 0
        tail = island["learner_command"].split("-m yeto.rl.adapters.miles.island_entry", 1)[1]
        parsed = island_entry.parse_args([{"$LEARNER_ID": str(learner), "$SYNCER_ADDR": "10.0.0.1:1"}.get(t, t)
                                          for t in shlex.split(tail)])
        path = parsed.rl_sample_store or parsed.rl_resume_store
        # the island's environment, as the launcher builds it for this island
        env = store_object_envs(sample_roles.island_args(args, m, 4), "modal")
        env[s3_store.STAGE_ENV] = str(tmp_path / f"stage{m}")  # one container each
        store = store_for(path, environ=env, client=FakeS3())
        resolved[m] = (store.uri, Path(store.root).relative_to(tmp_path / f"stage{m}"))
    # trainer k and its sampler (island 2 + k): one prefix, one staged layout; the pairs differ
    for k in range(2):
        assert resolved[k] == resolved[2 + k]
        assert resolved[k][0] == f"{S3_URI}/trainer{k}"
    assert resolved[0] != resolved[1]

"""Platform wiring for a Modal B300 cluster over an S3-compatible (R2) store (CPU,
fakes; no cloud resource):

* every Modal island gets ``NCCL_RAS_ENABLE=0`` unless the launch sets it;
* ``--modal-store-secret`` attaches a named Modal Secret and the island reads the
  store credential from the Secret's ``AWS_*`` variables (only when told so);
* an ``s3://`` store on a Modal island goes by object API from a local stage,
  ``--rl-object-store-endpoint`` reaches the head and the islands with
  ``region=auto``;
* a whole clustered restart re-runs ``island_main`` in fresh containers that
  resume from the ``LATEST`` the previous incarnation pushed to the store.
"""

from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace

import pytest

from tests.test_modal_runner import _cfg, fake_modal
from tests.test_rl_object_store import CREDS, FakeS3
from yeto import modal_runner as mr
from yeto.cloud import s3_store
from yeto.launch import storage
from yeto.rl.engine import resume as rs
from yeto.rl.engine.resume import STORE_IMPL_ENV, store_for

URI = "s3://yeto-bucket/runs/k31"
R2 = "https://acct.r2.cloudflarestorage.com"


# --- launch: default NCCL envs, the store Secret ------------------------------


def _launch_args(extra):
    from tests.test_launch_auto import _args

    return _args(["--gpu", "modal:2x8xb300", "--cluster-prefix", "run", "--training-mode", "rl",
                  "--rl-image", "docker:ghcr.io/x/miles@sha256:" + "d" * 64,
                  "--rl-single-island-no-sync", *extra])


def _modal_cfg(monkeypatch, tmp_path, extra):
    from tests.test_launch_auto import _specs
    from yeto.launcher import build_modal_island_config

    monkeypatch.setenv("HOME", str(tmp_path))
    for name in (*s3_store.CREDENTIAL_ENVS, *s3_store.SETTING_ENVS, "AWS_ENDPOINT_URL"):
        monkeypatch.delenv(name, raising=False)
    args = _launch_args(extra)
    (spec,) = _specs(args.gpu)
    return build_modal_island_config(args, spec, 0, SimpleNamespace(run="true", envs={}, setup=""), "none")


def test_modal_islands_get_nccl_ras_off_unless_the_launch_says_otherwise(monkeypatch, tmp_path):
    assert mr.MODAL_ISLAND_DEFAULT_ENVS == {"NCCL_RAS_ENABLE": "0"}
    cfg = _modal_cfg(monkeypatch, tmp_path, [])
    assert cfg.envs["NCCL_RAS_ENABLE"] == "0" and "NCCL_NVLS_ENABLE" not in cfg.envs
    cfg = _modal_cfg(monkeypatch, tmp_path, ["--modal-env", "NCCL_RAS_ENABLE=1",
                                             "--modal-env", "NCCL_NVLS_ENABLE=0"])
    assert cfg.envs["NCCL_RAS_ENABLE"] == "1" and cfg.envs["NCCL_NVLS_ENABLE"] == "0"


def test_store_secret_is_attached_and_the_island_reads_the_aws_pair(monkeypatch, tmp_path):
    cfg = _modal_cfg(monkeypatch, tmp_path, ["--rl-checkpoint-store", URI,
                                             "--modal-store-secret", "k31-store",
                                             "--rl-object-store-endpoint", R2])
    assert cfg.named_secrets == ("k31-store",)
    assert cfg.checkpoint_store_volume_name is None and cfg.checkpoint_store_mount is None
    assert "YETO_RL_STORE_MODAL_VOLUME" not in cfg.envs
    # the island task's store environment (rl_island_task puts it in the task envs)
    envs = storage.store_object_envs(_launch_args(["--rl-checkpoint-store", URI, "--modal-store-secret",
                                                   "k31-store", "--rl-object-store-endpoint", R2]), "modal")
    assert envs[STORE_IMPL_ENV] == s3_store.STORE_IMPL and envs[s3_store.URI_ENV] == URI
    assert envs[s3_store.STAGE_ENV] == s3_store.DEFAULT_STAGE  # no mount on Modal
    assert envs[s3_store.ENDPOINT_ENV] == R2 and envs[s3_store.REGION_ENV] == "auto"
    assert envs[s3_store.CREDENTIALS_ENV] == s3_store.CREDENTIALS_AWS_ENV
    assert not any(name in envs for name in s3_store.CREDENTIAL_ENVS)  # the Secret carries them
    cfg.validate()
    assert mr.ModalIslandConfig.from_json(cfg.to_json()) == cfg
    # both named Secrets when the sandbox one is given too
    cfg = _modal_cfg(monkeypatch, tmp_path, ["--rl-checkpoint-store", URI, "--modal-store-secret", "r2",
                                             "--modal-sandbox-secret", "sbx"])
    assert cfg.named_secrets == ("sbx", "r2")


def test_store_secret_is_refused_off_modal_and_without_credentials_the_launch_ships_its_own(monkeypatch):
    from tests.test_launch_auto import _args, _specs
    from yeto.launch.args import check_cloud_prerequisites

    args = _args(["--gpu", "aws:1xh100@us-east-1", "--modal-store-secret", "r2"])
    with pytest.raises(ValueError, match="--modal-store-secret"):
        check_cloud_prerequisites(_specs(args.gpu), args=args, modal_ok=True)
    with pytest.raises(ValueError, match="not a Modal Secret name"):
        storage.modal_store_secret_name(SimpleNamespace(modal_store_secret="bad name"))
    # no Secret: the launcher's store credential is required and shipped, as on sky
    for name in (*s3_store.CREDENTIAL_ENVS, s3_store.CREDENTIALS_ENV):
        monkeypatch.delenv(name, raising=False)
    plain = SimpleNamespace(rl_checkpoint_store=URI)
    with pytest.raises(ValueError, match="store credentials"):
        storage.store_object_envs(plain, "modal")
    for key, value in CREDS.items():
        monkeypatch.setenv(key, value)
    envs = storage.store_object_envs(plain, "modal", num_nodes=6)
    assert envs[s3_store.KEY_ID_ENV] == "AKIDSTORE" and s3_store.CREDENTIALS_ENV not in envs


# --- store client: AWS-named credential only on request, endpoint, region -------


def _sessions(monkeypatch):
    import boto3

    sessions = []

    class Session:
        def __init__(self, **kw):
            sessions.append(kw)

        def client(self, name, **kw):
            sessions.append(kw)
            return FakeS3()

    monkeypatch.setattr(boto3.session, "Session", Session)
    return sessions


def test_island_client_reads_the_secret_pair_only_when_told(tmp_path, monkeypatch):
    from yeto.rl.engine.object_store import ObjectStoreError

    sessions = _sessions(monkeypatch)
    aws = {"AWS_ACCESS_KEY_ID": "AKIDR2", "AWS_SECRET_ACCESS_KEY": "r2-secret", "AWS_ENDPOINT_URL": R2,
           "AWS_REGION": "auto"}
    env = {STORE_IMPL_ENV: s3_store.STORE_IMPL, s3_store.URI_ENV: URI, **aws}
    with pytest.raises(ObjectStoreError, match="never the default AWS chain"):
        store_for(tmp_path, environ=env).objects  # the launcher's AWS pair is not a store credential
    assert sessions == []
    env[s3_store.CREDENTIALS_ENV] = s3_store.CREDENTIALS_AWS_ENV  # the Secret's pair, by the launch
    assert store_for(tmp_path, environ=env).objects.bucket == "yeto-bucket"
    sessions[1].pop("config", None)  # the connection-pool Config is not part of the credential contract
    assert sessions == [{"aws_access_key_id": "AKIDR2", "aws_secret_access_key": "r2-secret",
                         "aws_session_token": None},
                        {"endpoint_url": R2, "region_name": "auto"}]
    # the YETO_RL_STORE_S3_* settings win over the AWS names
    sessions.clear()
    env.update({s3_store.ENDPOINT_ENV: "http://minio:9000", s3_store.REGION_ENV: "us-east-1"})
    store_for(tmp_path, environ=env).objects
    sessions[1].pop("config", None)
    assert sessions[1] == {"endpoint_url": "http://minio:9000", "region_name": "us-east-1"}


def test_object_store_endpoint_reaches_the_head_and_the_islands(monkeypatch):
    for name in (*s3_store.SETTING_ENVS, "AWS_ENDPOINT_URL", "AWS_REGION", "AWS_DEFAULT_REGION"):
        monkeypatch.delenv(name, raising=False)
    args = SimpleNamespace(rl_object_store_endpoint=R2)
    env = storage.store_client_environ(args, {})
    assert env == {s3_store.ENDPOINT_ENV: R2, s3_store.REGION_ENV: "auto"}
    # AWS_ENDPOINT_URL in the environment is the same thing; a configured region is kept
    env = storage.store_client_environ(SimpleNamespace(), {"AWS_ENDPOINT_URL": R2, "AWS_REGION": "weur"})
    assert env[s3_store.ENDPOINT_ENV] == R2 and env[s3_store.REGION_ENV] == "weur"
    # an explicit store endpoint is never overridden
    env = storage.store_client_environ(args, {s3_store.ENDPOINT_ENV: "http://minio:9000"})
    assert env[s3_store.ENDPOINT_ENV] == "http://minio:9000" and s3_store.REGION_ENV not in env
    # no endpoint: nothing added (AWS S3)
    assert storage.store_client_environ(SimpleNamespace(), {"X": "1"}) == {"X": "1"}
    # the launcher's own boto3 clients (head object API, code pack) see the endpoint too
    storage.apply_object_store_endpoint(args)
    import os

    assert os.environ[s3_store.ENDPOINT_ENV] == R2 and os.environ["AWS_ENDPOINT_URL"] == R2
    assert os.environ[s3_store.REGION_ENV] == "auto"
    for name in (s3_store.ENDPOINT_ENV, s3_store.REGION_ENV, "AWS_ENDPOINT_URL"):
        monkeypatch.delenv(name)


def test_s3_store_on_modal_is_head_readable_and_reachable():
    assert storage.store_unreachable_reason(URI, "modal") is None
    assert storage.store_unreachable_reason("modal-volume://v", "modal") is None
    assert "s3://" in storage.store_unreachable_reason("gs://b", "modal")
    assert storage.is_object_store_uri(URI) and not storage.is_object_store_uri("modal-volume://v")
    assert "s3" in storage.HEAD_READABLE_STORE_SCHEMES
    args = SimpleNamespace(rl_checkpoint_store=URI, rl_elastic=False, rl_single_island_no_sync=True,
                           rl_checkpoint_store_export_only=False)
    assert storage.resumes_in_new_container(args)
    assert not storage.resumes_in_new_container(SimpleNamespace(rl_checkpoint_store="gs://b", rl_elastic=False,
                                                                rl_single_island_no_sync=True))


def test_modal_trainer_binding_uses_the_object_api_for_an_s3_store(tmp_path, monkeypatch):
    from yeto.launch import trainer_island as ti
    from yeto.rl.engine.sample_buffer import FileSampleBuffer

    fake = FakeS3()
    monkeypatch.setattr(ti, "head_store_objects",
                        lambda uri, environ=None: s3_store.S3ObjectStore(uri, client=fake))
    args = SimpleNamespace(rl_checkpoint_store=URI, rollout_batch_size=4, rl_trainer_command_poll_seconds=None,
                           rl_object_store_endpoint=R2, rl_sample_role="both", rl_elastic=False)
    island = SimpleNamespace(modal_ops=None, modal_cfg=None, name="trainer", events_dir=None)
    islands = SimpleNamespace(release_incarnation=lambda *a: None, start_events=lambda *a: ())
    binding = ti.modal_binding(args, island, journal_dir=tmp_path, islands=islands)
    assert isinstance(binding.buffer.buffer, FileSampleBuffer)  # by object key, no mirror
    assert binding.buffer.buffer.store.bucket == "yeto-bucket"


def test_convertible_pool_accepts_an_s3_store_on_modal(monkeypatch):
    from tests.test_rl_convertible_pool_launch import POOL, _args
    from yeto.launch import trainer_loop

    for key, value in CREDS.items():
        monkeypatch.setenv(key, value)
    args = _args(base=[URI if t == "modal-volume://yeto-conv/run-a" else t for t in POOL])
    trainer_loop.check(args)
    args.rl_checkpoint_store = "gs://b/x"
    with pytest.raises(SystemExit, match="modal-volume:// or s3://"):
        trainer_loop.check(args)


# --- whole-cluster restart: the new containers resume from the store's LATEST ----


def test_whole_cluster_restart_resumes_from_the_latest_cut_in_the_store(monkeypatch, tmp_path):
    """Modal restarts every container of a clustered function at once (preemption,
    a failed container). Each re-run gets the same store environment and a fresh
    disk; the trainer must start from the LATEST the previous incarnation committed
    to the object store, never from what is on its own (new, empty) disk."""
    fake_modal(monkeypatch)
    monkeypatch.setattr(mr, "TAPE_SOURCE_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(mr, "TAPE_MOUNT", str(tmp_path / "vol"))
    monkeypatch.setattr(mr, "_volume_commit", lambda name: (lambda: None))
    monkeypatch.setattr(mr, "visible_gpu_names", lambda: ["NVIDIA B300"] * 8)
    sys.modules["modal.experimental"].get_cluster_info = lambda: types.SimpleNamespace(
        rank=0, container_ips=["fdaa::1", "fdaa::2"], container_ipv4_ips=["10.1.0.1", "10.1.0.2"])
    bucket = FakeS3()  # the one thing that outlives the containers
    seen: list[dict] = []

    def trainer_process(cmd, env):
        """The trainer's resume path: open the store from the island environment,
        read LATEST (a fresh disk has none), train, cut, commit."""
        incarnation = int(env["YETO_MODAL_INCARNATION"])
        disk = tmp_path / f"container-disk-{incarnation}"  # a new container, a new disk
        env = {**env, s3_store.STAGE_ENV: str(disk)}
        store = store_for("/root/yeto-checkpoint-store", environ=env, client=bucket)
        store.reload()  # pulls LATEST + its cut from the bucket when there is one
        latest = rs.read_latest(store.root)
        seen.append({"incarnation": incarnation, "resumed_from": None if latest is None else latest["cut"],
                     "stage": str(store.root), "uri": store.uri})
        cut = 0 if latest is None else int(latest["cut"]) + 1
        (store.root / "round-cuts").mkdir(parents=True, exist_ok=True)
        (store.root / "round-cuts" / f"cut-{cut:04d}.bin").write_bytes(b"w" * (cut + 1))
        (store.root / "LATEST").write_text(json.dumps({"cut": cut}), encoding="utf-8")
        store.commit()
        return 0

    monkeypatch.setattr(mr, "run_island_script", trainer_process)
    envs = {STORE_IMPL_ENV: s3_store.STORE_IMPL, s3_store.URI_ENV: URI, s3_store.STAGE_ENV: "unused",
            s3_store.CREDENTIALS_ENV: s3_store.CREDENTIALS_AWS_ENV,
            "AWS_ACCESS_KEY_ID": "AKIDR2", "AWS_SECRET_ACCESS_KEY": "r2-secret", "AWS_ENDPOINT_URL": R2}
    cfg = _cfg(gpu="B300", num_nodes=2, training_mode="rl", envs=envs,
               tape_volume_name="tapes", tape_subdir="yeto-run/l1")
    assert mr.island_main(cfg.to_json()) == 0  # first incarnation: nothing to resume from
    assert mr.island_main(cfg.to_json()) == 0  # the cluster restarted: resume
    assert mr.island_main(cfg.to_json()) == 0  # and again
    assert [(s["incarnation"], s["resumed_from"]) for s in seen] == [(0, None), (1, 0), (2, 1)]
    assert len({s["stage"] for s in seen}) == 3 and all(s["uri"] == URI for s in seen)
    keys = sorted(k for (_b, k) in bucket.objects)
    assert keys == ["runs/k31/LATEST", "runs/k31/round-cuts/cut-0000.bin", "runs/k31/round-cuts/cut-0001.bin",
                    "runs/k31/round-cuts/cut-0002.bin"]
    assert json.loads(bucket.objects[("yeto-bucket", "runs/k31/LATEST")]) == {"cut": 2}

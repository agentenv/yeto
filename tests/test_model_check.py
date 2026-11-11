"""Model preflight: config.json, the weight index and every shard it names, from one
LIST of a local directory, a bucket prefix, a Modal volume or the Hub (all fakes)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from test_rl_algorithm_provenance import _fake_sky, _no_modal_listing  # noqa: F401 - autouse: no cloud
from yeto.launch import model_check

INDEX = "model.safetensors.index.json"


def _index(shards):
    return json.dumps({"weight_map": {f"w{i}": s for i, s in enumerate(shards)}}).encode()


def _shards(n):
    return [f"model-{i + 1:05d}-of-{n:05d}.safetensors" for i in range(n)]


def test_complete_and_single_file_checkpoints_pass():
    files = {"config.json": b"{}", INDEX: _index(_shards(3)), **{s: b"" for s in _shards(3)}}
    assert model_check.weight_problems(files, files.get, "m") == []
    assert model_check.weight_problems(["config.json", "model.safetensors"], lambda n: None, "m") == []


def test_missing_shards_are_listed_first_few_with_a_count():
    shards = _shards(9)
    files = {"config.json": b"{}", INDEX: _index(shards), shards[0]: b"", shards[1]: b""}
    [problem] = model_check.weight_problems(files, files.get, "s3://wbkt/m")
    assert "7 of 9 weight shards named by model.safetensors.index.json are missing" in problem
    assert shards[2] in problem and shards[6] in problem and shards[7] not in problem and "(+2 more)" in problem


def test_no_config_no_weights_and_a_broken_index():
    assert model_check.weight_problems([], lambda n: None, "m") == [
        "no m/config.json", "no weights in m: neither model.safetensors.index.json nor model.safetensors"]
    [problem] = model_check.weight_problems(["config.json", INDEX], lambda n: b"not json", "m")
    assert "is not a valid index" in problem


def test_local_directory(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / INDEX).write_bytes(_index(_shards(2)))
    (tmp_path / _shards(2)[0]).write_bytes(b"")
    [problem] = model_check.local_problems(tmp_path)
    assert _shards(2)[1] in problem
    (tmp_path / _shards(2)[1]).write_bytes(b"")
    assert model_check.local_problems(tmp_path) == []


class _FakeS3:
    """The S3 calls S3ObjectStore.list / read make: list_objects_v2 pages of at
    most 1000 keys (the S3 / R2 limit), get_object. No cloud."""

    class exceptions:  # noqa: N801 - boto3's attribute name
        class NoSuchKey(Exception):
            pass

    def __init__(self):
        self.objects: dict[tuple[str, str], bytes] = {}
        self.pages = 0

    def put_object(self, Bucket, Key, Body):  # noqa: N803
        self.objects[(Bucket, Key)] = Body

    def get_object(self, Bucket, Key):  # noqa: N803
        if (Bucket, Key) not in self.objects:
            raise self.exceptions.NoSuchKey(Key)
        return {"Body": SimpleNamespace(read=lambda: self.objects[(Bucket, Key)])}

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        fake = self

        class Paginator:
            def paginate(self, Bucket, Prefix, **kw):  # noqa: N803
                keys = sorted(k for b, k in fake.objects if b == Bucket and k.startswith(Prefix))
                for start in range(0, max(len(keys), 1), 1000):
                    fake.pages += 1
                    yield {"Contents": [{"Key": k} for k in keys[start:start + 1000]]}

        return Paginator()


@pytest.fixture()
def bucket():
    return _FakeS3()


def test_bucket_prefix_is_listed_across_pages(bucket):
    """1205 shards: the LIST spans two pages (1000 keys each); a shard on the second
    page and one missing are both seen. Nothing but the index is read."""
    from yeto.cloud.s3_store import S3ObjectStore

    shards = _shards(1205)
    bucket.put_object(Bucket="wbkt", Key="models/m/config.json", Body=b"{}")
    bucket.put_object(Bucket="wbkt", Key=f"models/m/{INDEX}", Body=_index(shards))
    for s in shards[:-1]:
        bucket.put_object(Bucket="wbkt", Key=f"models/m/{s}", Body=b"")
    bucket.put_object(Bucket="wbkt", Key="models/m2/config.json", Body=b"{}")  # a sibling prefix is not ours
    objects = S3ObjectStore("s3://wbkt/models/m", client=bucket)
    reads = []
    real_read = objects.read
    objects.read = lambda key: reads.append(key) or real_read(key)
    [problem] = model_check.store_problems(objects, "s3://wbkt/models/m")
    assert bucket.pages == 2
    assert "1 of 1205 weight shards" in problem and shards[-1] in problem
    assert reads == [INDEX]
    bucket.put_object(Bucket="wbkt", Key=f"models/m/{shards[-1]}", Body=b"")
    assert model_check.store_problems(objects, "s3://wbkt/models/m") == []
    assert model_check.store_problems(S3ObjectStore("s3://wbkt/models/none", client=bucket), "x")[0] == \
        "no x/config.json"


def test_pool_dry_run_lists_the_bucket_copy(bucket, monkeypatch):
    from yeto.cloud.s3_store import S3ObjectStore
    from yeto.launch.nc import elastic, pool_run

    bucket.put_object(Bucket="wbkt", Key="models/m/config.json", Body=b"{}")
    bucket.put_object(Bucket="wbkt", Key=f"models/m/{INDEX}", Body=_index(_shards(2)))
    bucket.put_object(Bucket="wbkt", Key=f"models/m/{_shards(2)[0]}", Body=b"")
    monkeypatch.setattr(elastic, "s3_fence_store",
                        lambda uri, env: SimpleNamespace(objects=S3ObjectStore(uri, client=bucket)))
    [problem] = pool_run.model_problems("s3://wbkt/models/m", "unused.env")
    assert _shards(2)[1] in problem


class _Args(SimpleNamespace):
    def __init__(self, **kw):
        super().__init__(**{"model": "org/m", "model_revision": None, "modal_model_volume": None,
                            "preflight_model": "check", **kw})


def _hub(files):
    def hub(repo, revision):
        return list(files), files.get
    return hub


def test_trainer_hub_model_complete_missing_and_unreadable():
    shards = _shards(2)
    files = {"config.json": b"{}", INDEX: _index(shards), shards[0]: b""}
    record = model_check.trainer_model_check(_Args(), hub=_hub(files))
    assert record["source"] == "hub" and shards[1] in record["problems"][0]
    with pytest.raises(model_check.ModelCheckError, match="KIMI_K31_ASYNC_RL_GUIDE"):
        model_check.preflight_trainer_model(_Args(), hub=_hub(files))

    class RepositoryNotFoundError(Exception):
        pass

    def missing(repo, revision):
        raise RepositoryNotFoundError("404")

    record = model_check.trainer_model_check(_Args(), hub=missing)
    assert "not readable (RepositoryNotFoundError)" in record["problems"][0]

    def offline(repo, revision):
        raise ConnectionError("no route")

    record = model_check.trainer_model_check(_Args(), hub=offline)
    assert record["problems"] == [] and "could not reach the Hub" in record["warnings"][0]


def test_warn_and_off_modes(capsys, monkeypatch):
    files = {"config.json": b"{}"}
    assert model_check.preflight_trainer_model(_Args(preflight_model="warn"), hub=_hub(files))["problems"]
    assert "no weights" in capsys.readouterr().err
    assert model_check.preflight_trainer_model(_Args(preflight_model="off"), hub=None) is None
    monkeypatch.setenv(model_check.MODE_ENV, "off")
    assert model_check.preflight_trainer_model(_Args(preflight_model=None), hub=None) is None


def test_trainer_modal_volume_snapshot_and_fallback_to_hub():
    rev = "a" * 40
    manifest = json.dumps({"repo": "org/m", "revision": rev, "all_ok": True}).encode()
    vol_files = {"MANIFEST.json": manifest, "config.json": b"{}", INDEX: _index(_shards(2)), _shards(2)[0]: b""}
    seen = []

    def volume(name, path):
        seen.append((name, path))
        return list(vol_files), vol_files.get

    args = _Args(model_revision=rev, modal_model_volume="models")
    record = model_check.trainer_model_check(args, volume=volume, hub=None)
    assert seen == [("models", "hf/m/aaaaaaaa")] and record["source"] == "modal-volume"
    assert _shards(2)[1] in record["problems"][0]
    hub_files = {"config.json": b"{}", "model.safetensors": b""}
    record = model_check.trainer_model_check(args, volume=lambda n, p: None, hub=_hub(hub_files))
    assert record["source"] == "hub" and record["problems"] == [] and "downloads from the Hub" in record["warnings"][0]


def test_trainer_store_uri_and_local_paths(tmp_path):
    record = model_check.trainer_model_check(_Args(model="s3://wbkt/m"), store=lambda uri: [f"no {uri}/config.json"])
    assert record["source"] == "store" and record["problems"] == ["no s3://wbkt/m/config.json"]
    record = model_check.trainer_model_check(_Args(model=str(tmp_path)))
    assert record["source"] == "local" and record["problems"][0] == f"no {tmp_path}/config.json"
    record = model_check.trainer_model_check(_Args(model=str(tmp_path / "absent")))
    assert record["problems"] == [] and "not on this host" in record["warnings"][0]


def test_launch_dry_run_records_and_refuses_the_trainer_model(monkeypatch):
    """`yeto launch --dry-run` runs the model preflight: its record is in the plan, a
    problem rejects the dry run (a ValueError, nothing ordered)."""
    from test_verl_full_parameter_launch import _prepared, _verl_args
    from yeto.launcher import dry_run_plan

    args = _prepared(_verl_args())
    monkeypatch.setenv(model_check.MODE_ENV, "check")
    monkeypatch.setattr(model_check, "hub_files", _hub({"config.json": b"{}", "model.safetensors": b""}))
    plan = dry_run_plan(args)
    assert plan["trainer_model"]["source"] == "hub" and plan["trainer_model"]["problems"] == []
    monkeypatch.setattr(model_check, "hub_files", _hub({"config.json": b"{}"}))
    with pytest.raises(ValueError, match="trainer model: no weights"):
        dry_run_plan(args)

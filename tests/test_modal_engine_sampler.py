"""Engine sampler island on Modal (yeto.cloud.modal_engine_sampler): the spec, the
config from the ops variable names, the store env from either Secret style and
the CLI dry run. No Modal call."""

from __future__ import annotations

import json

import pytest

from yeto import cli
from yeto.cloud import modal_engine_sampler as mes


@pytest.fixture
def files(tmp_path):
    sel = tmp_path / "sel.json"
    sel.write_text(json.dumps({"sha256": "ab" * 32}))
    prompts = tmp_path / "p.jsonl"
    prompts.write_text("".join(json.dumps({"messages": [], "label": str(i)}) + "\n" for i in range(30)))
    return {"SELECTION": str(sel), "PROMPTS": str(prompts)}


def _env(files, **extra):
    return {"RUN": "r1", "SHARD": "modal-s0", "STORE_BUCKET": "bucket", "SAMPLER_SECRET": "store-secret",
            "SAMPLER_IMAGE": "registry.example/sampler@sha256:" + "0" * 64, **files, **extra}


def test_spec_from_the_ops_variables(files):
    c = mes.SamplerConfig.from_env(_env(files, RELOAD="hot", DROP_ZV="1", N="8", MAX_TOKENS="2048",
                                        MAX_MODEL_LEN="4096", GROUPS="4", THINKING_EFFORT="high",
                                        DATA_SLOT="1/3", EP="4"))
    spec = mes.build_spec(c, out="/vol/x")
    assert spec["sampler_id"] == "modal-s0" and spec["buffer_shards"] == ["modal-s0"]
    assert spec["expert_selection_sha256"] == "ab" * 32
    assert spec["sampling"] == {"max_tokens": 2048, "temperature": 1.0, "top_p": 1.0, "top_k": -1, "n": 8}
    assert spec["drop_zero_variance_groups"] is True
    assert spec["max_backlog_groups"] == 16  # 4 x groups
    assert spec["prompts"]["data_slot"] == "1/3" and spec["prompts"]["groups_per_round"] == 4
    assert spec["prompts"]["chat_template_kwargs"] == {"thinking_effort": "high"}
    assert spec["store"] == "/tmp/stage/bucket/runs/r1"
    server, options = spec["engine"]["server"], spec["engine"]["options"]
    assert server["in_place_reload"] is True and options["expert_file_update"] == "auto"
    assert options["expert_update_probe"] is True
    assert server["extra_args"][-2:] == ["--ep-size", "4"]


def test_restart_reload_and_defaults(files):
    c = mes.SamplerConfig.from_env(_env(files, RELOAD="restart"))
    spec = mes.build_spec(c, out="/vol/x")
    assert spec["engine"]["server"]["in_place_reload"] is False
    assert spec["engine"]["options"]["expert_file_update"] == "off"
    assert spec["drop_zero_variance_groups"] is False and "data_slot" not in spec["prompts"]
    assert spec["sampling"]["n"] == 4 and spec["sampling"]["max_tokens"] == 1024


def test_flags_override_env_and_bad_values_are_refused(files):
    c = mes.SamplerConfig.from_env(_env(files), shard="other", data_slot="2/3")
    assert c.shard == "other" and c.data_slot == "2/3"
    with pytest.raises(ValueError):
        mes.SamplerConfig.from_env(_env(files, DATA_SLOT="3/3"))
    with pytest.raises(ValueError):
        mes.SamplerConfig.from_env(_env(files, RELOAD="cold"))
    with pytest.raises(ValueError, match="max_model_len"):
        mes.SamplerConfig.from_env(_env(files, MAX_TOKENS="4000"))


def test_store_env_from_either_secret_style(files):
    c = mes.SamplerConfig.from_env(_env(files))
    yeto_named = mes.store_env(c, {"YETO_RL_STORE_S3_ACCESS_KEY_ID": "k", "YETO_RL_STORE_S3_ENDPOINT_URL": "https://e"})
    assert yeto_named["YETO_RL_STORE_S3_ENDPOINT_URL"] == "https://e"
    assert "YETO_RL_STORE_S3_CREDENTIALS" not in yeto_named
    assert yeto_named["YETO_RL_STORE_S3_URI"] == "s3://bucket/runs/r1"
    aws = mes.store_env(c, {"AWS_ACCESS_KEY_ID": "k", "AWS_ENDPOINT_URL": "https://a"})
    assert aws["YETO_RL_STORE_S3_CREDENTIALS"] == "aws-env" and aws["YETO_RL_STORE_S3_REGION"] == "auto"
    with pytest.raises(ValueError, match="endpoint"):
        mes.store_env(c, {"AWS_ACCESS_KEY_ID": "k"})


def test_record_round_trip(files):
    c = mes.SamplerConfig.from_env(_env(files, EXTRA_SHARDS="a,b"))
    assert mes.SamplerConfig.from_record(json.loads(json.dumps(c.record()))) == c
    assert c.trainer_buffer_shards == "a,b,modal-s0"


def test_cli_dry_run(files, monkeypatch, capsys):
    for k, v in _env(files).items():
        monkeypatch.setenv(k, v)
    assert cli.main(["engine-sampler-modal", "--dry-run", "--data-slot", "0/3"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["prompt_rows"] == 10 and report["data_slot"] == "0/3" and report["problems"] == []
    monkeypatch.setenv("GROUPS", "11")  # more groups per round than the slot's rows
    assert cli.main(["engine-sampler-modal", "--data-slot", "0/3"]) == 1


def test_no_default_image_bucket_or_secret(files):
    for drop, match in (("SAMPLER_IMAGE", "no default image"), ("STORE_BUCKET", "bucket"),
                        ("SAMPLER_SECRET", "Modal Secret")):
        env = _env(files)
        del env[drop]
        with pytest.raises(ValueError, match=match):
            mes.SamplerConfig.from_env(env)

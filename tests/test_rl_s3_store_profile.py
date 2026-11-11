"""The s3 store's client settings from a standard AWS setup: AWS_ENDPOINT_URL, a named
profile's endpoint_url / region / keys, the --endpoint-url / --profile CLI flags, and
error messages that name what to set. CPU only; fake profile files, no network."""
from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError, NoCredentialsError

from yeto.cloud import s3_store
from yeto.launch.storage import store_client_environ
from yeto.rl.engine.object_store import ObjectStoreError

R2 = "https://0123abcd.r2.cloudflarestorage.com"


@pytest.fixture
def aws_files(tmp_path, monkeypatch):
    config = tmp_path / "config"
    creds = tmp_path / "credentials"
    config.write_text(f"[profile r2]\nregion = auto\nendpoint_url = {R2}\n"
                      "[profile nested]\ns3 =\n  endpoint_url = https://minio.example:9000\n"
                      "[profile plain]\nregion = eu-north-1\n")
    creds.write_text("[r2]\naws_access_key_id = AKIDPROFILE\naws_secret_access_key = profile-secret\n")
    for name in ("AWS_PROFILE", "AWS_ENDPOINT_URL", "AWS_ENDPOINT_URL_S3", "AWS_REGION", "AWS_DEFAULT_REGION",
                 "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", *s3_store.CREDENTIAL_ENVS,
                 s3_store.ENDPOINT_ENV, s3_store.REGION_ENV, s3_store.PROFILE_ENV):
        monkeypatch.delenv(name, raising=False)
    base = {"AWS_CONFIG_FILE": str(config), "AWS_SHARED_CREDENTIALS_FILE": str(creds)}
    for k, v in base.items():
        monkeypatch.setenv(k, v)
    s3_store._profile_config_cached.cache_clear()
    yield base
    s3_store._profile_config_cached.cache_clear()


def test_aws_endpoint_url_alone_reaches_r2_with_region_auto(aws_files):
    env = {**aws_files, "AWS_ENDPOINT_URL": R2}
    assert s3_store.store_endpoint(env) == R2 and s3_store.store_region(env) == "auto"
    assert s3_store.store_endpoint({**aws_files, "AWS_ENDPOINT_URL_S3": R2}) == R2


def test_a_profile_supplies_endpoint_region_and_keys(aws_files):
    for env in ({**aws_files, s3_store.PROFILE_ENV: "r2"}, {**aws_files, "AWS_PROFILE": "r2"}):
        assert s3_store.store_endpoint(env) == R2 and s3_store.store_region(env) == "auto"
    assert s3_store.store_endpoint({**aws_files, "AWS_PROFILE": "nested"}) == "https://minio.example:9000"
    plain = {**aws_files, "AWS_PROFILE": "plain"}
    assert s3_store.store_endpoint(plain) is None and s3_store.store_region(plain) == "eu-north-1"
    # an explicit variable wins over the profile
    env = {**aws_files, "AWS_PROFILE": "r2", s3_store.ENDPOINT_ENV: "https://other", s3_store.REGION_ENV: "x"}
    assert s3_store.store_endpoint(env) == "https://other" and s3_store.store_region(env) == "x"


def test_the_client_uses_the_profile(aws_files):
    client = s3_store._client({**aws_files, s3_store.PROFILE_ENV: "r2"}, require_credentials=True)
    assert client.meta.endpoint_url == R2 and client.meta.region_name == "auto"
    assert client._request_signer._credentials.access_key == "AKIDPROFILE"


def test_part_checksums_auto_see_the_profiles_r2_endpoint(aws_files):
    assert s3_store.TransferSettings.from_env({**aws_files, "AWS_PROFILE": "r2"}).part_checksum is False


def test_a_missing_profile_is_named(aws_files):
    with pytest.raises(ObjectStoreError, match="profile 'nope'.*not configured"):
        s3_store._client({**aws_files, s3_store.PROFILE_ENV: "nope"}, require_credentials=False)
    # AWS_PROFILE of another host (an island) is not an error when the store keys are set
    env = {**aws_files, "AWS_PROFILE": "nope", s3_store.KEY_ID_ENV: "AKID", s3_store.SECRET_ENV: "s"}
    assert s3_store._client(env, require_credentials=True) is not None


def test_no_credentials_names_every_way_to_give_them(aws_files):
    with pytest.raises(ObjectStoreError) as info:
        s3_store._client(dict(aws_files), require_credentials=True)
    for name in (s3_store.KEY_ID_ENV, s3_store.SECRET_ENV, s3_store.PROFILE_ENV, "--profile"):
        assert name in str(info.value)


def test_island_envs_ship_the_profiles_key_endpoint_and_region(aws_files):
    env = {**aws_files, s3_store.PROFILE_ENV: "r2"}
    out = s3_store.island_envs("s3://bkt/run", env)
    assert out[s3_store.KEY_ID_ENV] == "AKIDPROFILE" and out[s3_store.SECRET_ENV] == "profile-secret"
    assert out[s3_store.ENDPOINT_ENV] == R2 and out[s3_store.REGION_ENV] == "auto"
    assert s3_store.PROFILE_ENV not in out  # the island has no profile
    with pytest.raises(ValueError, match=s3_store.PROFILE_ENV):
        s3_store.island_envs("s3://bkt/run", dict(aws_files))


def test_launch_profile_flag_gives_the_endpoint(aws_files):
    env = store_client_environ(SimpleNamespace(rl_object_store_endpoint=None, rl_object_store_profile="r2"),
                               dict(aws_files))
    assert env[s3_store.ENDPOINT_ENV] == R2 and env[s3_store.REGION_ENV] == "auto"
    assert env[s3_store.PROFILE_ENV] == "r2"


def test_cli_flags_set_the_environment():
    parser = argparse.ArgumentParser()
    s3_store.add_store_client_arguments(parser)
    env: dict[str, str] = {}
    s3_store.apply_store_client_arguments(parser.parse_args(["--endpoint-url", R2, "--profile", "r2"]), env)
    assert env == {s3_store.ENDPOINT_ENV: R2, s3_store.PROFILE_ENV: "r2"}


def _client_error(code):
    return ClientError({"Error": {"Code": code, "Message": code}}, "HeadObject")


def test_error_messages_name_what_to_set():
    msg = s3_store.describe_store_error(NoCredentialsError(), "s3://b", {})
    assert s3_store.KEY_ID_ENV in msg and s3_store.PROFILE_ENV in msg
    msg = s3_store.describe_store_error(_client_error("InvalidAccessKeyId"), "s3://b", {})
    assert "AWS S3 (no endpoint set)" in msg and "AWS_ENDPOINT_URL" in msg and "--endpoint-url" in msg
    msg = s3_store.describe_store_error(_client_error("SignatureDoesNotMatch"), "s3://b",
                                        {s3_store.ENDPOINT_ENV: R2})
    assert R2 in msg and s3_store.SECRET_ENV in msg and "AWS_ENDPOINT_URL" not in msg
    msg = s3_store.describe_store_error(EndpointConnectionError(endpoint_url=R2), "s3://b",
                                        {s3_store.ENDPOINT_ENV: R2})
    assert "cannot reach" in msg and s3_store.ENDPOINT_ENV in msg
    msg = s3_store.describe_store_error(_client_error("AuthorizationHeaderMalformed"), "s3://b",
                                        {s3_store.ENDPOINT_ENV: R2})
    assert f"{s3_store.REGION_ENV}=auto" in msg
    assert s3_store.describe_store_error(_client_error("NoSuchKey"), "s3://b", {}) is None


def test_store_dispense_status_reports_a_store_error_with_its_hint(monkeypatch, capsys):
    from yeto.rl.engine import store_dispense

    def refuse(store):
        raise _client_error("InvalidAccessKeyId")

    monkeypatch.setattr(store_dispense, "_objects_for", refuse)
    monkeypatch.delenv(s3_store.ENDPOINT_ENV, raising=False)
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.delenv(s3_store.PROFILE_ENV, raising=False)  # main() sets it; restored after
    assert store_dispense.main(["status", "--store", "s3://b/run", "--profile", "x", "--endpoint-url", R2]) == 2
    err = capsys.readouterr().err
    assert "refused the credentials" in err and R2 in err

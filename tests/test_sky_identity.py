"""yeto's sky launches do not copy this machine's cloud credential files onto nodes
(yeto.launch.sky_identity): sky's remote_identity is NO_UPLOAD for every cloud but the
node's own non-AWS cloud, and the exceptions say why."""

from __future__ import annotations

import io
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from yeto import island_credential_guard as guard
from yeto.launch import sky_identity as si

REPO = Path(__file__).resolve().parents[1]
CLOUDS = ("aws", "azure", "gcp", "kubernetes", "nebius", "oci", "slurm", "ssh", "vast")


def _task(*clouds, mounts=None):
    resources = [SimpleNamespace(cloud=c) for c in clouds]
    return SimpleNamespace(resources=resources, storage_mounts=mounts or {})


def _config(task, environ=None, configured=(), lines=None):
    return si.identity_config(task, "c1", environ or {}, log=(lines.append if lines is not None else lambda _: None),
                              configured=lambda cloud: cloud in configured, clouds_with_identity=CLOUDS)


def _no_upload(clouds):
    return {c: {"remote_identity": si.NO_UPLOAD} for c in clouds}


def test_an_aws_node_gets_no_credential_file_of_any_cloud():
    assert _config(_task("AWS")) == _no_upload(CLOUDS)


def test_a_non_aws_node_keeps_only_its_own_cloud_file():
    """sky's autodown on a Nebius node calls Nebius with that file; AWS's goes nowhere."""
    assert _config(_task("Nebius")) == _no_upload(c for c in CLOUDS if c != "nebius")
    assert _config(_task("Verda")) == _no_upload(CLOUDS)  # no remote_identity in sky's config schema for Verda


def test_a_task_open_to_any_cloud_leaves_out_aws_only():
    assert _config(SimpleNamespace(resources=[SimpleNamespace(cloud=None)])) == _no_upload(["aws"])
    assert _config(SimpleNamespace()) == _no_upload(["aws"])


def test_a_configured_remote_identity_is_left_to_the_sky_config():
    config = _config(_task("AWS"), configured=("aws", "gcp"))
    assert "aws" not in config and "gcp" not in config and config["nebius"] == {"remote_identity": si.NO_UPLOAD}


@pytest.mark.parametrize("storage", [
    SimpleNamespace(source="s3://b/p", stores={}),
    SimpleNamespace(source=["s3://b/p"], stores={}),
    SimpleNamespace(source=None, stores={"S3": object()}),
    SimpleNamespace(source=None, stores={__import__("enum").Enum("StoreType", {"S3": "S3"}).S3: object()}),
])
def test_an_s3_mount_on_a_non_aws_node_keeps_the_aws_file_and_says_why(storage):
    lines: list[str] = []
    config = _config(_task("Nebius", mounts={"/ckpt": storage}), lines=lines)
    assert "aws" not in config and config["gcp"] == {"remote_identity": si.NO_UPLOAD}
    (line,) = lines
    assert "WARNING" in line and "/ckpt" in line and "~/.aws/credentials" in line and "nebius" in line


def test_an_s3_mount_on_an_aws_node_uses_the_instance_profile():
    lines: list[str] = []
    assert _config(_task("AWS", mounts={"/ckpt": SimpleNamespace(source="s3://b", stores={})}),
                   lines=lines) == _no_upload(CLOUDS)
    assert lines == []


def test_an_s3_mount_on_a_node_that_may_land_off_aws_keeps_the_aws_file():
    lines: list[str] = []
    mounts = {"/ckpt": SimpleNamespace(source="s3://b", stores={})}
    assert "aws" not in _config(_task("AWS", "Nebius", mounts=mounts), lines=lines)
    assert lines and "/ckpt" in lines[0]


def test_a_sourceless_mount_off_aws_keeps_the_aws_file():
    """sky may place a mount without a source on S3 (when the node cloud's storage is
    not enabled on the API server): the mount must not lose its credential."""
    spot_volume = SimpleNamespace(source=None, stores={})
    for cloud in ("Nebius", "Verda"):
        lines: list[str] = []
        assert "aws" not in _config(_task(cloud, mounts={"/v": spot_volume}), lines=lines)
        assert lines and "/v" in lines[0]
    assert _config(_task("AWS", mounts={"/v": spot_volume})) == _no_upload(CLOUDS)
    other = SimpleNamespace(source="gs://b", stores={})
    assert "aws" in _config(_task("Nebius", mounts={"/g": other}))
    assert "aws" in _config(_task("Nebius", mounts={"/n": SimpleNamespace(source=None, stores={"NEBIUS": 1})}))


def test_the_opt_out_restores_sky_default_and_warns():
    lines: list[str] = []
    assert _config(_task("AWS"), environ={si.UPLOAD_ENV: "1"}, lines=lines) == {}
    assert "WARNING" in lines[0] and si.UPLOAD_ENV in lines[0]
    assert _config(_task("AWS"), environ={si.UPLOAD_ENV: "0"}) == _no_upload(CLOUDS)


def test_an_instance_profile_applies_only_to_a_node_that_can_only_be_aws():
    env = {si.AWS_INSTANCE_PROFILE_ENV: "yeto-island-min"}
    assert _config(_task("AWS"), environ=env)["aws"] == {"remote_identity": "yeto-island-min"}
    lines: list[str] = []
    assert _config(_task("AWS", "Nebius"), environ=env, lines=lines)["aws"] == {"remote_identity": si.NO_UPLOAD}
    assert lines and si.AWS_INSTANCE_PROFILE_ENV in lines[0]
    with pytest.raises(ValueError, match=si.AWS_INSTANCE_PROFILE_ENV):
        _config(_task("AWS"), environ={si.AWS_INSTANCE_PROFILE_ENV: "bad name; rm"})


def test_sky_config_accepts_the_override_and_the_request_carries_it(monkeypatch):
    """Against the installed sky: the clouds whose schema takes remote_identity, a valid
    override, and the client config a request snapshots while it is held (and not after)."""
    pytest.importorskip("sky")
    from sky.server.requests import payloads
    from sky.utils import annotations

    clouds = si.identity_clouds()
    assert "aws" in clouds and "nebius" in clouds and "verda" not in clouds
    monkeypatch.setattr(annotations, "is_on_api_server", False)  # what sky's client_api sets
    monkeypatch.setattr(si, "_configured", lambda cloud: False)
    before = payloads.get_override_skypilot_config_from_client()
    with si.launch_identity(_task("AWS"), "c1", {}, log=lambda _: None) as config:
        assert set(config) == set(clouds)
        snap = payloads.get_override_skypilot_config_from_client()
        assert {c: snap[c]["remote_identity"] for c in clouds} == dict.fromkeys(clouds, si.NO_UPLOAD)
    assert payloads.get_override_skypilot_config_from_client() == before


def test_submit_island_launch_holds_the_override_during_the_submit(monkeypatch):
    pytest.importorskip("sky")
    from sky import skypilot_config

    from yeto.launch import sky_ops

    seen = {}

    def launch(task, cluster_name, retry_until_up):
        seen["aws"] = skypilot_config.get_nested(("aws", "remote_identity"), None)
        return "rid"

    monkeypatch.setattr(si, "_configured", lambda cloud: False)
    before = skypilot_config.get_nested(("aws", "remote_identity"), None)
    args = SimpleNamespace(training_mode="sft", retry_until_up=False)
    assert sky_ops.submit_island_launch(SimpleNamespace(launch=launch), args,
                                        SimpleNamespace(cloud="aws", region=None), _task("AWS"), "c1") == "rid"
    assert seen["aws"] == si.NO_UPLOAD
    assert skypilot_config.get_nested(("aws", "remote_identity"), None) == before


def test_every_sky_launch_in_yeto_holds_the_identity_override():
    """A new ``sky.launch`` call must go through launch_identity (the fleet path's
    default is no upload)."""
    found = []
    for path in sorted((REPO / "yeto").rglob("*.py")):
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if re.search(r"\bsky\.launch\(", line) and not line.lstrip().startswith(("#", '"', "'")):
                found.append(path.relative_to(REPO).as_posix())
                window = "\n".join(lines[max(0, i - 3):i])
                assert "launch_identity(" in window, f"{path.relative_to(REPO)}:{i + 1}: {line.strip()}"
    assert len(found) >= 7, found


def test_the_island_warning_names_the_risk_and_never_the_value(tmp_path):
    (tmp_path / ".aws").mkdir()
    (tmp_path / ".aws" / "credentials").write_text("[default]\naws_secret_access_key = zzz\n")
    out = io.StringIO()
    guard.check_island_credentials({}, home=tmp_path, stream=out)
    text = out.getvalue()
    assert "cloud credential files present on this island: ~/.aws/credentials" in text
    assert "Anyone who can run code on this island" in text and "NO_UPLOAD" in text and "zzz" not in text


def test_a_test_double_in_place_of_sky_gets_no_override(monkeypatch):
    import sys
    import types

    monkeypatch.setitem(sys.modules, "sky", types.ModuleType("sky"))
    with si.launch_identity(_task("AWS"), "c1", {}) as config:
        assert config == {}


def test_overlapping_submits_from_two_threads_leave_no_override_behind(monkeypatch):
    """sky's client config override is one per process: A (an AWS node: gcp and aws left
    out) and B (a GCP node: its own gcp file kept) overlap, A leaves first. Each submit
    sees exactly its own override, and the process config is the one before both."""
    import threading
    import time

    pytest.importorskip("sky")
    from sky import skypilot_config

    monkeypatch.setattr(si, "_configured", lambda cloud: False)

    def now(cloud):
        return skypilot_config.get_nested((cloud, "remote_identity"), None)

    before = {c: now(c) for c in ("aws", "gcp")}
    a_in, b_try, a_out = threading.Event(), threading.Event(), threading.Event()
    seen: dict[str, dict] = {}
    errors: list[BaseException] = []

    def a():
        try:
            with si.launch_identity(_task("AWS"), "a", {}, log=lambda _: None):
                a_in.set()
                b_try.wait(5)
                time.sleep(0.3)  # B is trying to enter now
                seen["a"] = {c: now(c) for c in ("aws", "gcp")}
        except BaseException as error:  # noqa: BLE001
            errors.append(error)
        finally:
            a_out.set()

    def b():
        try:
            a_in.wait(5)
            b_try.set()
            with si.launch_identity(_task("GCP"), "b", {}, log=lambda _: None):
                seen["b_entered_after_a"] = a_out.is_set()
                a_out.wait(5)  # without the lock: B still inside after A left
                seen["b"] = {c: now(c) for c in ("aws", "gcp")}
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=a), threading.Thread(target=b)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(15)
    assert not errors and not any(t.is_alive() for t in threads)
    assert seen["b_entered_after_a"]
    assert seen["a"] == {"aws": si.NO_UPLOAD, "gcp": si.NO_UPLOAD}
    assert seen["b"] == {"aws": si.NO_UPLOAD, "gcp": before["gcp"]}
    assert {c: now(c) for c in ("aws", "gcp")} == before


def test_every_client_config_override_in_yeto_is_taken_under_the_submit_lock():
    sites = []
    for path in sorted((REPO / "yeto").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"override_skypilot_config\(", text):
            sites.append(path.relative_to(REPO).as_posix())
            head = text[:match.start()]
            block = head[head.rfind("\ndef "):]
            assert "with SUBMIT_LOCK:" in block, f"{path.relative_to(REPO)}: override outside SUBMIT_LOCK"
    assert sorted(sites) == ["yeto/launch/sky_identity.py", "yeto/launch/sky_ops.py"]

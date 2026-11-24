"""``--modal-store-secret`` per island: in a launch (an island fleet) that mixes Modal
islands with sky or SSH islands, a Modal island attaches the named Modal Secret and
reads the s3 store credential from it, every other island gets the launcher's store
credential (a sky secret, as without the flag). No island gets both, and no key
reaches an argv, a run script or the dry-run output. A launch without any Modal
island still refuses the flag. CPU, fakes; no cloud resource."""

from __future__ import annotations

import json
import shlex
from types import SimpleNamespace

import pytest

from test_rl_algorithm_provenance import _fake_sky, _no_modal_listing  # noqa: F401
from tests.test_run_up_fleet import TRAINER_ARGV, _cfg, _main_args, _write, files, no_subprocess  # noqa: F401
from yeto.cloud import s3_store
from yeto.launch import storage

URI = "s3://b/runs/r"
KEY_ID, SECRET = "AKIDSTOREFAKE", "store-secret-value-not-to-be-seen"
SECRET_NAME = "r2-store"
MIXED_GPU = "modal:1x2xh100,aws:1x8xh100@us-east-1"


@pytest.fixture
def launcher_key(monkeypatch):
    """The launcher's store credential (fake), and none of the AWS-named variables."""
    for name in (*s3_store.CREDENTIAL_ENVS, *s3_store.SETTING_ENVS, s3_store.CREDENTIALS_ENV,
                 "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_ENDPOINT_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(s3_store.KEY_ID_ENV, KEY_ID)
    monkeypatch.setenv(s3_store.SECRET_ENV, SECRET)


def _mixed_fleet(files):
    cfg = _cfg(files)
    cfg["islands"][1]["trainer"]["gpu"] = "aws:1x8xh100@us-east-1"
    cfg["fleet"]["trainer_argv"] = [*TRAINER_ARGV, "--reward-function", "pkg.reward:score", "--trust-remote-code",
                                    "--rl-max-policy-age", "2", "--rl-parameter-layout-sha256", "e" * 64,
                                    "--modal-store-secret", SECRET_NAME]
    return cfg


def _launch_args(argv):
    """The fleet's ``yeto launch`` argv, parsed and prepared as ``yeto launch`` does."""
    from yeto.cli import parse_args
    from yeto.launcher import _prepare_rl_args

    tail = argv[argv.index("launch") + 1:]
    args = parse_args([a for a in tail if a != "--dry-run"])
    args.model_revision, args.data_revision = "a" * 40, "b" * 40
    args.source_sha256, args.reward_sha256 = "c" * 64, "d" * 64
    args.controller = "local"
    _prepare_rl_args(args)
    return args


def test_a_fleet_of_a_modal_and_an_aws_island_dry_runs_with_the_store_secret(
        files, no_subprocess, launcher_key, capsys):
    from yeto.gpu_spec import parse_gpu_spec
    from yeto.launch.args import check_cloud_prerequisites
    from yeto.launch.modal_islands import build_modal_island_config
    from yeto.launcher import _island_task, dry_run_plan, make_miles_island_task
    from yeto.rl import up

    assert up.main(_main_args(_write(files, _mixed_fleet(files)))) == 0
    (launch,) = [a for a in no_subprocess if "launch" in a]
    assert launch[launch.index("--gpu") + 1] == MIXED_GPU
    assert launch.count("--modal-store-secret") == 1  # one launch flag; the islands take what they need
    args = _launch_args(launch)
    specs = parse_gpu_spec(args.gpu)
    check_cloud_prerequisites(specs, args=args, modal_ok=True)  # used to refuse the aws island
    plan = dry_run_plan(args)
    assert [(i["cloud"], i["sample_role"]) for i in plan["island_requests"]] == [
        ("modal", "trainer"), ("aws", "trainer")]
    modal_spec, aws_spec = specs
    # the Modal island: the named Secret, told to read its AWS pair; no launcher key
    modal_task = _island_task(make_miles_island_task, args, modal_spec, 0, 2, "none", {})
    cfg = build_modal_island_config(args, modal_spec, 0, modal_task, "none")
    assert cfg.named_secrets == (SECRET_NAME,)
    assert cfg.envs[s3_store.CREDENTIALS_ENV] == s3_store.CREDENTIALS_AWS_ENV
    assert not any(name in cfg.envs for name in s3_store.CREDENTIAL_ENVS)
    assert SECRET not in cfg.to_json() and KEY_ID not in cfg.to_json()
    # the aws island: the launcher's store key, the secret part as a sky secret; no Secret name
    aws_task = _island_task(make_miles_island_task, args, aws_spec, 1, 2, "none", {})
    assert aws_task.secrets[s3_store.SECRET_ENV] == SECRET
    assert aws_task.envs[s3_store.KEY_ID_ENV] == KEY_ID
    assert s3_store.SECRET_ENV not in aws_task.envs and s3_store.CREDENTIALS_ENV not in aws_task.envs
    assert SECRET_NAME not in json.dumps(aws_task.envs) + json.dumps(aws_task.secrets)
    # no key in any argv, run / setup script, plan or log line
    out = capsys.readouterr()
    seen = "\n".join([shlex.join(launch), *(shlex.join(a) for a in no_subprocess), json.dumps(plan),
                      modal_task.run, modal_task.setup, aws_task.run, aws_task.setup, cfg.run_script,
                      cfg.setup_script or "", out.out, out.err])
    assert SECRET not in seen and KEY_ID not in seen
    assert all("--modal-store-secret" not in (i["learner_command"] or "") for i in plan["island_requests"])


def test_without_the_launcher_key_the_aws_island_of_a_mixed_launch_is_named(monkeypatch):
    from tests.test_launch_auto import _specs

    for name in (*s3_store.CREDENTIAL_ENVS, s3_store.PROFILE_ENV, s3_store.CREDENTIALS_ENV):
        monkeypatch.delenv(name, raising=False)
    args = SimpleNamespace(rl_checkpoint_store=URI, modal_store_secret=SECRET_NAME, rl_sample_role="trainer")
    assert storage.store_object_envs(args, "modal")[s3_store.CREDENTIALS_ENV] == s3_store.CREDENTIALS_AWS_ENV
    with pytest.raises(ValueError, match="store credentials.*--modal-store-secret covers the Modal islands "
                                         "only; the aws island takes the launcher's store credential"):
        storage.store_object_envs(args, "aws")
    modal_spec, aws_spec = _specs(MIXED_GPU)
    assert storage.modal_store_secret_name(args, modal_spec) == SECRET_NAME
    assert storage.modal_store_secret_name(args, aws_spec) is None


@pytest.mark.parametrize("cloud", ["aws", "nebius", "ssh"])
def test_each_island_gets_only_its_own_kind_of_store_credential(cloud, launcher_key):
    args = SimpleNamespace(rl_checkpoint_store=URI, modal_store_secret=SECRET_NAME, rl_sample_role="trainer")
    modal = storage.store_object_envs(args, "modal")
    assert modal[s3_store.CREDENTIALS_ENV] == s3_store.CREDENTIALS_AWS_ENV
    assert not any(name in modal for name in s3_store.CREDENTIAL_ENVS)
    other = storage.store_object_envs(args, cloud)  # forwarded by the launcher, as without the flag
    assert (other[s3_store.KEY_ID_ENV], other[s3_store.SECRET_ENV]) == (KEY_ID, SECRET)
    assert s3_store.CREDENTIALS_ENV not in other
    plain = SimpleNamespace(rl_checkpoint_store=URI, rl_sample_role="trainer")
    assert other == storage.store_object_envs(plain, cloud)


def test_the_store_secret_is_still_refused_when_no_island_is_on_modal():
    from tests.test_launch_auto import _args, _specs
    from yeto.launch.args import check_cloud_prerequisites

    for gpu in ("aws:1xh100@us-east-1", "aws:1xh100@us-east-1,nebius:8xh100@eu-north1"):
        args = _args(["--gpu", gpu, "--modal-store-secret", SECRET_NAME])
        with pytest.raises(ValueError, match="--modal-store-secret attaches a Modal Secret to the Modal islands "
                                             "of a launch; this launch has none"):
            check_cloud_prerequisites(_specs(gpu), args=args, modal_ok=True, project_ids={"eu-north1": "p"})
    args = _args(["--gpu", MIXED_GPU, "--modal-store-secret", "bad name"])
    with pytest.raises(ValueError, match="not a Modal Secret name"):
        check_cloud_prerequisites(_specs(MIXED_GPU), args=args, modal_ok=True)

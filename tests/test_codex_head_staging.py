"""S17 M1: a signed Codex run under --controller head stages its bundle and harness env on the head."""

from types import SimpleNamespace

from yeto import launcher


def test_non_codex_run_stages_nothing(monkeypatch):
    monkeypatch.setattr(launcher, "codex_harness_launch", lambda args, environ=None: None)
    assert launcher.codex_head_staging(SimpleNamespace(), {"TBENCH_REWARD_HMAC_KEY": "k"}) == ({}, {})


def test_codex_run_mounts_bundle_and_forwards_harness_env_as_secrets(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "codex_harness_launch", lambda args, environ=None: ("", {}, {}))
    env = {
        launcher.CODEX_BUNDLE_DIR_ENV: str(tmp_path),
        launcher.HARNESS_ENVIRONMENT_PROVIDER_ENV: "yeto.rl.harness.codex.tb2_provider:modal_provider",
        "TBENCH_REWARD_HMAC_KEY": "secret",
        "MODAL_TOKEN_ID": "id",
        "MODAL_TOKEN_SECRET": "sec",
        "YETO_SANDBOX_MODAL_TOKEN_ID": "sid",
        "YETO_SANDBOX_MODAL_TOKEN_SECRET": "ssec",
        "OPENENV_RUN_ID": "run",
        "YETO_HARNESS_TB2_TASKS_DIR": "/opt/yeto/codex/tb2-tasks",
        "YETO_HARNESS_TB2_MODAL_APP": "yeto-tbench2",
        "UNRELATED": "x",
        "SECRLENV_MAX_TURNS": "",
    }
    mounts, secrets = launcher.codex_head_staging(SimpleNamespace(), env)
    assert mounts == {launcher.HEAD_CODEX_BUNDLE_PATH: str(tmp_path.resolve())}
    assert secrets == {k: env[k] for k in (
        launcher.HARNESS_ENVIRONMENT_PROVIDER_ENV, "TBENCH_REWARD_HMAC_KEY",
        "YETO_SANDBOX_MODAL_TOKEN_ID", "YETO_SANDBOX_MODAL_TOKEN_SECRET", "OPENENV_RUN_ID", "YETO_HARNESS_TB2_TASKS_DIR", "YETO_HARNESS_TB2_MODAL_APP")}


def test_args_bundle_dir_wins_over_env(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "codex_harness_launch", lambda args, environ=None: ("", {}, {}))
    mounts, _ = launcher.codex_head_staging(SimpleNamespace(codex_bundle_dir=str(tmp_path)),
                                            {launcher.CODEX_BUNDLE_DIR_ENV: "/nope"})
    assert list(mounts.values()) == [str(tmp_path.resolve())]

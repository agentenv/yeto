"""Bug 26: an island image whose ``docker pull`` fails the same way on every machine (a layer
the daemon cannot register, a manifest the registry lacks, a refused reference) ends the
launch as a configuration error, not an infrastructure failure relaunched onto a new
machine each time; a timeout, a rate limit or a registry 5xx keeps its relaunches."""

from __future__ import annotations

import sys
import types

import pytest

import yeto.launcher as launcher
from test_rl_algorithm_provenance import _fake_sky, _launcher_args  # noqa: F401  (fixture)
from test_rl_launcher_island_failure import Clock, _no_modal_listing, _setup  # noqa: F401  (fixture)
from yeto.config_errors import CONFIG_ERROR_EXIT, CONFIG_ERROR_MARKER
from yeto.launch import island_lifecycle as lifecycle
from yeto.launch.image_pull import (CREDENTIAL, IMAGE, ImagePullRefused, PullFailure, deterministic_pull_error,
                                    image_error_message, pull_failure)

DIGEST = "ghcr.io/x/yeto-miles@sha256:" + "0" * 64
# r8 (Nebius H200): sky's error carries the pull's output after its first line
R8 = (f"Command sudo docker pull {DIGEST[:60]}... failed with return code 1.\n"
      "Failed to run docker setup commands.\n"
      f"{DIGEST}: Pulling from x/yeto-miles\n"
      "689b91d88a0f: Pulling fs layer\n"
      "2e676d178d2f: Pull complete\n"
      "failed to register layer: openat opt/sglang/lib/python3.12/site-packages/"
      "sglang-0.5.24.dev80+g35708f7.dist-info/INSTALLER: not a directory\n"
      "I1126 04:08:12.519258 52022726 ev_poll_posix.cc:588] FD from fork parent still in poll list: fd(43)\n")


def _pull(line: str) -> str:
    return f"Command sudo docker pull {DIGEST} failed with return code 1.\n{DIGEST}: Pulling from x\n{line}\n"


@pytest.mark.parametrize("line, kind", [
    (f"Error response from daemon: manifest for {DIGEST} not found: manifest unknown: manifest unknown", IMAGE),
    ("Error response from daemon: no matching manifest for linux/arm64/v8 in the manifest list entries", IMAGE),
    ("Error response from daemon: Head \"https://ghcr.io/v2/x/manifests/sha256:0\": unauthorized", CREDENTIAL),
    ("Error response from daemon: pull access denied for x/yeto-miles, repository does not exist or may require "
     "'docker login': denied: requested access to the resource is denied", CREDENTIAL),
    ("Error response from daemon: Head \"https://1.dkr.ecr.us-east-1.amazonaws.com/v2/x/manifests/0\": "
     "no basic auth credentials", CREDENTIAL),
    ("Error response from daemon: denied: Your authorization token has expired. Reauthenticate and try again.",
     CREDENTIAL),
])
def test_an_image_or_credential_failure_on_dockers_own_line_is_deterministic(line, kind):
    assert pull_failure(_pull(line)) == PullFailure(kind, line)
    assert pull_failure(R8).kind == IMAGE and "failed to register layer" in pull_failure(R8).line


@pytest.mark.parametrize("line", [
    "Error response from daemon: Get \"https://ghcr.io/v2/\": net/http: TLS handshake timeout",
    "Error response from daemon: toomanyrequests: retry-after: 39.1us, allowed: 44000/minute",
    "Error response from daemon: received unexpected HTTP status: 503 Service Unavailable",
    "error pulling image configuration: download failed after attempts=6: unexpected EOF",
    # the machine's own disk, not the image: another machine may pull it
    "failed to register layer: write /var/lib/docker/x: no space left on device",
    "failed to register layer: openat /var/lib/docker/overlay2/x/diff/opt: input/output error",
    "failed to register layer: mkdir /var/lib/docker/overlay2/x: read-only file system",
])
def test_a_network_registry_or_host_failure_keeps_its_relaunches(line):
    assert pull_failure(_pull(line)) is None


@pytest.mark.parametrize("other", [
    "401 Client Error: Unauthorized for url: https://huggingface.co/api/models/x",
    "fatal: Authentication required for 'https://github.com/x/y.git/'",
    "remote: denied: requested access to the resource is denied",
])
def test_another_tools_line_after_a_pull_failure_keeps_its_relaunches(other):
    """Only docker's own error lines count: a failed pull (its own error a timeout here, or
    none) followed by a Hugging Face 401 or a git refusal is not an image problem."""
    assert pull_failure(_pull("Error response from daemon: Get \"https://ghcr.io/v2/\": i/o timeout") + other) is None
    assert pull_failure(_pull("2e676d178d2f: Pull complete") + other) is None


def test_only_an_image_pull_counts():
    """An "unauthorized" outside a pull (an API, ssh) says nothing about the image."""
    assert deterministic_pull_error("Error response from daemon: unauthorized") is None
    assert deterministic_pull_error(None) is None


def test_the_credential_error_says_check_the_registry_login():
    message = image_error_message("r-i0-l0", PullFailure(CREDENTIAL, "Error response from daemon: unauthorized"))
    assert "credential or image permission problem: check the registry login" in message
    assert "ECR password lasts 12 hours" in message


def test_the_launch_that_cannot_pull_its_image_ends_as_a_configuration_error(monkeypatch, tmp_path, capsys):
    """r8: the moved island's machine could not register a layer of the Miles image; the
    launch failed as infrastructure and the trainer worker ordered a new H200 for each of
    3 relaunches. Now the launch exits 10 with the configuration error line, which the
    trainer worker's classification reads as not relaunched."""
    record, clock = [], Clock()
    _setup(monkeypatch, tmp_path, failing=set(), clock=clock, record=record)
    sky = sys.modules["sky"]

    def stream_and_get(rid):
        if "-l0" in str(rid):
            raise RuntimeError(R8)
        return 1, types.SimpleNamespace(head_ip="10.0.0.1")

    sky.stream_and_get = stream_and_get
    args = _launcher_args("ports", ("--controller", "local", "--rl-image", "docker:" + DIGEST),
                          gpu="aws:1xa100@us-east-1,aws:1xa100@us-east-1")
    args.keep = False
    assert launcher.run(args) == CONFIG_ERROR_EXIT
    err = capsys.readouterr().err
    marker = next(line for line in err.splitlines() if CONFIG_ERROR_MARKER in line)
    assert "cannot be pulled" in marker and "failed to register layer" in marker
    names = launcher.learner_cluster_names(args.cluster_prefix, launcher.parse_gpu_spec(args.gpu))
    # the rest is torn down here; the failed cluster is in the resource ledger, which the
    # CLI's backstop releases, as for any launch that fails to provision
    assert ("terminate", names[1]) in record and ("terminate", f"{args.cluster_prefix}-syncer") in record
    decision = lifecycle.classify(exit_code=CONFIG_ERROR_EXIT, log_text=err[-4000:], detail="")
    assert decision.cause == lifecycle.CONFIG_ERROR and not decision.relaunch


def test_a_pull_that_timed_out_still_fails_the_launch_as_before(monkeypatch, tmp_path):
    record, clock = [], Clock()
    _setup(monkeypatch, tmp_path, failing=set(), clock=clock, record=record)
    sky = sys.modules["sky"]

    def stream_and_get(rid):
        if "-l0" in str(rid):
            raise RuntimeError(_pull("net/http: TLS handshake timeout"))
        return 1, types.SimpleNamespace(head_ip="10.0.0.1")

    sky.stream_and_get = stream_and_get
    args = _launcher_args("ports", ("--controller", "local", "--rl-image", "docker:" + DIGEST),
                          gpu="aws:1xa100@us-east-1,aws:1xa100@us-east-1")
    args.keep = False
    with pytest.raises(RuntimeError, match="failed to provision"):
        launcher.run(args)


def test_a_relaunch_that_cannot_pull_the_image_ends_the_run_once(monkeypatch, tmp_path, capsys):
    """An island whose job failed is relaunched by the fleet controller: a relaunch whose
    pull fails deterministically is not tried again until the recovery timeout."""
    record, clock = [], Clock()
    _setup(monkeypatch, tmp_path, failing={0}, clock=clock, record=record)

    def relaunch(self, task, cluster):
        record.append(("relaunch", cluster))
        raise ImagePullRefused(f"the island image cannot be pulled on {cluster}: failed to register layer")

    monkeypatch.setattr(launcher.SkySDKOps, "relaunch", relaunch)
    args = _launcher_args("ports", ("--controller", "local", "--rl-image", "docker:" + DIGEST),
                          gpu="aws:1xa100@us-east-1,aws:1xa100@us-east-1")
    args.keep, args.recover_timeout, args.controller_poll = False, 600.0, 30.0
    assert launcher.run(args) == CONFIG_ERROR_EXIT
    assert len([r for r in record if r[0] == "relaunch"]) == 1
    assert "cannot be pulled" in capsys.readouterr().err


class _Task:
    def __init__(self, secrets):
        self.secrets, self.updated = dict(secrets), []

    def update_secrets(self, secrets):
        self.updated.append(dict(secrets))
        self.secrets.update(secrets)


def _sky(monkeypatch, errors, launched=None):
    sky = types.ModuleType("sky")

    def launch(task, cluster_name):
        error = next(errors)
        if error is None:
            if launched is not None:
                launched.append(dict(task.secrets))
            return (7, None)
        raise error

    sky.launch, sky.get = launch, lambda rid: rid
    monkeypatch.setitem(sys.modules, "sky", sky)
    monkeypatch.setattr("yeto.launch.sky_ops.maybe_install_nebius_reclaim_hook", lambda *a, **k: None)


def test_a_refused_login_on_a_relaunch_is_read_again_and_tried_once_more(monkeypatch):
    """A relaunch hours into a launch reuses the login the launch read at its start (an ECR
    password lasts 12 hours): the registry refusing it gets one more try with the login read
    now; refused again, the island ends as a configuration error naming the login."""
    from yeto.launch.sky_ops import SkySDKOps

    expired = RuntimeError(_pull("Error response from daemon: denied: Your authorization token has expired."))
    monkeypatch.setenv("SKYPILOT_DOCKER_USERNAME", "AWS")
    monkeypatch.setenv("SKYPILOT_DOCKER_PASSWORD", "fresh-token")
    monkeypatch.setenv("SKYPILOT_DOCKER_SERVER", "1.dkr.ecr.us-east-1.amazonaws.com")
    old = {"SKYPILOT_DOCKER_USERNAME": "AWS", "SKYPILOT_DOCKER_PASSWORD": "old-token",
           "SKYPILOT_DOCKER_SERVER": "1.dkr.ecr.us-east-1.amazonaws.com"}
    launched = []
    _sky(monkeypatch, iter([expired, None]), launched)
    ops = SkySDKOps.__new__(SkySDKOps)
    task = _Task(old)
    assert ops.relaunch(task, "r-i0-l0") == 7
    assert len(launched) == 1 and launched[0]["SKYPILOT_DOCKER_PASSWORD"] == "fresh-token"
    assert len(task.updated) == 1

    _sky(monkeypatch, iter([expired, expired]))
    with pytest.raises(ImagePullRefused, match="check the registry login"):
        ops.relaunch(_Task(old), "r-i0-l0")


def test_a_refused_login_that_has_not_changed_is_not_tried_again(monkeypatch, tmp_path):
    """The login came from this launch's SKYPILOT_DOCKER_* variables, which do not change while
    it runs: reading it again gives the refused one, so no second machine is ordered."""
    from yeto.launch.sky_ops import SkySDKOps

    expired = RuntimeError(_pull("Error response from daemon: denied: Your authorization token has expired."))
    same = {"SKYPILOT_DOCKER_USERNAME": "AWS", "SKYPILOT_DOCKER_PASSWORD": "old-token",
            "SKYPILOT_DOCKER_SERVER": "1.dkr.ecr.us-east-1.amazonaws.com"}
    for key, value in same.items():
        monkeypatch.setenv(key, value)
    calls = []
    errors = iter([expired, None])

    def counted():
        for error in errors:
            calls.append(1)
            yield error

    _sky(monkeypatch, counted())
    task = _Task(same)
    with pytest.raises(ImagePullRefused, match="registry credentials refused, the login is unchanged") as raised:
        SkySDKOps.__new__(SkySDKOps).relaunch(task, "r-i0-l0")
    assert len(calls) == 1 and task.updated == []
    assert "old-token" not in str(raised.value)

    # no login found at all (variables gone, no ~/.docker/config.json entry): not retried either
    for key in same:
        monkeypatch.delenv(key)
    monkeypatch.setenv("HOME", str(tmp_path))
    calls.clear()
    errors = iter([expired, None])
    _sky(monkeypatch, counted())
    task = _Task(same)
    with pytest.raises(ImagePullRefused, match="the login is unchanged"):
        SkySDKOps.__new__(SkySDKOps).relaunch(task, "r-i0-l0")
    assert len(calls) == 1 and task.updated == []


def test_sky_ops_relaunch_names_a_deterministic_pull_and_keeps_the_others(monkeypatch):
    from yeto.launch.sky_ops import SkySDKOps

    sky = types.ModuleType("sky")
    errors = iter([RuntimeError(R8), RuntimeError(_pull("net/http: TLS handshake timeout"))])

    def launch(task, cluster_name):
        raise next(errors)

    sky.launch, sky.get = launch, lambda rid: rid
    monkeypatch.setitem(sys.modules, "sky", sky)
    ops = SkySDKOps.__new__(SkySDKOps)
    with pytest.raises(ImagePullRefused, match="cannot be pulled on r-i0-l0: failed to register layer"):
        ops.relaunch(object(), "r-i0-l0")
    assert ops.relaunch(object(), "r-i0-l0") is None  # a timeout: retried as before

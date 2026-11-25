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
from yeto.launch.image_pull import ImagePullRefused, deterministic_pull_error

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


@pytest.mark.parametrize("line", [
    f"Error response from daemon: manifest for {DIGEST} not found: manifest unknown: manifest unknown",
    "Error response from daemon: Head \"https://ghcr.io/v2/x/manifests/sha256:0\": unauthorized",
    "Error response from daemon: pull access denied for x/yeto-miles, repository does not exist",
    "denied: requested access to the resource is denied",
    "no matching manifest for linux/arm64/v8 in the manifest list entries",
])
def test_an_image_or_reference_failure_is_deterministic(line):
    assert deterministic_pull_error(_pull(line)) == line
    assert "failed to register layer" in deterministic_pull_error(R8)


@pytest.mark.parametrize("line", [
    "Error response from daemon: Get \"https://ghcr.io/v2/\": net/http: TLS handshake timeout",
    "toomanyrequests: retry-after: 39.1µs, allowed: 44000/minute",
    "Error response from daemon: received unexpected HTTP status: 503 Service Unavailable",
    "error pulling image configuration: download failed after attempts=6: unexpected EOF",
    "failed to register layer: write /var/lib/docker/x: no space left on device",
])
def test_a_network_registry_or_host_failure_keeps_its_relaunches(line):
    assert deterministic_pull_error(_pull(line)) is None


def test_only_an_image_pull_counts():
    """An "unauthorized" outside a pull (an API, ssh) says nothing about the image."""
    assert deterministic_pull_error("sky.exceptions.ClusterOwnerIdentityMismatchError: unauthorized") is None
    assert deterministic_pull_error(None) is None


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

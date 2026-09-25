"""terminate_and_verify: down a cluster, then confirm at the cloud level
that no instance survives (sky.down has been seen to report success while a
spot instance lingers)."""

from yeto.launcher import terminate_and_verify


class FakeSky:
    def __init__(self):
        self.downs = 0

    def down(self, cluster):
        self.downs += 1
        return ("rid", cluster)

    def get(self, rid):
        return None


def _no_sleep(_seconds):
    pass


def test_trusts_down_when_no_probe():
    sky = FakeSky()
    # probe=None models a cluster/cloud that can't be cloud-verified.
    assert terminate_and_verify(sky, "c", probe=None, sleep_fn=_no_sleep) is True
    assert sky.downs == 1


def test_without_probe_only_a_clean_or_never_existed_down_counts():
    class Down:
        def __init__(self, exc):
            self.exc = exc

        def __call__(self):
            raise self.exc

    assert terminate_and_verify(
        None, "c", probe=None, down=Down(ValueError("Cluster 'c' does not exist.")), sleep_fn=_no_sleep
    ) is True
    assert terminate_and_verify(
        None, "c", probe=None, down=Down(RuntimeError("API server unreachable")), sleep_fn=_no_sleep
    ) is False


def test_down_hook_replaces_sky_down():
    calls = []
    assert terminate_and_verify(
        None, "c", probe=lambda: [], down=lambda: calls.append(1), sleep_fn=_no_sleep
    ) is True
    assert calls == [1]


def test_confirmed_gone_on_first_check():
    sky = FakeSky()
    assert terminate_and_verify(sky, "c", probe=lambda: [], sleep_fn=_no_sleep) is True
    assert sky.downs == 1  # no retry needed


def test_retries_down_until_cloud_reports_empty():
    sky = FakeSky()
    calls = {"n": 0}

    def probe():
        calls["n"] += 1
        return ["i-abc"] if calls["n"] < 3 else []  # alive twice, then gone

    assert terminate_and_verify(sky, "c", probe=probe, sleep_fn=_no_sleep) is True
    assert sky.downs == 3  # initial + 2 retries


def test_returns_false_when_instance_never_dies():
    sky = FakeSky()
    assert (
        terminate_and_verify(
            sky, "c", probe=lambda: ["i-zombie"], attempts=3, sleep_fn=_no_sleep
        )
        is False
    )
    assert sky.downs == 4  # initial + 3 retries, all failed


def test_probe_error_falls_back_to_trusting_down():
    sky = FakeSky()

    def probe():
        raise RuntimeError("cloud API down")

    assert terminate_and_verify(sky, "c", probe=probe, sleep_fn=_no_sleep) is True


def test_cloud_probe_builds_against_sky_status_version_enum(monkeypatch):
    """sky's StatusVersion defines only ``__ge__``; a ``<`` comparison raised
    and the probe fell back to "trust sky.down" for every cloud, which is how
    a stopped head and orphaned learners went unnoticed."""
    import enum
    import sys
    import types

    from yeto import launcher

    class StatusVersion(enum.Enum):
        CLOUD_CLI = 1
        SKYPILOT = 2

        def __ge__(self, other):
            return self.value >= other.value

    class Cloud:
        STATUS_VERSION = StatusVersion.SKYPILOT

        def __repr__(self):
            return "Nebius"

    handle = types.SimpleNamespace(
        launched_resources=types.SimpleNamespace(cloud=Cloud()),
        cluster_name="c",
        cluster_name_on_cloud="c-abc",
        cluster_yaml="/tmp/c.yaml",
    )
    queries = []

    def query_instances(cloud_name, name, name_on_cloud, provider_config, non_terminated_only):
        queries.append((cloud_name, name, name_on_cloud, provider_config, non_terminated_only))
        return {"i-live": ("RUNNING", None), "i-gone": (None, None)}

    sky_pkg = types.ModuleType("sky")
    sky_pkg.clouds = types.SimpleNamespace(StatusVersion=StatusVersion)
    sky_pkg.global_user_state = types.SimpleNamespace(
        get_cluster_from_name=lambda cluster: {"handle": handle},
        get_cluster_yaml_dict=lambda path: {"provider": {"region": "eu-north1"}},
    )
    sky_pkg.provision = types.SimpleNamespace(query_instances=query_instances)
    monkeypatch.setitem(sys.modules, "sky", sky_pkg)
    monkeypatch.setitem(sys.modules, "sky.clouds", sky_pkg.clouds)
    monkeypatch.setitem(sys.modules, "sky.global_user_state", sky_pkg.global_user_state)
    monkeypatch.setitem(sys.modules, "sky.provision", sky_pkg.provision)

    probe = launcher._cloud_live_instances_probe("c")
    assert probe is not None
    assert probe() == ["i-live"]
    assert queries == [("Nebius", "c", "c-abc", {"region": "eu-north1"}, True)]

    Cloud.STATUS_VERSION = StatusVersion.CLOUD_CLI
    assert launcher._cloud_live_instances_probe("c") is None

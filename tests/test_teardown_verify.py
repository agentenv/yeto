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

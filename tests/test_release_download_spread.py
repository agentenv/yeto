"""Each sampler starts a background release download at its own fixed delay
(``yeto.rl.engine.policy_release.download_delay_s``)."""

from __future__ import annotations


def test_each_sampler_waits_its_own_fixed_part_of_the_spread(monkeypatch):
    """main: every background download slept ``random.uniform(0, 5)``."""
    from yeto.launch.nc.ssh_sampler import sampler_env
    from yeto.rl.engine import policy_release as pr

    monkeypatch.setattr(pr, "DOWNLOAD_SPREAD_S", 5.0)  # conftest turns the delay off for the suite
    env = lambda key, spread="30": {pr.DOWNLOAD_SPREAD_KEY_ENV: key, pr.DOWNLOAD_SPREAD_ENV: spread}  # noqa: E731
    delays = [pr.download_delay_s(env(f"nc-sampler-{i:02d}")) for i in range(16)]
    assert delays == [pr.download_delay_s(env(f"nc-sampler-{i:02d}")) for i in range(16)]  # the same at every release
    assert all(0.0 <= d < 30.0 for d in delays) and len({round(d, 3) for d in delays}) == 16
    assert max(delays) - min(delays) > 15.0  # 16 samplers use most of the window
    assert pr.download_delay_s(env("nc-sampler-03", "0")) == 0.0
    assert pr.download_delay_s(env("nc-sampler-03", "60")) == 2 * pr.download_delay_s(env("nc-sampler-03"))
    assert 0.0 <= pr.download_delay_s({}) < 5.0  # no setting: the host name, DOWNLOAD_SPREAD_S
    assert pr.download_delay_s({}) == pr.download_delay_s({})
    monkeypatch.setattr(pr, "DOWNLOAD_SPREAD_S", 0.0)
    assert pr.download_delay_s({}) == 0.0
    assert sampler_env("run")[pr.DOWNLOAD_SPREAD_ENV] == "30"  # NC samplers: 200 GB releases, 16 nodes

"""A deterministic start failure (agentic.setup) keeps the node and halts the pool for pool-retry."""
from yeto.launch.pool import island_pool


class SetupError(RuntimeError):
    pass


def test_setup_error_is_a_config_error():
    assert island_pool.is_config_error(SetupError("agentic setup [docker_cli]: pull unauthorized"))
    assert not island_pool.is_config_error(RuntimeError("ssh: connection reset"))
    e = RuntimeError("x")
    e.config_error = True
    assert island_pool.is_config_error(e)


def test_start_failed_with_setup_error_keeps_member(monkeypatch):
    events, retired = [], []

    class P:
        halted = None
        _event = staticmethod(lambda kind, **f: events.append(kind))
        log = staticmethod(lambda msg: None)
        _retire = staticmethod(lambda m, why, lost: retired.append(why))
        _fail = staticmethod(lambda now, why: retired.append("fail"))

    m = island_pool.Member(5, "t1", "s-0")
    p = P()
    island_pool.IslandPool._start_failed.__get__(p)  # bound below
    p._config_error = lambda mem, d: island_pool.IslandPool._config_error(p, mem, d)
    island_pool.IslandPool._start_failed(p, m, 0.0, SetupError("agentic setup [docker_cli]"))
    assert m.config_error and p.halted and not retired
    assert "sampler_config_error" in events

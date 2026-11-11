import pytest

from yeto.launch.agentic_setup import _shard_port


def test_port_range_gives_each_shard_its_own_listen_port():
    assert [_shard_port("18081-18084", f"run-x-{k}") for k in range(4)] == [18081, 18082, 18083, 18084]


@pytest.mark.parametrize("sid", ["run-x-4", "__SHARD__", None])
def test_port_range_refuses_shard_without_port(sid):
    with pytest.raises(ValueError):
        _shard_port("18081-18084", sid)


def test_pool_warns_when_the_sampler_spec_turns_the_incarnation_lease_off(tmp_path):
    import json

    from yeto.launch.nc import cli

    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"incarnation_lease": False}))
    assert any("incarnation_lease is false" in n for n in cli.pool_config_warnings({"spec": str(spec)}))
    spec.write_text(json.dumps({}))
    assert not any("incarnation_lease" in n for n in cli.pool_config_warnings({"spec": str(spec)}))

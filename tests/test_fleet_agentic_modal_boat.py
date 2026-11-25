"""Agentic pools beyond the NC ssh route: Modal pools with boat / AgentENV sandboxes,
the boat account's quota shared by every boat sampler of a run, ``sandbox.provider``
boat, a fleet's records relay / tunnel ports and its per-island compaction warnings.
No sandbox, no Modal call: fake ``modal``, fake forward, fake processes."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from yeto.launch import agentic_setup as su
from yeto.launch.pool import modal_provider as mp
from yeto.launch.rollout import pools
from yeto.rl import fleet, up, up_consistency
from yeto.rl import fleet_sandbox as fsb
from yeto.rl.engine.sampler_breakers import apply_harness_env
from yeto.rl.harness.harbor.pins import HARBOR_COMMIT

STORE = "s3://b/runs/r"


# ------------------------------------------------------------------ compaction per island


def _compacting_spec() -> dict:
    return {"engine": {"adapter": "vllm-server", "server": {}}, "reload_max_defer_s": 3600,
            "agentic": {"compaction": {"context_budget": 65536}, "weight_switch": "drain", "drain_timeout_s": 2400,
                        "task_defaults": {"wall_clock_s": 20000}}}


def test_compaction_warnings_read_the_islands_own_policy_age_and_release_cadence():
    args = SimpleNamespace(rl_max_policy_age=8, rl_release_every=4, rl_release_skip_busy=False)
    spec = _compacting_spec()
    assert up_consistency.compaction_switch_warnings(spec, args) == []
    island = SimpleNamespace(index=1, max_policy_age=2, release_every=1)
    age, every = up_consistency.compaction_switch_warnings(spec, args, island=island)
    assert "island 1's trainer's --rl-max-policy-age 2" in age
    assert "island 1's trainer's release every version" in every
    assert args.rl_max_policy_age == 8, "the fleet's args are not changed"
    # an island that keeps the fleet's values: as without an island
    same = SimpleNamespace(index=0, max_policy_age=None, release_every=None)
    assert up_consistency.compaction_switch_warnings(spec, args, island=same) == []
    assert up_consistency.island_view(args, None) is args


# ------------------------------------------------------------------ relay / tunnel ports


def _relay_spec(listen: str = "19100-19103") -> dict:
    return {"agentic": {"proxy": {"port": 18080}, "setup": {"records_relay": {"host": "relay.invalid",
                                                                              "listen_port": listen}}}}


def test_pools_sharing_a_port_range_take_consecutive_blocks():
    key = fleet.port_share_key(_relay_spec())
    assert key == ("records_relay", "relay.invalid", "19100-19103")
    assert fleet.port_share_key(_relay_spec("19100")) is None, "one port: no shard index"
    tunnel = {"agentic": {"setup": {"sandbox": {"docker_host": "ssh://u@h"}, "tunnel": {"listen": "10.0.0.1:19001-19009"}}}}
    assert fleet.port_share_key(tunnel) == ("tunnel", "ssh://u@h", "10.0.0.1:19001-19009")
    offsets = fleet.share_offsets([("i0-a", key, 2), ("i0-b", None, 4), ("i1-a", key, 2), ("i1-c", ("x",), 3)])
    assert offsets == {"i0-a": 0, "i1-a": 2}, "a pool alone on its range keeps its ports"


def test_a_shard_offset_moves_the_relay_and_tunnel_port():
    relay = su.RelaySection(host="relay.invalid", user="u", port=22, key=su.SecretRef(path="/k"), known_hosts=None,
                            port_lo=19100, port_hi=19103, public_host="relay.invalid", bind="0.0.0.0",
                            allow_cidrs=(), settle_s=1.0, ready_timeout_s=1.0)
    assert relay.for_shard("ag-i0-1", target_port=18080).listen_port == 19101
    assert relay.for_shard("ag-i1-1", target_port=18080, offset=2).listen_port == 19103
    with pytest.raises(ValueError, match="no port for shard"):
        relay.for_shard("ag-i1-2", target_port=18080, offset=2)
    assert su._shard_port("19001-19009", "ag-i1-0", 4) == 19005
    assert su._shard_offset({}) == 0 and su._shard_offset({"shard_offset": 3}) == 3
    with pytest.raises(ValueError, match="shard_offset"):
        su._shard_offset({"shard_offset": -1})


def test_port_findings_name_a_range_too_small_and_overlapping_ranges():
    key = ("records_relay", "relay.invalid", "19100-19102")
    problems = fsb.port_findings([("island 0 (a), pool i0-x", key, 0, 2), ("island 1 (b), pool i1-x", key, 2, 2)])
    assert len(problems) == 1 and problems[0].startswith("island 1 (b), pool i1-x: shards 2..3 of the records_relay "
                                                         "port range 19100-19102 need ports up to 19103")
    other = ("records_relay", "relay.invalid", "19101-19110")
    clash = fsb.port_findings([("island 0 (a), pool i0-x", key, 0, 2), ("island 1 (b), pool i1-x", other, 0, 2)])
    assert clash == ["island 0 (a), pool i0-x and island 1 (b), pool i1-x: their shards listen on the same "
                     "records_relay port(s) [19101]: two nodes would forward one port (set one listen range for the "
                     "pools, the fleet then gives each its own block of it, or give them ranges that do not overlap)"]
    assert fsb.port_findings([("p", key, 0, 2), ("q", ("records_relay", "other.invalid", "19100-19102"), 0, 2)]) == []


@pytest.fixture
def relay_fleet(tmp_path, monkeypatch):
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "rl-up")
    spec = {"sampler_id": "x", "store": "/work/stage/b/runs/__RUN_PREFIX__", "engine": {"adapter": "vllm-server"},
            "event_tape": "/work/sampler/events-__RUN_PREFIX__.jsonl", "sampling": {"n": 8},
            "prompts": {"path": "p.jsonl", "input_key": "q", "groups_per_round": 1, "format": "agentic",
                        "dispense": {"backend": "store", "namespace": "task-dispense"}},
            "agentic": {"max_in_flight": 8, "proxy": {"port": 18080}, "task_defaults": {"provider": "agentenv"},
                        "setup": {"provider": "agentenv",
                                  "agentenv": {"gateway_url": "https://gw.invalid", "api_key": {"env": "AK"}},
                                  "records_relay": {"host": "relay.invalid", "key": {"path": "/k"},
                                                    "listen_port": "19100-19103"}}}}
    (tmp_path / "agentic.json").write_text(json.dumps(spec))
    (tmp_path / "store.env").write_text("R2_BUCKET=b\n")
    return tmp_path


def _fleet_cfg(tmp_path: Path, limit: int = 2) -> dict:
    from tests.test_run_up_fleet import TRAINER_ARGV

    def pool(base):
        return {"name": "agent", "provider": {"kind": "modal", "image": "i", "secret": "s", "bucket": "b",
                                              "gpu": "H100"},
                "spec": str(tmp_path / "agentic.json"), "run_prefix": "r", "r2_env_file": str(tmp_path / "store.env"),
                "shard_base": base, "shard_limit": limit, "target": 1, "slots": [1, 2], "run": "r", "usd_per_h": 4.0}

    return {"run": "r", "store": STORE, "budget": {"usd": 40, "wall_h": 4, "nc_usd_per_h": 0},
            "fleet": {"trainer_argv": list(TRAINER_ARGV)},
            "islands": [{"name": "east", "trainer": {"gpu": "modal:1x2xh100", "usd_per_h": 8.0},
                         "samplers": [pool("ag")]},
                        {"name": "west", "trainer": {"gpu": "modal:1x2xh100", "usd_per_h": 8.0},
                         "samplers": [pool("ag")]}]}


def test_the_compile_gives_two_islands_on_one_relay_range_their_own_ports(relay_fleet):
    cfg, _plan = fleet.compile_fleet(_fleet_cfg(relay_fleet))
    knobs = {p["name"]: p["sampler"].get("shard_offset") for p in cfg["samplers"]}
    assert knobs == {"i0-agent": 0, "i1-agent": 2}
    ports = {}
    for pool in cfg["samplers"]:
        spec = up_consistency.effective_spec(pool)
        for n in range(2):
            shard = f"{pool['shard_base']}-{n}"
            setup = su.parse({**spec, "sampler_id": shard})
            ports[shard] = setup.relay.listen_port
    assert ports == {"ag-i0-0": 19100, "ag-i0-1": 19101, "ag-i1-0": 19102, "ag-i1-1": 19103}
    with pytest.raises(ValueError, match="sampler.shard_offset is written by the fleet compile"):
        bad = _fleet_cfg(relay_fleet)
        bad["islands"][0]["samplers"][0]["sampler"] = {"shard_offset": 1}
        fleet.compile_fleet(bad)


def test_run_up_refuses_a_shared_relay_range_too_small_for_the_fleet(relay_fleet):
    path = relay_fleet / "run.json"
    path.write_text(json.dumps(_fleet_cfg(relay_fleet, limit=3)))
    run = up.Run(up.load_config(str(path)))
    rows = run.fleet_port_rows()
    assert [(w.split(", ")[-1], off, lim) for w, _k, off, lim in rows] == [("pool i0-agent", 0, 3),
                                                                            ("pool i1-agent", 3, 3)]
    problems = fsb.port_findings(rows)
    assert len(problems) == 1 and "pool i1-agent: shards 3..5" in problems[0] and "past its end 19103" in problems[0]


# ------------------------------------------------------------------ the boat account


def test_boat_shares_split_the_account_over_every_process():
    pools_ = {"a": {"processes": 2, "max_in_flight": 20, "explicit": {}},
              "b": {"processes": 2, "max_in_flight": 10, "explicit": {}}}
    shares = pools.boat_shares(fsb.boat_account({}), pools_)
    assert shares == {"a": {"boat_max_active_n": 33, "boat_starts_per_minute_n": 3, "boat_starts_per_hour_n": 15},
                      "b": {"boat_max_active_n": 16, "boat_starts_per_minute_n": 3, "boat_starts_per_hour_n": 15}}
    assert sum(pools_[n]["processes"] * s["boat_max_active_n"] for n, s in shares.items()) <= 100
    # a value the pool sets itself is kept and counted
    pools_["a"]["explicit"] = {"boat_starts_per_minute_n": 4}
    shares = pools.boat_shares(fsb.boat_account({"boat": {"starts_per_hour_n": None}}), pools_)
    assert "boat_starts_per_minute_n" not in shares["a"] and shares["b"]["boat_starts_per_minute_n"] == 2
    assert "boat_starts_per_hour_n" not in shares["b"], "an hourly limit that is not held is not split"
    with pytest.raises(ValueError, match="holds 100 active sandboxes .* need 120"):
        pools.boat_shares(fsb.boat_account({}), {"a": {"processes": 4, "max_in_flight": 30, "explicit": {}}})
    with pytest.raises(ValueError, match="starts 12 sandboxes a minute .* 13 boat sampler process"):
        pools.boat_shares(fsb.boat_account({}), {"a": {"processes": 13, "max_in_flight": 1, "explicit": {}}})


def test_sandbox_boat_limits_are_checked():
    assert fsb.boat_account({"boat": {"max_active_n": 200}})["max_active_n"] == 200
    for bad, match in (({"boat": {"max_active": 1}}, "unknown sandbox.boat keys"),
                       ({"boat": {"max_active_n": 0}}, "whole number >= 1"), ({"boat": 3}, "must be an object")):
        with pytest.raises(ValueError, match=match):
            fsb.boat_account(bad)


def _boat_pool(tmp_path: Path, name: str, *, target: int, mif: int, env_file: str | None = None) -> dict:
    spec = {"sampler_id": "x", "store": "/s", "sampling": {"n": 1},
            "prompts": {"path": "p.jsonl", "input_key": "q", "groups_per_round": 1, "format": "agentic"},
            "agentic": {"max_in_flight": mif, "task_defaults": {"provider": "boat"}}}
    (tmp_path / f"{name}.spec.json").write_text(json.dumps(spec))
    doc = {"name": name, "provider": {"kind": "modal", "image": "i", "secret": "s", "bucket": "b", "gpu": "H100"},
           "spec": str(tmp_path / f"{name}.spec.json"), "run_prefix": "r", "shard_base": name, "target": target,
           "slots": [1, 2, 3], "run": "r"}
    if env_file:
        doc["agentic_env_file"] = env_file
    return doc


def test_run_up_gives_each_boat_pool_its_share_and_the_samplers_hold_it(tmp_path):
    cfg = {"run": "r", "store": STORE, "trainer": {"argv": []},
           "samplers": [_boat_pool(tmp_path, "a", target=2, mif=20), _boat_pool(tmp_path, "b", target=1, mif=10)]}
    refs = pools.resolve(cfg, tmp_path / "state")
    spec = up_consistency.effective_spec(refs[0].load())
    assert spec["breakers"] == {"harness": {"sandbox_caps": {"boat_max_active_n": 40, "boat_starts_per_minute_n": 4,
                                                             "boat_starts_per_hour_n": 20}}}
    assert "harness.sandbox_caps.boat_max_active_n" in refs[0].spec_carried
    env: dict[str, str] = {}
    apply_harness_env(spec, env)
    assert env == {"YETO_BOAT_MAX_ACTIVE": "40", "YETO_BOAT_STARTS_PER_MINUTE": "4", "YETO_BOAT_STARTS_PER_HOUR": "20"}
    # an env file's own value is kept, never overwritten
    env_file = tmp_path / "boat.env"
    env_file.write_text("YETO_BOAT_STARTS_PER_MINUTE=4\n")
    cfg["samplers"][0]["agentic_env_file"] = str(env_file)
    refs = pools.resolve(cfg, tmp_path / "state2")
    caps = up_consistency.effective_spec(refs[0].load())["breakers"]["harness"]["sandbox_caps"]
    assert "boat_starts_per_minute_n" not in caps
    other = up_consistency.effective_spec(refs[1].load())["breakers"]["harness"]["sandbox_caps"]
    assert other["boat_starts_per_minute_n"] == 12 - 4 * 2, "the rest of the account's rate"
    # a Modal pool's provider.env reaches its samplers too: kept and counted the same way
    cfg["samplers"][0].pop("agentic_env_file")
    cfg["samplers"][0]["provider"] = dict(cfg["samplers"][0]["provider"], env={"YETO_BOAT_MAX_ACTIVE": "30"})
    refs = pools.resolve(cfg, tmp_path / "state3")
    assert "boat_max_active_n" not in up_consistency.effective_spec(refs[0].load())["breakers"]["harness"]["sandbox_caps"]
    rest = up_consistency.effective_spec(refs[1].load())["breakers"]["harness"]["sandbox_caps"]
    assert rest["boat_max_active_n"] == 100 - 30 * 2


def test_runs_without_two_boat_processes_are_left_as_they_are(tmp_path):
    one = {"run": "r", "store": STORE, "trainer": {"argv": []}, "samplers": [_boat_pool(tmp_path, "a", target=1, mif=8)]}
    refs = pools.resolve(one, tmp_path / "s1")
    assert "breakers" not in up_consistency.effective_spec(refs[0].load())
    k31 = Path(__file__).resolve().parents[1] / "examples" / "k31"
    pool = json.loads((k31 / "pool.nc-mi355x-agentic-262k.json").read_text())
    pool["spec"] = str(k31 / "sampler-spec.nc-mi355x-vllm-agentic-262k.json")
    (tmp_path / "k31.json").write_text(json.dumps(pool))
    ref = pools.PoolRef("pool", str(tmp_path / "k31.json"))
    assert pools.with_boat_shares({"run": "r"}, [ref], tmp_path / "s2") == [ref], "the customer's remote-docker pool"


def test_sandbox_provider_boat_in_the_run_config():
    base = {"run": "r", "store": STORE, "trainer": {"argv": ["x"]}}
    up._check_run_sections({**base, "sandbox": {"provider": "boat", "boat": {"max_active_n": 50}}})
    with pytest.raises(ValueError, match="boat sandboxes have no docker host"):
        up._check_run_sections({**base, "sandbox": {"provider": "boat", "docker_host": "ssh://u@h"}})
    with pytest.raises(ValueError, match="sandbox.boat.max_active_n must be a whole number"):
        up._check_run_sections({**base, "sandbox": {"boat": {"max_active_n": -1}}})
    pool = fsb.SandboxPool(name="i0-a", where="island 0 (e), pool i0-a", island=0, provider="boat", max_in_flight=8,
                           shards=("a-i0-0",), max_nodes=1, envs={"run": "r", "hosts": []})
    problems, _ = fsb.provider_findings([pool], sandbox={"provider": "boat"})
    assert problems and "sandbox.provider is boat" in problems[0]
    over = [fsb.SandboxPool(name=f"i{k}-a", where=f"pool i{k}-a", island=k, provider="boat", max_in_flight=30,
                            shards=(f"a-i{k}-0",), max_nodes=1) for k in (0, 1)]
    assert fsb.capacity_findings(over, boat={"max_active_n": 50})[0], "sandbox.boat replaces the plan's limit"
    assert not fsb.capacity_findings(over, boat={"max_active_n": 80})[0]


# ------------------------------------------------------------------ agentic.setup provider boat (NC ssh)


def _boat_setup_spec(tmp_path: Path, **setup) -> dict:
    token = tmp_path / "boat.token"
    token.write_text("tok-" + "x" * 20 + "\n")
    token.chmod(0o600)
    tasks = tmp_path / "tasks" / "a"
    tasks.mkdir(parents=True, exist_ok=True)
    (tasks / "task.toml").write_text("[task]\n")
    doc = {"provider": "boat", "boat": {"api_key": {"path": str(token)}},
           "records_proxy": {"public_base": "https://records.example.com"},
           "tasks": {"source": str(tmp_path / "tasks"), "ids": ["a"]}, **setup}
    return {"sampler_id": "s-0", "agentic": {"proxy": {"port": 18080}, "task_defaults": {"provider": "boat"},
                                             "setup": doc}}


def test_agentic_setup_takes_boat_sandboxes(tmp_path):
    s = su.parse(_boat_setup_spec(tmp_path))
    assert s.provider == "boat" and s.sandbox is None and s.tunnel is None and s.agentenv is None
    env = s.container_env()
    assert env["HARBOR_ENV_TYPE"] == "boat" and env["BOAT_API_KEY_FILE"] == s.boat_key_file
    assert env["YETO_RECORDS_PROXY_PUBLIC_BASE"] == "https://records.example.com"
    assert "DOCKER_HOST" not in env and not any("tok-" in str(v) for v in env.values()), "the token stays in its file"
    lines = "\n".join(su.describe(s))
    assert su.BOAT_SDK_REQUIREMENT in lines and "boat: key" in lines and "tok-" not in lines
    assert su.local_problems(s) == []
    for change, match in (({"boat": {}}, "boat.api_key is required"),
                          ({"sandbox": {"docker_host": "ssh://u@h"}}, "boat sandboxes have no docker host"),
                          ({"agentenv": {"gateway_url": "https://g"}}, "agentic.setup.agentenv is for provider agentenv")):
        with pytest.raises(ValueError, match=match):
            su.parse(_boat_setup_spec(tmp_path, **change))
    placeholder = su.parse(_boat_setup_spec(tmp_path, records_proxy={"public_base": "https://<records>"}))
    assert placeholder.boat.records_placeholder and any("WARNING" in n for n in placeholder.notes)


def test_the_boat_client_reads_the_token_file(tmp_path, monkeypatch):
    from yeto.rl.harness.harbor import boat_client

    path = tmp_path / "k"
    path.write_text("export BOAT_API_KEY='abc123'\n")
    monkeypatch.delenv("BOAT_API_KEY", raising=False)
    monkeypatch.setenv("BOAT_API_KEY_FILE", str(path))
    assert boat_client.load_token() == "abc123"
    path.write_text("bare-token\n")
    assert boat_client.load_token() == "bare-token"
    path.write_text("")
    with pytest.raises(boat_client.BoatError, match="holds no BOAT_API_KEY"):
        boat_client.load_token()


# ------------------------------------------------------------------ Modal agentic pools


def _modal_pool(tmp_path: Path, **provider) -> dict:
    spec = {"sampler_id": "x", "store": "/s", "sampling": {"n": 1}, "event_tape": str(tmp_path / "tape.jsonl"),
            "prompts": {"path": "/mnt/yeto-models/prompts/tb2.jsonl", "input_key": "q", "groups_per_round": 1,
                        "format": "agentic"},
            "agentic": {"max_in_flight": 8, "proxy": {"host": "127.0.0.1", "port": 18080},
                        "task_defaults": {"provider": "boat"}}}
    (tmp_path / "modal.spec.json").write_text(json.dumps(spec))
    section = {"kind": "modal", "image": "i", "secret": "store", "secrets": ["boat-token"], "bucket": "b",
               "gpu": "H100", "model_volume": "models", "tasks_dir": "/mnt/yeto-models/tasks/tb2", **provider}
    return {"provider": section, "spec": str(tmp_path / "modal.spec.json"), "run_prefix": "r/trainer0",
            "shard_base": "m", "target": 1, "slots": [1], "run": "r", "r2_env_file": str(tmp_path / "store.env")}


def test_an_agentic_modal_pool_payload_tunnels_the_records_proxy(tmp_path):
    cfg = _modal_pool(tmp_path)
    provider = mp.from_config(cfg, "r", client=object())
    assert provider.harbor == "boat"
    payload = mp.agentic_payload({"spec": json.loads(Path(cfg["spec"]).read_text()), "shard": "m-0",
                                  "env": {"HARBOR_TRIALS_DIR": "/elsewhere"}}, provider.config)
    assert payload["spec"]["agentic"]["proxy"] == {"host": "0.0.0.0", "port": 18080}
    assert payload["records_port"] == 18080 and payload["harbor_site"] == mp.HARBOR_SITE
    assert payload["env"] == {"HARBOR_TASKS_DIR": "/mnt/yeto-models/tasks/tb2", "HARBOR_TRIALS_DIR": "/elsewhere",
                              "HARBOR_ENV_TYPE": "boat"}, "provider.env still wins"
    plain = mp.from_config({**cfg, "spec": None}, "r", client=object())
    assert plain.harbor is None


class _Proc:
    returncode = 0
    stderr = None

    def __init__(self, seen, env):
        seen.append(dict(env))

    def poll(self):
        return 0


def _env(monkeypatch):
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://e.invalid")


def test_the_call_runs_the_sampler_inside_the_records_tunnel(tmp_path, monkeypatch):
    _env(monkeypatch)
    seen, forwarded = [], []

    class Tunnel:
        url = "https://abc.modal.run/"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            forwarded.append("closed")

    def forward(port):
        forwarded.append(port)
        return Tunnel()

    payload = {"shard": "m-0", "bucket": "b", "run_prefix": "r", "spec": {"agentic": {}}, "records_port": 18080,
               "harbor_site": mp.HARBOR_SITE, "env": {}}
    out = mp.rollout_sampler_body(payload, popen=lambda argv, env, **k: _Proc(seen, env), sleep=lambda s: None,
                                  root=str(tmp_path), forward=forward)
    assert out["rc"] == 0 and forwarded == [18080, "closed"], "the tunnel spans the sampler's life"
    env = seen[0]
    assert env["YETO_RECORDS_PROXY_PUBLIC_BASE"] == "https://abc.modal.run"
    secret = Path(env["YETO_RECORDS_PROXY_SECRET_FILE"])
    assert secret.read_text() and stat.S_IMODE(os.stat(secret).st_mode) == 0o600
    assert env["PYTHONPATH"].endswith(":" + mp.HARBOR_SITE)
    seen.clear()
    mp.rollout_sampler_body({"shard": "m-1", "bucket": "b", "run_prefix": "r", "spec": {}},
                            popen=lambda argv, env, **k: _Proc(seen, env), sleep=lambda s: None, root=str(tmp_path),
                            forward=lambda port: pytest.fail("no tunnel for a pool that is not agentic"))
    assert "YETO_RECORDS_PROXY_PUBLIC_BASE" not in seen[0]


class _FakeModal:
    def __init__(self):
        self.functions, self.commands = [], []
        modal = self

        class Image:
            def entrypoint(self, _):
                return self

            def run_commands(self, *cmds):
                modal.commands += list(cmds)
                return self

            def env(self, _):
                return self

            def add_local_python_source(self, *a, **k):
                return self

        self.image = Image()

        class Secret:
            @staticmethod
            def from_name(name):
                return f"secret:{name}"

        class Volume:
            @staticmethod
            def from_name(name):
                return SimpleNamespace(read_only=lambda: f"volume:{name}")

        class App:
            def __init__(self, name):
                pass

            def function(self, **kw):
                modal.functions.append(kw)
                return lambda fn: fn

        self.Secret, self.Volume, self.App = Secret, Volume, App


def test_the_agentic_app_image_carries_pinned_harbor_and_every_secret(tmp_path, monkeypatch):
    from yeto.cloud import modal_images

    fake = _FakeModal()
    monkeypatch.setattr(modal_images, "engine_registry_image", lambda modal, ref: fake.image)
    cfg = _modal_pool(tmp_path)
    client = mp.ModalSdkClient(mp.from_config(cfg, "r", client=object()).config, run_id="r", modal=fake, harbor="boat")
    client.build_app()
    assert fake.functions[0]["secrets"] == ["secret:store", "secret:boat-token"]
    assert fake.functions[0]["volumes"] == {"/mnt/yeto-models": "volume:models"}
    install = next(c for c in fake.commands if "pip install" in c)
    assert HARBOR_COMMIT in install and su.BOAT_SDK_REQUIREMENT in install and mp.HARBOR_SITE in install
    assert "e2b" not in install and len(fake.commands) == 3
    assert all("\n" not in c for c in fake.commands), "each command is one Dockerfile RUN line"
    fake2 = _FakeModal()
    monkeypatch.setattr(modal_images, "engine_registry_image", lambda modal, ref: fake2.image)
    mp.ModalSdkClient({"image": "i", "secret": "store"}, run_id="r", modal=fake2).build_app()
    assert fake2.commands == [] and fake2.functions[0]["secrets"] == ["secret:store"], "a plain pool's app as before"


def test_the_dry_run_names_what_an_agentic_modal_pool_lacks(tmp_path):
    good = _modal_pool(tmp_path)
    assert mp.agentic_problems(good, good["provider"]) == []
    bad = _modal_pool(tmp_path, tasks_dir="/data/tasks", secrets=[], env={"BOAT_API_KEY": "x", "HF_TOKEN": "y"})
    found = mp.agentic_problems(bad, bad["provider"])
    assert any("tasks_dir /data/tasks is not on the model volume" in p for p in found)
    assert any("['BOAT_API_KEY', 'HF_TOKEN'] travel in the call payload in clear text" in p for p in found)
    assert any("a boat pool needs the boat token as BOAT_API_KEY in a Modal secret" in p for p in found)
    spec = json.loads(Path(good["spec"]).read_text())
    spec["agentic"].update(proxy={"port": 0}, task_defaults={"provider": "remote-docker"}, setup={"root": "/work/h"})
    Path(good["spec"]).write_text(json.dumps(spec))
    found = mp.agentic_problems(good, good["provider"])
    assert any("needs a fixed agentic.proxy.port" in p for p in found)
    assert any("'remote-docker': a Modal pool's trials run in cloud sandboxes" in p for p in found)
    assert any("agentic.setup is the NC ssh node setup" in p for p in found)
    plain = _modal_pool(tmp_path)
    plain["spec"] = None
    assert mp.agentic_problems(plain, plain["provider"]) == []


def test_the_harbor_layer_prune_command_runs_as_one_shell_line(tmp_path, monkeypatch):
    import subprocess
    import sys

    site = tmp_path / "site"
    (site / "empty_pkg").mkdir(parents=True)
    monkeypatch.setattr(mp, "HARBOR_SITE", str(site))
    fake = _FakeModal()
    mp.agentic_harbor_layer(fake.image, python=sys.executable, provider="boat")
    prune = fake.commands[-1]
    assert "\n" not in prune
    out = subprocess.run(["sh", "-c", prune], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "YETO_SETUP_PRUNED"
    assert not (site / "empty_pkg").exists(), "the prune script itself ran (it removes empty package dirs)"

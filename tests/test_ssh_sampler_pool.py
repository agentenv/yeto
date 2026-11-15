"""The sampler pool on the operator's own SSH nodes (provider kind ``ssh``)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from yeto.launch import ssh_sampler_pool as sp
from yeto.launch.nc import cli
from yeto.launch.node_inventory import parse_node
from yeto.launch.pool import provider as providers

IDLE = (Path(__file__).parent / "data" / "npu_smi_910b4_idle.txt").read_text()
BUSY = IDLE.replace(
    "| No running processes found in NPU 3                                                            |",
    "| 3       0                 | 4242          | python3                  | 3000                    |")
DIGEST = "quay.nju.edu.cn/ascend/vllm-ascend@sha256:" + "e" * 64


def npu_node(**kw):
    raw = {"name": "node-a", "host": "node-a", "device_family": "npu", "cpu_arch": "aarch64", "card": "910B4",
           "cards_per_node": 8, "hbm_gib": 64, "host_memory_gib": 1024, "runtime": "rootfs",
           "mounts": {"/srv/yeto/shared": "/srv/yeto/shared"}, **kw}
    return parse_node(raw, index=0)


def provider_section(**kw):
    return {"kind": "ssh", "nodes": ["node-a"], "workdir": "/srv/yeto/run0", "runtime": "container",
            "container": {"engine": "podman", "image": "3a7d901460ab", "pull_ref": DIGEST,
                          "security_opt": ["seccomp=unconfined"]},
            "devices": [0], "store": "/srv/yeto/run0/store", **kw}


def pool_cfg(tmp_path, **section):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"sampler_id": "x", "store": "/s", "prompts": {"groups_per_round": 2},
                                "engine": {"factory": "f:g"}}))
    return {"target": 1, "slots": [0], "shard_base": "f09", "spec": str(spec), "run_prefix": "f09",
            "provider": provider_section(**section)}


def conf(**section):
    return sp.parse_config({"provider": provider_section(**section)}, inventory=[npu_node()])


def test_kind_ssh_is_registered_and_other_kinds_keep_their_defaults():
    assert providers.kinds() == ["daytona", "modal", "nc", "sky", "ssh"]
    assert providers.registered("ssh").pool_ops == "yeto.launch.ssh_sampler_pool:pool_ops"
    assert providers.kind_of({}) == "nc"  # the default provider is unchanged
    assert cli.REQUIRED_SSH_KEYS == ("spec", "run_prefix", "r2_env_file")


def test_pool_config_needs_r2_env_file_only_on_the_r2_path(tmp_path):
    cfg = pool_cfg(tmp_path)
    assert cli.pool_config(cfg).target == 1  # node-local store: no r2_env_file
    cfg["provider"].pop("store")
    with pytest.raises(ValueError, match="r2_env_file"):
        cli.pool_config(cfg)
    nc = {k: v for k, v in pool_cfg(tmp_path).items() if k not in ("provider", "slots")}
    with pytest.raises(ValueError, match="r2_env_file"):
        cli.pool_config(dict(nc, accounts=[1]))  # a National Compute pool still needs it


def test_config_refusals():
    with pytest.raises(ValueError, match="runtime"):
        conf(runtime="kubernetes")
    with pytest.raises(ValueError, match="engine"):
        conf(container={"engine": "lxc", "image": "x"})
    with pytest.raises(ValueError, match="not in the inventory"):
        conf(nodes=["node-b"])
    with pytest.raises(ValueError, match="workdir"):
        conf(workdir="/data")
    with pytest.raises(ValueError, match="venv.python"):
        conf(runtime="venv", container=None, venv={})
    with pytest.raises(ValueError, match="unknown keys"):
        conf(colour="red")


def test_offline_problems_refuse_a_shared_local_store(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "load_inventory", lambda _p: [npu_node()])
    cfg = pool_cfg(tmp_path, inventory="inv.yaml")
    assert sp.offline_problems(cfg) == []
    assert any("target must be 1" in p for p in sp.offline_problems(dict(cfg, target=2)))
    cfg["provider"].pop("store")
    assert any("r2_env_file" in p for p in sp.offline_problems(cfg))


def test_container_argv_carries_the_configured_seccomp_and_cards():
    c = conf()
    argv = sp.container_run_argv(c, npu_node(), name="yeto-sampler-a", image="3a7d901460ab", yeto_dir="/y",
                                 spec_path="/w/a.json", env={"K": "V"}, env_file=None, cards=[0])
    joined = " ".join(argv)
    assert argv[:3] == ["podman", "run", "-d"]
    assert "--security-opt seccomp=unconfined" in joined
    assert "--device /dev/davinci0 " in joined and "/dev/davinci1" not in joined
    assert "-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro" in joined
    assert "-v /srv/yeto/run0:/srv/yeto/run0" in joined
    assert "PYTHONPATH=" not in " ".join(argv[:-1]) and argv[-3] == "3a7d901460ab"
    assert argv[-1].startswith('export PYTHONPATH=/y"${PYTHONPATH:+:$PYTHONPATH}"; ')  # the image's path kept
    assert "exec python3 -m yeto.rl.engine.engine_sampler" in argv[-1]
    no_opt = sp.parse_config({"provider": provider_section(container={"image": "img"})}, inventory=[npu_node()])
    assert "--security-opt" not in sp.container_run_argv(no_opt, npu_node(), name="n", image="img", yeto_dir="/y",
                                                         spec_path="/s", env={}, env_file=None, cards=[0])


def test_pull_by_digest_when_the_image_is_missing():
    cmd = sp.container_pull_command(conf(), "3a7d901460ab")
    assert f"podman pull {DIGEST}" in cmd and "inspect --type image 3a7d901460ab" in cmd
    never = sp.parse_config({"provider": provider_section(container={"image": "i", "pull": "never"})},
                            inventory=[npu_node()])
    assert "pull" not in sp.container_pull_command(never, "i").replace("YETO_SSH_NO_IMAGE", "")


def test_venv_script_and_the_pid_stop():
    c = conf(runtime="venv", container=None, venv={"python": "/e/bin/python", "source": ["/cann/set_env.sh"],
                                                  "ld_preload": "/opt/libgomp.so.1"})
    p = sp.node_paths(c.workdir, "a")
    script = sp.start_script(c, yeto_dir="/y", spec_path=p["spec"], pid_path=p["pid"], env={"A": "1"}, env_file=None)
    lines = script.splitlines()
    assert lines.index("source /cann/set_env.sh") < lines.index("set -u")
    assert 'export PYTHONPATH=/y"${PYTHONPATH:+:$PYTHONPATH}"' in lines  # CANN's own path kept (A07 pit 3)
    assert lines[-2] == f"echo $$ > {p['pid']}" and lines[-1].startswith("exec /e/bin/python -m")
    stop = sp.venv_stop_command(p, 30)
    assert "pkill" not in stop and "pgrep" not in stop
    assert "ps -o args= -p" in stop and 'kill -TERM "$p"' in stop
    assert stop.index('kill -TERM "$p"') < stop.index("kill -TERM -- -")  # the sampler first, the group last
    assert sp.LEFT_LINE in stop and "grep -qF -- \"$m\"" in stop  # the group only when it carries the marker
    assert ": > " + p["log"] in sp.venv_launch_command(p, sp.start_marker("r", "a", "r-node-a-1"))  # a fresh log
    assert "setsid" in sp.venv_launch_command(p, sp.start_marker("r", "a", "r-node-a-1"))


def test_preflight_checks_cards_idle_and_disk():
    c, node = conf(), npu_node()
    ok = f"YETO_SSH_ARCH aarch64\nYETO_SSH_FREE_GIB 300\nYETO_SSH_CARDS_BEGIN\n{IDLE}\nYETO_SSH_CARDS_END rc=0\n"
    facts = sp.check_preflight(node, c, ok)
    assert facts["cards"] == 8 and facts["busy"] == []
    assert sp.busy_npus(BUSY) == [3]
    sp.check_preflight(node, c, ok.replace(IDLE, BUSY))  # card 3 is not this provider's (devices [0])
    with pytest.raises(sp.PreflightError, match="cards \\[3\\]"):
        sp.check_preflight(node, conf(devices=[3]), ok.replace(IDLE, BUSY))
    with pytest.raises(sp.PreflightError, match="GiB free"):
        sp.check_preflight(node, c, ok.replace("FREE_GIB 300", "FREE_GIB 3"))
    with pytest.raises(sp.PreflightError, match="uname"):
        sp.check_preflight(node, c, ok.replace("aarch64", "x86_64"))


def test_provider_tickets(tmp_path):
    prov = sp.SshNodeProvider("r", conf(), slots=[0], state_path=tmp_path / "s.json")
    assert prov.account_free(0)
    t = prov.order(0)
    assert t == "r-node-a-1" and prov.status(0, t).state == "active" and not prov.account_free(0)
    assert prov.run_tickets() == [(0, t)]
    prov.release(0, t)
    assert prov.status(0, t).end_reason == "released" and prov.released_by_us(t) and prov.account_free(0)
    with pytest.raises(ValueError, match="no node"):
        sp.SshNodeProvider("r", conf(), slots=[1], state_path=tmp_path / "x.json")


class FakeRun:
    def __init__(self):
        self.calls = []

    def __call__(self, argv, **kw):
        self.calls.append((argv, kw))
        if argv[0] == "git":
            return SimpleNamespace(returncode=0, stdout="abc123\n", stderr="")
        cmd = argv[-1]
        if "YETO_SSH_CARDS_BEGIN" in cmd:
            out = f"YETO_SSH_ARCH aarch64\nYETO_SSH_FREE_GIB 300\nYETO_SSH_CARDS_BEGIN\n{IDLE}\nYETO_SSH_CARDS_END rc=0\n"
        elif "run -d" in cmd:
            out = "YETO_SSH_STARTED container=yeto-sampler-f09-0\n"
        elif "YETO_NC_PROBE" in cmd:
            out = "YETO_NC_PROBE status=running exit=0 joined=1 age=3 tw=0 tok=- run=- wq=- vh=0 vm=-\n"
        else:
            out = "stopped\n"
        return SimpleNamespace(returncode=0, stdout=out, stderr="")


def test_ops_start_probe_drain(tmp_path, monkeypatch):
    class Pop:
        stdout = None

        def __init__(self, *a, **k):
            pass

        def wait(self):
            return 0

    monkeypatch.setattr(sp.subprocess, "Popen", Pop)
    monkeypatch.setattr(cli.tasks, "sampler_spec", lambda spec, sampler_id, shards: dict(spec, sampler_id=sampler_id))
    monkeypatch.setattr(cli.tasks, "apply_sampler_overrides", lambda spec, **kw: spec)
    run = FakeRun()
    cfg = pool_cfg(tmp_path)
    ops = sp.SshSamplerPoolOps(cfg, conf=conf(), run=run, state_dir=tmp_path)
    t = ops.order(0)
    m = SimpleNamespace(account=0, ticket=t, shard="f09-0", data_slot=None, replaces_shard=None,
                        predecessor_lost=False)
    ops.start(m, ["f09-0"])
    cmds = [c[0][-1] for c in run.calls if c[0][0] == "ssh"]
    assert all(c[0][1] == "-n" for c in run.calls if c[0][0] == "ssh" and c[1].get("input") is None
               and "stdin" not in c[1])
    spec_upload = next(c for c in run.calls if c[0][0] == "ssh" and "f09-0.json" in c[0][-1])
    spec = json.loads(spec_upload[1]["input"])
    assert spec["store"] == spec["buffer_root"] == "/srv/yeto/run0/store" and spec["start_on_latest"]
    run_cmd = next(c for c in cmds if "run -d" in c)
    assert "YETO_SPOT_RECLAIM_MARKER=" in run_cmd and "ASCEND_RT_VISIBLE_DEVICES=0" in run_cmd
    assert ops.probe(m).joined
    ops.drain(m)
    stop = run.calls[-1][0][-1]
    assert "podman kill --signal TERM yeto-sampler-f09-0" in stop and "seq 120" in stop
    assert stop.index("kill --signal TERM") < stop.index("podman stop -t 10")
    assert ops.poll(m).state == "ended"
    assert not any("pkill" in c or "pgrep" in c for c in cmds)


def test_preflight_refusal_stops_the_start(tmp_path):
    class Busy(FakeRun):
        def __call__(self, argv, **kw):
            r = super().__call__(argv, **kw)
            if "YETO_SSH_CARDS_BEGIN" in argv[-1]:
                r.stdout = r.stdout.replace(IDLE, BUSY)
            return r

    run = Busy()
    ops = sp.SshSamplerPoolOps(pool_cfg(tmp_path), conf=conf(devices=[3]), run=run, state_dir=tmp_path)
    m = SimpleNamespace(account=0, ticket=ops.order(0), shard="f09-0")
    with pytest.raises(sp.PreflightError):
        ops.start(m, ["f09-0"])
    assert not any("run -d" in c[0][-1] for c in run.calls if c[0][0] == "ssh")


def test_ssh_timeout_reads_unreachable(tmp_path):
    def run(argv, **kw):
        raise subprocess.TimeoutExpired(argv, 1)

    ops = sp.SshSamplerPoolOps(pool_cfg(tmp_path), conf=conf(), run=run, state_dir=tmp_path)
    m = SimpleNamespace(account=0, ticket=ops.order(0), shard="f09-0")
    assert ops.probe(m).node == "unreachable"


def test_npu_host_paths_follow_the_preflight():
    c = conf()
    facts = sp.check_preflight(npu_node(), c, "YETO_SSH_ARCH aarch64\nYETO_SSH_FREE_GIB 300\n"
                               "YETO_SSH_HOSTPATH /dev/davinci_manager\nYETO_SSH_HOSTPATH /usr/local/Ascend/driver\n"
                               f"YETO_SSH_CARDS_BEGIN\n{IDLE}\nYETO_SSH_CARDS_END rc=0\n")
    argv = " ".join(sp.container_run_argv(c, npu_node(), name="n", image="i", yeto_dir="/y", spec_path="/s",
                                          env={}, env_file=None, cards=[0], host_paths=facts["host_paths"]))
    assert "/dev/davinci_manager" in argv and "/usr/local/Ascend/driver:" in argv
    assert "/usr/local/dcmi" not in argv and "/dev/hisi_hdc" not in argv


def test_the_left_line_is_what_the_sampler_prints_after_its_leave():
    import inspect

    from yeto.rl.engine import engine_sampler_entry

    assert sp.LEFT_LINE in inspect.getsource(engine_sampler_entry.hold_after_reclaim)


def test_own_nodes_cost_nothing_unless_the_config_says(tmp_path):
    cfg = pool_cfg(tmp_path)
    assert cli.pool_price(cfg) == (0.0, "usd") and cli.pool_hourly_ucr(cfg) == 0
    cfg["provider"]["price_usd_h"] = 2.5
    assert cli.pool_price(cfg) == (2.5, "usd")
    assert cli.pool_price({"provider": {"kind": "nc"}})[0] is None  # National Compute unchanged


# -- the node hour: None (no price stated) is the MI355X rate, an explicit 0 costs nothing

MAIN_UNPRICED_SPEND = ("estimated spend: 24000000 ucr/h per node (24 credits/h, the 8-GPU MI355X rate); steady "
                       "48000000 ucr/h (48 credits/h); up to 48000000 ucr/h (48 credits/h) while replacements overlap")


def _dry_run_spend(cfg, tmp_path, monkeypatch, capsys):
    from yeto.launch.nc import pool_run

    monkeypatch.setenv("YETO_NC_STATE_DIR", str(tmp_path / "state"))
    pool_run.dry_run(cfg, "pool.json")
    return [line for line in capsys.readouterr().out.splitlines() if line.startswith("estimated spend")]


def _priced_cfg(tmp_path, provider):
    spec = tmp_path / "s.json"
    spec.write_text(json.dumps({"sampler_id": "x", "store": "/s", "prompts": {"groups_per_round": 2},
                                "engine": {"factory": "f:g"}}))
    env = tmp_path / "r2.env"
    env.write_text("R2_BUCKET=b\n")
    env.chmod(0o600)
    return {"target": 2, "shard_base": "c", "spec": str(spec), "run_prefix": "c", "r2_env_file": str(env),
            "slots": [0, 1], "provider": provider}


@pytest.mark.parametrize("provider", [{"kind": "daytona"},
                                      {"kind": "modal", "image": "i", "secret": "s", "bucket": "b"}])
def test_unpriced_pools_keep_the_mi355x_rate_word_for_word(provider, tmp_path, monkeypatch, capsys):
    from yeto.launch.nc.client import hourly_cost_ucr

    cfg = _priced_cfg(tmp_path, provider)
    assert cli.pool_config(cfg).hourly() == hourly_cost_ucr()
    assert _dry_run_spend(cfg, tmp_path, monkeypatch, capsys) == [MAIN_UNPRICED_SPEND]  # origin/main's line


def test_nc_pool_without_a_price_keeps_the_mi355x_rate(tmp_path):
    from yeto.launch.nc.client import hourly_cost_ucr
    from yeto.launch.nc.lease_pool import PoolConfig

    cfg = {k: v for k, v in _priced_cfg(tmp_path, {}).items() if k not in ("slots", "provider")}
    assert cli.pool_config(dict(cfg, accounts=[1, 4]), allowed=None).hourly() == hourly_cost_ucr()
    assert PoolConfig(target=1, accounts=[1], shard_base="s").hourly() == hourly_cost_ucr()  # unset
    assert PoolConfig(target=1, accounts=[1], shard_base="s", hourly_cost=0).hourly() == 0  # explicit 0
    assert PoolConfig(target=1, accounts=[1], shard_base="s", hourly_cost=5.0).hourly() == 5.0


def test_a_stated_price_is_unchanged(tmp_path, monkeypatch, capsys):
    cfg = _priced_cfg(tmp_path, {"kind": "modal", "image": "i", "secret": "s", "bucket": "b", "price_usd_h": 10})
    assert _dry_run_spend(cfg, tmp_path, monkeypatch, capsys) == [
        "estimated spend: $10.00 per node hour; steady $20.00/h; up to $20.00/h while replacements overlap"]


def test_an_ssh_pool_dry_runs_at_zero_usd(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sp, "load_inventory", lambda _p: [npu_node()])
    cfg = pool_cfg(tmp_path, inventory="inv.yaml")
    assert cli.pool_config(cfg).hourly() == 0
    assert _dry_run_spend(cfg, tmp_path, monkeypatch, capsys) == [
        "estimated spend: $0.00 per node hour; steady $0.00/h; up to $0.00/h while replacements overlap"]



# -- the stop on a node, run in bash with a fake process table (kill / ps are shell functions,
#    /proc/<pid>/environ files under tmp_path): F10 review, a stale log line and a reused PID

FAKE_PRELUDE = r"""
kill() {
  echo "kill $*" >> "$CALLS"
  if [ "$1" = -0 ]; then t="$2"; [ "$t" = "--" ] && t="$3"; [ -e "$FAKE/alive/${t#-}" ]; return; fi
  if [ "$1" = -TERM ] && [ "$2" != "--" ] && [ -n "$ON_TERM" ]; then echo "$ON_TERM" >> "$LOG"; fi
  if [ "$1" = -KILL ]; then rm -f "$FAKE/alive/$2"; fi
  if [ "$2" = "--" ]; then t="$3"; rm -f "$FAKE/alive/${t#-}"; fi
  return 0
}
ps() { cat "$FAKE/args/$4" 2>/dev/null; }
"""


def _fake_node(tmp_path, *, environ_mark, leader_args, log_lines=()):
    p = sp.node_paths(str(tmp_path / "w"), "a")
    Path(p["dir"]).mkdir(parents=True)
    Path(p["logs"]).mkdir(parents=True)
    ours = sp.start_marker("run1", "a", "run1-node-a-1")
    Path(p["marker"]).write_text(ours + "\n")
    Path(p["pid"]).write_text("4242\n")
    Path(p["pgid"]).write_text("4200\n")
    Path(p["log"]).write_text("".join(f"{line}\n" for line in log_lines))
    fake = tmp_path / "fake"
    for d in ("alive", "args", "proc/4242"):
        (fake / d).mkdir(parents=True)
    (fake / "alive" / "4242").touch()
    (fake / "alive" / "4200").touch()
    (fake / "args" / "4242").write_text("python -m yeto.rl.engine.engine_sampler --spec x\n")
    (fake / "args" / "4200").write_text(leader_args + "\n")
    (fake / "proc" / "4242" / "environ").write_bytes(f"A=1\0{sp.MARK_ENV}={environ_mark}\0".encode())
    return p, fake, ours


def _run_stop(p, fake, tmp_path, *, on_term="", grace_s=2):
    script = sp.venv_stop_command(p, grace_s).replace("/proc/", f"{fake}/proc/")
    calls = tmp_path / "calls"
    env = {"PATH": "/usr/bin:/bin", "FAKE": str(fake), "CALLS": str(calls), "LOG": p["log"], "ON_TERM": on_term}
    out = subprocess.run(["bash", "-c", FAKE_PRELUDE + script], capture_output=True, text=True, env=env, timeout=60)
    return out.stdout, calls.read_text() if calls.exists() else ""


def test_a_left_line_of_an_earlier_run_is_not_read_as_this_ones(tmp_path):
    old = f"[engine-sampler] {sp.LEFT_LINE} 280 s for the container to stop"
    p, fake, ours = _fake_node(tmp_path, environ_mark=sp.start_marker("run1", "a", "run1-node-a-1"),
                               leader_args=f"bash -c x {sp.start_marker('run1', 'a', 'run1-node-a-1')}",
                               log_lines=["old run", old])
    out, calls = _run_stop(p, fake, tmp_path)  # the sampler prints nothing new: no early "left"
    assert "kill -TERM 4242" in calls and "YETO_SSH_LEFT" not in out
    # the same stop once the sampler does print its line after the TERM: it is read at once
    p2, fake2, _ = _fake_node(tmp_path / "again", environ_mark=ours, leader_args=f"bash -c x {ours}",
                              log_lines=["old run", old])
    out2, _ = _run_stop(p2, fake2, tmp_path / "again", on_term=old, grace_s=30)
    assert "YETO_SSH_LEFT after=1" in out2


def test_a_reused_pid_of_another_runs_sampler_is_not_killed(tmp_path):
    other = sp.start_marker("run2", "a", "run2-node-a-1")
    p, fake, ours = _fake_node(tmp_path, environ_mark=other, leader_args=f"bash -c x {other}")
    out, calls = _run_stop(p, fake, tmp_path)
    assert "-TERM" not in calls and "-KILL" not in calls  # neither the sampler PID nor the group
    assert "YETO_SSH_TERM" not in out and "YETO_SSH_GROUP_TERM" not in out
    # the state reads absent, not "running", for the other run's process
    state = sp.venv_state_command(p).replace("/proc/", f"{fake}/proc/")
    res = subprocess.run(["bash", "-c", FAKE_PRELUDE + state], capture_output=True, text=True,
                         env={"PATH": "/usr/bin:/bin", "FAKE": str(fake), "CALLS": str(tmp_path / "c2")})
    assert res.stdout.strip() == "absent -"


def test_our_own_sampler_and_group_are_stopped(tmp_path):
    ours = sp.start_marker("run1", "a", "run1-node-a-1")
    p, fake, _ = _fake_node(tmp_path, environ_mark=ours, leader_args=f"bash -c x {ours}")
    out, calls = _run_stop(p, fake, tmp_path, grace_s=1)
    assert "kill -TERM 4242" in calls and "kill -KILL 4242" in calls
    assert "kill -TERM -- -4200" in calls and "YETO_SSH_GROUP_TERM pgid=4200" in out


def test_a_marker_is_plain_words():
    assert sp.start_marker("r", "s", "r-node-a-1") == "yeto-ssh-sampler:r:s:r-node-a-1"
    with pytest.raises(ValueError):
        sp.start_marker("r x", "s", "t")

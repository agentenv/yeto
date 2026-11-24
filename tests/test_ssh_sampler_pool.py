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
    monkeypatch.setattr(sp, "load_profiles", lambda _p: [npu_node()])
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
    # the route's own sampler start (node_runtime: false, kept for one release); the node runtime: below
    ops = sp.SshSamplerPoolOps(cfg, conf=conf(node_runtime=False), run=run, state_dir=tmp_path)
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
    monkeypatch.setattr(sp, "load_profiles", lambda _p: [npu_node()])
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


# -- device defaults by family (unified-sampler-pool 3.1)

K31_SPEC = Path(__file__).resolve().parents[1] / "examples" / "k31" / "sampler-spec.nc-mi355x-vllm.json"
# origin/main's rendering of an Ascend start (podman, cards 0 and 3, every host path / two of them)
MAIN_NPU_ARGV_HEAD = ['podman', 'run', '-d', '--name', 'yeto-sampler-f09-0', '--network', 'host', '--ipc', 'host',
                      '--shm-size', '16g', '--security-opt', 'seccomp=unconfined']
MAIN_NPU_DEVICES_ALL = ['--device', '/dev/davinci0', '--device', '/dev/davinci3', '--device', '/dev/davinci_manager',
                        '--device', '/dev/devmm_svm', '--device', '/dev/hisi_hdc',
                        '-v', '/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro',
                        '-v', '/usr/local/dcmi:/usr/local/dcmi:ro', '-v', '/usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro',
                        '-v', '/usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro',
                        '-v', '/etc/ascend_install.info:/etc/ascend_install.info:ro', '-v', '/etc/vnpu.cfg:/etc/vnpu.cfg:ro']
MAIN_NPU_DEVICES_TWO = ['--device', '/dev/davinci0', '--device', '/dev/davinci3', '--device', '/dev/davinci_manager',
                        '-v', '/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro']
MAIN_NPU_ARGV_TAIL = ['-v', '/srv/yeto/run0:/srv/yeto/run0', '-v', '/srv/yeto/shared:/srv/yeto/shared',
                      '--env-file', '/w/.store.env', '-e', 'K=V', '--entrypoint', 'bash', '3a7d901460ab', '-c',
                      'export PYTHONPATH=/y"${PYTHONPATH:+:$PYTHONPATH}"; export YETO_RL_STORE_S3_URI="s3://${R2_BUCKET}'
                      '/runs/${YETO_NC_RUN_PREFIX}" YETO_RL_STORE_S3_ENDPOINT_URL="$AWS_ENDPOINT_URL"; '
                      'exec python3 -m yeto.rl.engine.engine_sampler --spec /w/a.json']
MAIN_NPU_PREFLIGHT = (
    "set -u; mkdir -p /srv/yeto/run0/sampler /srv/yeto/run0/logs || exit 3; echo YETO_SSH_ARCH $(uname -m); "
    "for f in /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc /usr/local/Ascend/driver /usr/local/dcmi "
    "/usr/local/bin/npu-smi /usr/local/sbin/npu-smi /etc/ascend_install.info /etc/vnpu.cfg; do [ -e $f ] && "
    "echo YETO_SSH_HOSTPATH $f; done; echo YETO_SSH_FREE_GIB $(df -Pk /srv/yeto/run0 | awk 'NR==2{printf \"%d\", "
    "$4/1048576}'); echo YETO_SSH_CARDS_BEGIN; npu-smi info 2>&1; echo YETO_SSH_CARDS_END rc=$?")


def test_an_ascend_start_renders_as_on_main():
    c = conf(devices=[0, 3])
    for host_paths, devices in ((None, MAIN_NPU_DEVICES_ALL),
                                (["/dev/davinci_manager", "/usr/local/Ascend/driver"], MAIN_NPU_DEVICES_TWO)):
        argv = sp.container_run_argv(c, npu_node(), name="yeto-sampler-f09-0", image="3a7d901460ab", yeto_dir="/y",
                                     spec_path="/w/a.json", env={"K": "V"}, env_file="/w/.store.env", cards=[0, 3],
                                     host_paths=host_paths)
        assert argv == MAIN_NPU_ARGV_HEAD + devices + MAIN_NPU_ARGV_TAIL
    assert sp.preflight_script(npu_node(), c) == MAIN_NPU_PREFLIGHT
    assert sp.NPU_HOST_DEVICES == ("/dev/davinci_manager", "/dev/devmm_svm", "/dev/hisi_hdc")


def cuda_node(**kw):
    raw = {"name": "node-a", "host": "node-a", "device_family": "cuda", "cpu_arch": "x86_64", "card": "H200",
           "cards_per_node": 8, "hbm_gib": 141, "host_memory_gib": 1024, **kw}
    return parse_node(raw, index=0)


def rocm_node(**kw):
    return cuda_node(device_family="rocm", card="MI355X", hbm_gib=288, **kw)


def family_conf(node, **section):
    sec = provider_section(**{"container": {"engine": "docker", "image": "img"}, **section})
    return sp.parse_config({"provider": sec}, inventory=[node])


def _env(node, c, tmp_path):
    ops = sp.SshSamplerPoolOps(pool_cfg(tmp_path), conf=c, run=FakeRun(), state_dir=tmp_path)
    return ops.sampler_env(node, SimpleNamespace(shard="f09-0"))


def test_cuda_containers_get_the_gpus_and_no_rocm_devices(tmp_path):
    node = cuda_node()
    c = family_conf(node, devices=[2, 3])
    argv = sp.container_run_argv(c, node, name="n", image="img", yeto_dir="/y", spec_path="/s", env={},
                                 env_file=None, cards=sp.SshSamplerPoolOps(pool_cfg(tmp_path), conf=c, run=FakeRun(),
                                                                           state_dir=tmp_path).cards(node))
    joined = " ".join(argv)
    assert '--gpus "device=2,3"' in joined and "/dev/kfd" not in joined and "davinci" not in joined
    env = _env(node, c, tmp_path)
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1" and env["YETO_DEVICE_FAMILY"] == "cuda"
    assert "HIP_VISIBLE_DEVICES" not in env and "ASCEND_RT_VISIBLE_DEVICES" not in env
    podman = sp.parse_config({"provider": provider_section(container={"engine": "podman", "image": "img"})},
                             inventory=[node])
    cdi = sp.container_run_argv(podman, node, name="n", image="img", yeto_dir="/y", spec_path="/s", env={},
                                env_file=None, cards=[0])
    assert "--device nvidia.com/gpu=0" in " ".join(cdi)
    venv = family_conf(node, runtime="venv", container=None, venv={"python": "/e/bin/python"}, devices=[2, 3])
    assert _env(node, venv, tmp_path)["CUDA_VISIBLE_DEVICES"] == "2,3"  # a host process: host numbers


def test_rocm_containers_get_kfd_dri_video_and_seccomp(tmp_path):
    node = rocm_node()
    c = family_conf(node, devices=None)
    argv = " ".join(sp.container_run_argv(c, node, name="n", image="img", yeto_dir="/y", spec_path="/s", env={},
                                          env_file=None, cards=list(range(8))))
    assert ("--device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined" in argv)
    assert "--cap-add SYS_PTRACE" in argv and "--ulimit memlock=-1" in argv  # the NC route's flags
    assert argv.count("--shm-size") == 1 and "--shm-size 16g" in argv  # the pool's container.shm_size
    assert "--gpus" not in argv
    env = _env(node, c, tmp_path)
    assert "HIP_VISIBLE_DEVICES" not in env and env["YETO_DEVICE_FAMILY"] == "rocm"  # no devices: not numbered
    assert _env(node, family_conf(node, devices=[1, 6]), tmp_path)["HIP_VISIBLE_DEVICES"] == "1,6"
    site = rocm_node(docker_args=["--device", "/dev/kfd", "--device", "/dev/dri/renderD128"])
    argv = sp.container_run_argv(family_conf(site), site, name="n", image="img", yeto_dir="/y", spec_path="/s",
                                 env={}, env_file=None, cards=[0])
    assert "--device /dev/kfd --device /dev/dri/renderD128" in " ".join(argv) and "video" not in " ".join(argv)


AMD_STATIC = json.dumps([{"gpu": i, "asic": {"market_name": "AMD Instinct MI355X"},
                          "vram": {"size": {"value": 294896, "unit": "MB"}}} for i in range(8)])
AMD_IDLE_OLD = json.dumps([{"gpu": i, "process_list": [{"process_info": "No running processes detected"}]}
                           for i in range(8)])
AMD_BUSY_OLD = json.dumps([{"gpu": 0, "process_list": [{"process_info": "No running processes detected"}]},
                           {"gpu": 5, "process_list": [{"process_info": {"name": "python3", "pid": 4242}}]}])
AMD_BUSY_NEW = json.dumps([{"gpu": 1, "process_list": [{"name": "python3", "pid": 77, "mem": 1}]},
                           {"gpu": 2, "process_list": []}])
NV_IDLE = "0, 00000000:18:00.0\n1, 00000000:2A:00.0\nYETO_SSH_APPS\n"
NV_BUSY = NV_IDLE + "00000000:2A:00.0, 31337\n"


def test_busy_cards_from_nvidia_smi_and_amd_smi():
    assert sp.busy_cuda(NV_IDLE) == [] and sp.busy_cuda(NV_BUSY) == [1]
    assert sp.busy_cuda("bash: nvidia-smi: command not found\nYETO_SSH_APPS\n") is None
    assert sp.busy_amd(AMD_IDLE_OLD) == [] and sp.busy_amd(AMD_BUSY_OLD) == [5] and sp.busy_amd(AMD_BUSY_NEW) == [1]
    assert sp.busy_amd("bash: amd-smi: command not found") is None


def _preflight_text(cards, idle=None, paths=()):
    out = "YETO_SSH_ARCH x86_64\nYETO_SSH_FREE_GIB 300\n" + "".join(f"YETO_SSH_HOSTPATH {p}\n" for p in paths)
    out += f"YETO_SSH_CARDS_BEGIN\n{cards}\nYETO_SSH_CARDS_END rc=0\n"
    if idle is not None:
        out += f"YETO_SSH_IDLE_BEGIN\n{idle}\nYETO_SSH_IDLE_END rc=0\n"
    return out


def test_require_idle_reads_nvidia_smi_on_cuda():
    node = cuda_node()
    assert "YETO_SSH_IDLE_BEGIN" not in sp.preflight_script(node, family_conf(node))  # opt-in off ascend
    on = {"require_idle": True}
    script = sp.preflight_script(node, family_conf(node, preflight=on))
    assert "nvidia-smi --query-compute-apps=gpu_bus_id,pid" in script and "YETO_SSH_HOSTPATH" not in script
    assert "npu-smi" not in script and "amd-smi" not in script
    cards = "\n".join(f"{i}, NVIDIA H200, 143771" for i in range(8))
    facts = sp.check_preflight(node, family_conf(node, devices=[0], preflight=on), _preflight_text(cards, NV_BUSY))
    assert facts["busy"] == []  # card 1 is not this pool's
    with pytest.raises(sp.PreflightError, match=r"cards \[1\] run other processes \(nvidia-smi compute apps\)"):
        sp.check_preflight(node, family_conf(node, devices=[1], preflight=on), _preflight_text(cards, NV_BUSY))
    sp.check_preflight(node, family_conf(node, devices=[1]), _preflight_text(cards, NV_BUSY))  # default: off
    unread = sp.check_preflight(node, family_conf(node, devices=[1], preflight=on), _preflight_text(cards, "no driver"))
    assert unread["busy"] is None and "not checked" in unread["idle"]
    off = family_conf(node, preflight={"require_idle": False})
    assert "YETO_SSH_IDLE_BEGIN" not in sp.preflight_script(node, off)


def test_require_idle_reads_amd_smi_and_a_rocm_container_needs_kfd():
    node = rocm_node()
    c = family_conf(node, devices=[5], preflight={"require_idle": True})
    script = sp.preflight_script(node, c)
    assert "amd-smi static --json || rocm-smi" in script and "amd-smi process --json" in script
    assert "for f in /dev/kfd /dev/dri;" in script
    ok = _preflight_text(AMD_STATIC, AMD_IDLE_OLD, paths=("/dev/kfd", "/dev/dri"))
    assert sp.check_preflight(node, c, ok)["busy"] == []
    with pytest.raises(sp.PreflightError, match=r"cards \[5\] run other processes \(amd-smi process\)"):
        sp.check_preflight(node, c, ok.replace(AMD_IDLE_OLD, AMD_BUSY_OLD))
    with pytest.raises(sp.PreflightError, match="/dev/kfd missing on the host"):
        sp.check_preflight(node, c, _preflight_text(AMD_STATIC, AMD_IDLE_OLD, paths=("/dev/dri",)))
    venv = family_conf(node, runtime="venv", container=None, venv={"python": "/e/bin/python"})
    sp.check_preflight(node, venv, _preflight_text(AMD_STATIC, AMD_IDLE_OLD))  # a host process needs no flags


def test_auto_image_picks_per_node_and_the_dry_run_refuses_what_no_image_serves(tmp_path):
    from yeto.rl import sampler_pins as pins

    spec = json.loads(K31_SPEC.read_text())
    a = rocm_node(name="mi355x", host="mi355x")
    b = cuda_node(name="b300", host="b300", card="B300", hbm_gib=288)
    sec = provider_section(nodes=["mi355x", "b300"], container={"engine": "docker", "image": "auto"})
    c = sp.parse_config({"provider": sec}, inventory=[a, b])
    assert sp.node_image(c, c.nodes[0], spec) == pins.VLLM_SAMPLER_IMAGE_ROCM.removeprefix("docker:")
    # K3.1 in place: the vLLM CUDA image restarts, so the B300 node has no image for this spec
    problems = sp.image_problems(c, spec)
    assert len(problems) == 1 and "node b300" in problems[0] and "load in_place" in problems[0]
    restart = dict(spec, engine=dict(spec["engine"], server=dict(spec["engine"]["server"], in_place_reload=False)))
    assert sp.image_problems(c, restart) == []
    assert sp.node_image(c, c.nodes[1], restart) == pins.VLLM_SAMPLER_IMAGE_CUDA.removeprefix("docker:")
    # a named catalog image is held to the node too; a site image is the site's
    named = sp.parse_config({"provider": provider_section(nodes=["b300"], container={
        "engine": "docker", "image": pins.VLLM_SAMPLER_IMAGE_ROCM.removeprefix("docker:")})}, inventory=[a, b])
    assert "it is a rocm image" in sp.image_problems(named, restart)[0]
    assert sp.image_problems(family_conf(cuda_node()), spec) == []


def test_the_dry_run_reports_an_incompatible_image(tmp_path, monkeypatch):
    from yeto.rl import sampler_pins as pins

    monkeypatch.setattr(sp, "load_profiles", lambda _p: [cuda_node()])
    cfg = pool_cfg(tmp_path, inventory="inv.yaml",
                   container={"engine": "docker", "image": pins.VLLM_SAMPLER_IMAGE_ROCM.removeprefix("docker:")})
    assert any("node node-a" in p and "rocm image" in p for p in sp.offline_problems(cfg))


# -- run.json shorthand (unified-sampler-pool 3.4)


def test_a_run_config_may_name_provider_ssh_and_its_pool_dry_runs(tmp_path, monkeypatch, capsys):
    from yeto.launch.nc.provider import provider_kind
    from yeto.launch.rollout import pools as rollout_pools

    assert "ssh" in rollout_pools.PROVIDER_KINDS
    assert set(rollout_pools.provider_kinds()) == set(providers.kinds())
    monkeypatch.setattr(sp, "load_profiles", lambda _p: [npu_node()])
    pool = tmp_path / "pool.ssh.json"
    pool.write_text(json.dumps(pool_cfg(tmp_path, inventory="inv.yaml")))
    run = {"samplers": [{"name": "own", "provider": "ssh", "config": str(pool)}]}
    (ref,) = rollout_pools.resolve(run, tmp_path / "state")
    assert ref.config == str(pool) and ref.kind() == "ssh" and provider_kind(ref.load()) == "ssh"
    assert _dry_run_spend(ref.load(), tmp_path, monkeypatch, capsys) == [
        "estimated spend: $0.00 per node hour; steady $0.00/h; up to $0.00/h while replacements overlap"]
    with pytest.raises(ValueError, match="provider 'ssh' but the pool config's provider is 'nc'"):
        nc_pool = tmp_path / "pool.nc.json"
        nc_pool.write_text(json.dumps({"target": 1, "shard_base": "x"}))
        rollout_pools.resolve({"samplers": [{"provider": "ssh", "config": str(nc_pool)}]}, tmp_path / "state")
    with pytest.raises(ValueError, match="is not one of daytona, modal, nc, sky, ssh"):
        rollout_pools.resolve({"samplers": [{"provider": "k8s", "config": str(pool)}]}, tmp_path / "state")


def test_rollout_commands_accept_provider_ssh():
    import argparse

    from yeto.launch.rollout import entry

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    entry.add_parser(sub)
    args = parser.parse_args(["rollout", "status", "--config", "run.json", "--provider", "ssh"])
    assert args.provider == "ssh"


def test_nodes_the_pool_cannot_ssh_to_are_refused():
    from yeto.launch.node_inventory import NodeProfile, Reach

    outbound = NodeProfile.from_node(cuda_node(), reach=Reach("outbound"))
    with pytest.raises(ValueError, match=r"nodes \['node-a \(outbound\)'\] take no SSH connection"):
        family_conf(outbound)
    family_conf(NodeProfile.from_node(cuda_node(), reach=Reach("relay", "relay.example:2222")))  # task 3.2's
    jump = NodeProfile.from_node(cuda_node(), reach=Reach("jump", "ops@bastion"))
    assert family_conf(jump).nodes[0].ssh_options() == ["-o", "ProxyJump=ops@bastion"]


# -- review of the defaults: existing kind ssh users keep their behaviour


def test_require_idle_defaults_on_for_ascend_only():
    assert conf().require_idle is None and sp.idle_required(npu_node(), conf())
    assert not sp.idle_required(cuda_node(), family_conf(cuda_node()))
    assert not sp.idle_required(rocm_node(), family_conf(rocm_node()))
    assert not sp.idle_required(npu_node(), conf(preflight={"require_idle": False}))


def test_a_node_with_docker_args_starts_exactly_as_on_main(tmp_path):
    flags = ["--gpus", '"device=4,5,6,7"', "--shm-size", "32g"]
    node = cuda_node(docker_args=flags)
    c = family_conf(node, devices=[4, 5, 6, 7])
    argv = sp.container_run_argv(c, node, name="n", image="img", yeto_dir="/y", spec_path="/s", env={},
                                 env_file=None, cards=[4, 5, 6, 7])
    assert argv[argv.index("--gpus"):argv.index("--gpus") + 4] == flags  # verbatim, its --shm-size kept
    env = _env(node, c, tmp_path)
    assert "CUDA_VISIBLE_DEVICES" not in env and "YETO_DEVICE_FAMILY" not in env
    npu = npu_node(docker_args=["--device", "/dev/davinci4"])
    assert sp.device_env(npu, conf(devices=[4])) == {"ASCEND_RT_VISIBLE_DEVICES": "4"}  # as on main


def test_ascend_env_is_unchanged_and_a_venv_keeps_its_own_card_variable(tmp_path):
    assert sp.device_env(npu_node(), conf(devices=None)) == {}
    assert sp.device_env(npu_node(), conf(devices=[0, 3])) == {"ASCEND_RT_VISIBLE_DEVICES": "0,3"}
    venv = family_conf(cuda_node(), runtime="venv", container=None, venv={"python": "/e/bin/python"}, devices=None)
    assert sp.device_env(cuda_node(), venv) == {"YETO_DEVICE_FAMILY": "cuda"}  # no CUDA_VISIBLE_DEVICES over source


def test_a_restart_stops_its_own_shard_before_the_idle_check(tmp_path, monkeypatch):
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
    ops = sp.SshSamplerPoolOps(pool_cfg(tmp_path), conf=conf(), run=run, state_dir=tmp_path)
    m = SimpleNamespace(account=0, ticket=ops.order(0), shard="f09-0", data_slot=None, replaces_shard=None,
                        predecessor_lost=False)
    ops.restart(m, ["f09-0"])
    cmds = [c[0][-1] for c in run.calls if c[0][0] == "ssh"]
    stop = next(i for i, c in enumerate(cmds) if "kill --signal TERM" in c)
    pre = next(i for i, c in enumerate(cmds) if "YETO_SSH_CARDS_BEGIN" in c)
    assert stop < pre


def test_an_unchecked_idle_is_an_event(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(sp.SshSamplerPoolOps, "event", lambda self, name, **d: events.append((name, d)))
    node = cuda_node()
    c = family_conf(node, devices=[0], preflight={"require_idle": True})
    ops = sp.SshSamplerPoolOps(pool_cfg(tmp_path), conf=c, run=FakeRun(), state_dir=tmp_path)
    cards = "\n".join(f"{i}, NVIDIA H200, 143771" for i in range(8))
    monkeypatch.setattr(ops, "ssh", lambda *a, **k: SimpleNamespace(returncode=0, stderr="",
                                                                    stdout=_preflight_text(cards, "no driver")))
    monkeypatch.setattr(ops, "upload_code", lambda node: (_ for _ in ()).throw(RuntimeError("stop here")))
    m = SimpleNamespace(account=0, ticket=ops.order(0), shard="f09-0")
    with pytest.raises(RuntimeError, match="stop here"):
        ops.start(m, ["f09-0"])
    assert events and events[0][0] == "ssh_pool_idle_unchecked" and "not checked" in events[0][1]["detail"]


def test_idle_parsers_do_not_read_a_failure_as_idle():
    assert sp.busy_amd(AMD_BUSY_OLD + "\nWARNING: amd-smi: unknown field\n") == [5]  # stderr after the JSON
    assert sp.busy_cuda(NV_IDLE + "Failed to initialize NVML: Driver/library version mismatch\n") is None


def test_auto_image_takes_no_pull_ref():
    with pytest.raises(ValueError, match="drop pull_ref"):
        family_conf(cuda_node(), container={"engine": "docker", "image": "auto", "pull_ref": "x@sha256:1"})


def test_a_spec_for_another_platform_or_card_is_refused_per_node():
    spec = json.loads(K31_SPEC.read_text())  # platform rocm, device_family rocm, card_model MI355X
    b300 = cuda_node(card="B300", hbm_gib=288)
    problems = sp.spec_node_problems(family_conf(b300), spec)
    assert any("cuda node, the spec's platform is rocm" in p for p in problems)
    assert any("B300 cards, the spec's engine.server.card_model is MI355X" in p for p in problems)
    assert sp.spec_node_problems(family_conf(rocm_node()), spec) == []
    assert sp.spec_node_problems(family_conf(b300), {"engine": {"server": {"card_model": "<card>"}}}) == []

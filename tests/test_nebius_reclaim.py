"""Nebius preemptible reclaim path on CPU: host hook script, notice-file poller,
droppable listener wiring, launcher install hook (S19 real-spot experiment, option C)."""

from __future__ import annotations

import os
import subprocess
import threading
from types import SimpleNamespace

from yeto import launcher
from yeto.cloud import droppable as dr
from yeto.cloud import nebius_reclaim as nr
from yeto.cloud import preemption as pre


def _env(tmp_path):
    env = dict(dr.role_env("droppable", "nebius", "eu-north1"))
    env[nr.NOTICE_ENV] = str(tmp_path / "notice")
    env[nr.ACK_ENV] = str(tmp_path / "ack")
    return env


def test_nebius_droppable_reclaim_leaves_writes_event_and_acks(tmp_path):
    events, left = [], []
    env = _env(tmp_path)
    (tmp_path / "notice").write_text("stale\n")  # left over from a previous container life
    poller = dr.install_reclaim_listener("1", leave=lambda: left.append(1) or True,
                                         emit=lambda e, **f: events.append((e, f)),
                                         environ=env, start=False)
    assert isinstance(poller, nr.NoticeFilePoller)
    assert not (tmp_path / "notice").exists()  # stale notice removed at install
    assert poller.poll_once() is None and left == []
    poller.clock = lambda: 1000.0
    (tmp_path / "notice").write_text("990.5\n")
    poller.poll_once()
    assert left == [1] and (tmp_path / "ack").exists()
    name, ev = events[0]
    assert name == pre.RECLAIM_EVENT
    assert ev["cloud"] == "nebius" and ev["role"] == "droppable" and ev["billing"] == "spot"
    assert ev["source"] == nr.SOURCE and ev["notified_at"] == 990.5
    assert ev["deadline_s"] == 60.0 and ev["leave_confirmed"] is True and ev["saved"] is False
    assert poller.poll_once() is None and len(events) == 1  # fires once


def test_poller_acks_even_if_handler_raises(tmp_path):
    def boom(_):
        raise RuntimeError("x")

    p = nr.NoticeFilePoller("1", boom, notice_path=str(tmp_path / "n"), ack_path=str(tmp_path / "a"))
    (tmp_path / "n").write_text("1\n")
    try:
        p.poll_once()
    except RuntimeError:
        pass
    assert (tmp_path / "a").exists()


def test_poller_thread_fires(tmp_path):
    got = threading.Event()
    p = nr.NoticeFilePoller("1", lambda n: got.set(), notice_path=str(tmp_path / "n"),
                            ack_path=str(tmp_path / "a"), interval_s=0.02)
    p.start()
    (tmp_path / "n").write_text("5\n")
    assert got.wait(2.0)
    p.stop()


def test_parse_notice():
    assert nr.parse_notice("12.5\n", now=20) == 12.5
    assert nr.parse_notice("", now=20) == 20
    assert nr.parse_notice("garbage", now=20) == 20
    assert nr.parse_notice("999999", now=20) == 20  # future time: use now


def test_listener_other_clouds_unchanged():
    noop = dict(leave=lambda: True, emit=lambda *a, **k: None, start=False)
    assert dr.install_reclaim_listener("1", environ=dr.role_env("droppable", "verda", None), **noop) is None
    assert dr.install_reclaim_listener("0", environ=dr.role_env("anchor", "nebius", None), **noop) is None


def _fake_docker(tmp_path, ack_after: int):
    """A `docker` on PATH: `exec <c> sh -c CMD` runs CMD locally; `exec <c> test -e F`
    succeeds after ``ack_after`` calls."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    count = tmp_path / "count"
    (bin_dir / "docker").write_text(
        "#!/bin/bash\n"
        'if [ "$1" = info ]; then exit 0; fi\n'
        'shift 2\n'
        'if [ "$1" = sh ]; then exec "$@"; fi\n'
        f'n=$(cat {count} 2>/dev/null || echo 0); n=$((n+1)); echo $n > {count}\n'
        f'[ $n -ge {ack_after} ]\n')
    (bin_dir / "docker").chmod(0o755)
    for tool in ("logger", "sleep"):
        (bin_dir / tool).write_text("#!/bin/bash\nexit 0\n")
        (bin_dir / tool).chmod(0o755)
    return bin_dir


def test_hook_script_writes_notice_and_waits_for_ack(tmp_path):
    bin_dir = _fake_docker(tmp_path, ack_after=3)
    notice = tmp_path / "notice"
    script = tmp_path / "hook.sh"
    script.write_text(nr.hook_script(notice=str(notice), ack=str(tmp_path / "ack"), wait_s=5))
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    r = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=30)
    assert r.returncode == 0
    assert float(notice.read_text().split()[0]) > 0
    assert (tmp_path / "count").read_text().strip() == "3"  # stopped polling at the ack


def test_hook_script_never_blocks_shutdown_without_ack(tmp_path):
    bin_dir = _fake_docker(tmp_path, ack_after=10**6)
    script = tmp_path / "hook.sh"
    script.write_text(nr.hook_script(notice=str(tmp_path / "n"), ack=str(tmp_path / "a"), wait_s=4))
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    r = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=30)
    assert r.returncode == 0 and (tmp_path / "count").read_text().strip() == "4"


def test_unit_orders_before_docker_on_shutdown():
    u = nr.unit_text()
    assert "After=docker.service" in u and f"ExecStop={nr.HOOK_BIN}" in u
    assert "RemainAfterExit=yes" in u and "TimeoutStopSec=55" in u
    assert nr.ACK_WAIT_S + 10 < 60 + 10  # ack wait inside the 60 s notice


def test_install_host_hook_uses_runner_and_never_raises():
    calls, logs = [], []

    class R:
        def run(self, cmd, **kw):
            calls.append(cmd)
            return 0, "active\n", ""

    assert nr.install_host_hook(object(), name="c", runner=R(), log=logs.append) is True
    assert nr.UNIT in calls[0] and "systemctl enable --now" in calls[0]

    class Bad:
        def run(self, cmd, **kw):
            raise OSError("ssh down")

    assert nr.install_host_hook(object(), name="c", runner=Bad(), log=logs.append) is False
    assert "NOT installed" in logs[-1]


def test_launcher_installs_hook_only_for_nebius_droppable(monkeypatch):
    seen = []
    monkeypatch.setattr(nr, "install_host_hook", lambda h, name="": seen.append((h, name)) or True)
    neb = SimpleNamespace(envs=dr.role_env("droppable", "nebius", "eu-north1"))
    assert launcher.maybe_install_nebius_reclaim_hook(neb, (7, "H"), "c1") is True
    assert seen == [("H", "c1")]
    for envs in (dr.role_env("anchor", "nebius", None), dr.role_env("droppable", "aws", None), {}):
        assert launcher.maybe_install_nebius_reclaim_hook(SimpleNamespace(envs=envs), (7, "H"), "c") is None
    assert launcher.maybe_install_nebius_reclaim_hook(neb, (7, None), "c2") is False
    assert len(seen) == 1

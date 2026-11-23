"""Ctrl-C (SIGINT) of ``yeto run up``: the SIGTERM exit, announced, with a forced second.

In-process tests drive ``up.main`` over two fake pools (tests.test_rollout_entry) and
send the signal from the monitor; the subprocess tests run ``up.main`` in a child (a
second Ctrl-C ends the process with ``os._exit``, and a child started with SIGINT
ignored is the background ``run up`` of a launcher script that only SIGTERM ended).
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from tests.test_rollout_entry import _main_harness
from yeto.launch.pool import controller as ctl
from yeto.rl import up

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("YETO_NC_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("YETO_NC_ACCOUNTS", "1,2,3,4")
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "rl-up")
    old = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    yield
    for sig, handler in old.items():
        signal.signal(sig, handler)


def _signal_in_monitor(run, monkeypatch, sig):
    def monitor():
        os.kill(os.getpid(), sig)
        time.sleep(5)  # the handler raises out of the monitor's wait
        raise AssertionError(f"signal {sig} did not end the monitor")

    monkeypatch.setattr(run, "monitor", monitor)


def _harness_dir(tmp_path, name):
    path = tmp_path / name
    path.mkdir()
    return path


def test_ctrl_c_in_the_monitor_takes_the_sigterm_exit(tmp_path, monkeypatch, capsys):
    """Same exit code, same drain and release of the in-process pools as SIGTERM."""
    term_run, term_providers, term_args = _main_harness(_harness_dir(tmp_path, "term"), monkeypatch)
    _signal_in_monitor(term_run, monkeypatch, signal.SIGTERM)
    with pytest.raises(SystemExit) as term:
        up.main(term_args)
    term_out = capsys.readouterr().out

    int_run, int_providers, int_args = _main_harness(_harness_dir(tmp_path, "int"), monkeypatch)
    _signal_in_monitor(int_run, monkeypatch, signal.SIGINT)
    with pytest.raises(SystemExit) as ctrl_c:  # not KeyboardInterrupt: no traceback
        up.main(int_args)
    int_out = capsys.readouterr().out

    assert term.value.code == ctrl_c.value.code == 128 + signal.SIGTERM
    for run, providers in ((term_run, term_providers), (int_run, int_providers)):
        assert run.controller.stopped == "process exit" and run.controller not in ctl.active()
        assert all(not p.live() for p in run.controller.pools.values())
        assert {n: len(p.orders) for n, p in providers.items()} == {"nc": 1, "modal": 1}
    assert ({n: len(p.released) for n, p in term_providers.items()}
            == {n: len(p.released) for n, p in int_providers.items()})
    assert "Ctrl-C" not in term_out
    line = [ln for ln in int_out.splitlines() if "Ctrl-C:" in ln]
    assert len(line) == 1, int_out
    assert "stopping the run up monitor, as on SIGTERM" in line[0] and "draining and releasing" in line[0]
    assert signal.getsignal(signal.SIGTERM) is up._exit_on_sigterm  # SIGTERM unchanged


def test_ctrl_c_during_the_exit_drain_is_held_like_sigterm(tmp_path, monkeypatch, capsys):
    """The first Ctrl-C while the drain runs does what SIGTERM does there: logged, the
    drain finishes, main returns its code."""
    run, _, args = _main_harness(tmp_path, monkeypatch)
    real_start = run.start_controller

    def start(**kw):
        c = real_start(**kw)
        real_stop = c.stop

        def stop(why, **kw2):
            os.kill(os.getpid(), signal.SIGINT)
            return real_stop(why, **kw2)

        monkeypatch.setattr(c, "stop", stop)
        return c

    monkeypatch.setattr(run, "start_controller", start)
    assert up.main(args) == 0
    out = capsys.readouterr().out
    assert "Ctrl-C: stopping the run up monitor" in out
    assert "SIGTERM during the exit drain" in out
    assert all(not pool.live() for pool in run.controller.pools.values())


def test_a_second_ctrl_c_forces_the_exit(tmp_path, monkeypatch, capsys):
    run, _, args = _main_harness(tmp_path, monkeypatch)
    _signal_in_monitor(run, monkeypatch, signal.SIGINT)
    real_start = run.start_controller

    class Forced(BaseException):
        pass

    def fake_exit(code):
        raise Forced(code)

    def start(**kw):
        c = real_start(**kw)

        def stop(why, **kw2):  # a drain that hangs; the operator presses Ctrl-C again
            os.kill(os.getpid(), signal.SIGINT)
            time.sleep(5)
            raise AssertionError("the second Ctrl-C did not force the exit")

        monkeypatch.setattr(c, "stop", stop)
        return c

    monkeypatch.setattr(run, "start_controller", start)
    monkeypatch.setattr(up.os, "_exit", fake_exit)
    with pytest.raises(Forced) as forced:
        up.main(args)
    assert forced.value.args == (up.FORCED_EXIT_CODE,) and up.FORCED_EXIT_CODE == 130
    out = capsys.readouterr().out
    assert "second Ctrl-C: forcing the exit now" in out and "yeto run down" in out


def test_main_restores_sigint_when_no_ctrl_c_came(tmp_path, monkeypatch):
    run, _, args = _main_harness(tmp_path, monkeypatch)
    before = signal.getsignal(signal.SIGINT)
    assert up.main(args) == 0
    assert signal.getsignal(signal.SIGINT) is before


# -- in a child process: the real signal dispositions -----------------------------------

CHILD = textwrap.dedent("""
    import os, sys, time, types
    from pathlib import Path
    import pytest
    tmp, mode, drain = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    os.environ["YETO_NC_STATE_DIR"] = str(tmp / "state")
    os.environ["YETO_NC_ACCOUNTS"] = "1,2,3,4"
    from tests.test_rollout_entry import _main_harness, _two_pools
    from yeto.rl import up
    up.STATE_ROOT = tmp / "rl-up"
    mp = pytest.MonkeyPatch()
    marker = tmp / "released.txt"

    def note(what):
        with open(marker, "a") as fh:
            fh.write(what + "\\n")

    def monitor():
        print("READY", flush=True)
        while True:
            time.sleep(0.05)

    if mode == "in_process":
        run, providers, args = _main_harness(tmp, mp)
        if drain == "hang":
            real_start = run.start_controller
            def start(**kw):
                c = real_start(**kw)
                def stop(why, **kw2):
                    print("DRAINING", flush=True)
                    while True:
                        time.sleep(0.05)
                mp.setattr(c, "stop", stop)
                return c
            mp.setattr(run, "start_controller", start)
    else:  # controller: subprocess (the default): the pool loops are processes of their own
        run_cfg, _, _ = _two_pools(tmp)
        run = up.Run(dict(run_cfg))
        mp.setattr(up, "Run", lambda cfg: run)
        mp.setattr(up, "load_config", lambda path: {})
        mp.setattr(run, "sandbox_prepare", lambda: True)
        mp.setattr(run, "pool", lambda: None)
        mp.setattr(run, "trainer", lambda: None)
        mp.setattr(run, "dashboard", lambda: None)
        args = types.SimpleNamespace(config="x", rl_command="up", skip_preflight=True, dry_run=False,
                                     no_monitor=False, trainer_now=True, allow_after_forced_stop=False)
    mp.setattr(run, "monitor", monitor)
    mp.setattr(run, "down", lambda: note("down") or True)
    mp.setattr(run, "_cmd", lambda argv, **kw: note(" ".join(argv)) or 0)
    try:
        rc = up.main(args)
    finally:
        if mode == "in_process":
            live = sum(len(p.live()) for p in run.controller.pools.values())
            print(f"LIVE {live} STOPPED {run.controller.stopped}", flush=True)
    sys.exit(rc)
""")


def _child(tmp_path, mode, *, drain="ok", ignore_sigint=False):
    script = tmp_path / "child.py"
    script.write_text(CHILD)
    env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONUNBUFFERED="1")

    def preexec():
        if ignore_sigint:  # what `cmd &` in a non-interactive shell gives the job
            signal.signal(signal.SIGINT, signal.SIG_IGN)

    return subprocess.Popen([sys.executable, str(script), str(tmp_path), mode, drain], cwd=REPO, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, preexec_fn=preexec)


def _wait_for(proc, word, timeout=60.0):
    deadline = time.monotonic() + timeout
    seen = []
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if not line:
            break
        seen.append(line)
        if line.strip() == word:
            return seen
    proc.kill()
    raise AssertionError(f"child never printed {word}: {''.join(seen)}{proc.stderr.read()}")


def _signal_child(tmp_path, mode, sig, *, ignore_sigint=False):
    tmp = tmp_path / f"{mode}-{signal.Signals(sig).name}-{ignore_sigint}"
    tmp.mkdir()
    proc = _child(tmp, mode, ignore_sigint=ignore_sigint)
    head = _wait_for(proc, "READY")
    proc.send_signal(sig)
    try:
        out, err = proc.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        raise AssertionError(f"{signal.Signals(sig).name} did not end run up ({mode}): {out}{err}") from None
    released = (tmp / "released.txt").read_text() if (tmp / "released.txt").exists() else ""
    return proc.returncode, "".join(head) + out, err, released


@pytest.mark.parametrize("mode", ["subprocess", "in_process"])
@pytest.mark.parametrize("ignore_sigint", [False, True], ids=["foreground", "sigint-ignored-at-start"])
def test_ctrl_c_ends_run_up_like_sigterm(tmp_path, mode, ignore_sigint):
    """A ``run up`` started with SIGINT ignored (the background job of a launcher script)
    ignored Ctrl-C / ``kill -INT`` for good; now both ways it ends as SIGTERM ends it:
    the same exit status, nothing released that SIGTERM keeps, no traceback."""
    term = _signal_child(tmp_path, mode, signal.SIGTERM, ignore_sigint=ignore_sigint)
    ctrl_c = _signal_child(tmp_path, mode, signal.SIGINT, ignore_sigint=ignore_sigint)
    code, out, err, released = ctrl_c
    assert code == term[0], (ctrl_c, term)
    assert code == (-signal.SIGTERM if mode == "subprocess" else 128 + signal.SIGTERM)
    assert released == term[3] == ""  # no down, no pool-stop: SIGTERM never ran them either
    assert "Traceback" not in err and "KeyboardInterrupt" not in err, err
    assert sum("Ctrl-C:" in ln for ln in out.splitlines()) == 1, out
    if mode == "subprocess":
        assert "the trainer and the dashboard running in their own sessions" in out
    else:
        assert "LIVE 0 STOPPED process exit" in out and "LIVE 0 STOPPED process exit" in term[1]


def test_a_second_ctrl_c_during_a_hung_drain_ends_the_process(tmp_path):
    proc = _child(tmp_path, "in_process", drain="hang")
    _wait_for(proc, "READY")
    proc.send_signal(signal.SIGINT)
    _wait_for(proc, "DRAINING")
    proc.send_signal(signal.SIGTERM)  # held during the drain, as before
    time.sleep(0.5)
    assert proc.poll() is None
    proc.send_signal(signal.SIGINT)
    out, err = proc.communicate(timeout=60)
    assert proc.returncode == 130
    assert "SIGTERM during the exit drain" in out
    assert "second Ctrl-C: forcing the exit now" in out and "yeto run down" in out
    assert "Traceback" not in err, err

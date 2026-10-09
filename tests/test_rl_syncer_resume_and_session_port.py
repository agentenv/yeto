"""S19 small fixes: session server default port; syncer --resume only with a checkpoint."""

import subprocess
from types import SimpleNamespace

from yeto import launcher as L
from yeto.rl import ssh_harness as sh


def test_session_server_port_defaults_outside_dynamic_range():
    args = SimpleNamespace(use_session_server=True, session_server_port=None)
    L._default_session_server_port(args)
    assert args.session_server_port == [L.DEFAULT_SESSION_SERVER_PORT]
    assert 30000 <= L.DEFAULT_SESSION_SERVER_PORT and L.DEFAULT_SESSION_SERVER_PORT + 256 < 32768


def test_session_server_port_explicit_or_disabled_is_untouched():
    args = SimpleNamespace(use_session_server=True, session_server_port=[25000])
    L._default_session_server_port(args)
    assert args.session_server_port == [25000]
    args = SimpleNamespace(use_session_server=False, session_server_port=None)
    L._default_session_server_port(args)
    assert args.session_server_port is None


def _plan():
    return {"syncer_port": 29400, "learner": {"global_rounds": 3}, "learner_count": 1}


def _argvs(monkeypatch):
    monkeypatch.setattr(sh, "_learner_count", lambda plan: 1)
    return sh._syncer_argv(_plan()), sh._syncer_argv(_plan(), resume=False)


def test_syncer_argv_resume_flag_is_optional(monkeypatch):
    resume, fresh = _argvs(monkeypatch)
    assert "--resume" in resume and "--resume" not in fresh
    assert [v for v in resume if v != "--resume"] == fresh


def _match(monkeypatch, actual, start=0):
    """Run the shell identity helpers against one actual argv."""
    monkeypatch.setattr(sh, "_learner_count", lambda plan: 1)
    script = "\n".join([
        'RUN=/r', 'WRAPPER="$RUN/state/run-syncer"', 'EXIT_FILE="$RUN/state/syncer.exit"',
        sh._syncer_expected_argv_lines(_plan()),
        sh._syncer_runtime_identity_functions(),
        "ACTUAL=(" + " ".join(f"'{a}'" for a in actual) + ")",
        f'syncer_argv_matches ACTUAL {start} EXPECTED_SYNCER_ARGV '
        f'|| syncer_argv_matches ACTUAL {start} EXPECTED_SYNCER_ARGV_FRESH',
    ])
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True).returncode == 0


def _expand(argv):
    return [a.replace("$RUN", "/r") for a in argv]


def test_identity_accepts_both_variants_and_rejects_others(monkeypatch):
    resume, fresh = _argvs(monkeypatch)
    assert _match(monkeypatch, _expand(resume))
    assert _match(monkeypatch, _expand(fresh))
    # unit form: wrapper + exit file prefix, matched from an offset
    assert _match(monkeypatch, ["/bin/bash"] + _expand(fresh)[0:], start=1)
    other = _expand(fresh)
    other[other.index("--port") + 1] = "1"
    assert not _match(monkeypatch, other)
    assert not _match(monkeypatch, _expand(fresh) + ["--extra"])
    assert not _match(monkeypatch, _expand(fresh)[:-1])


def test_all_five_identity_scripts_define_both_variants(monkeypatch):
    monkeypatch.setattr(sh, "_learner_count", lambda plan: 1)
    plan = {**_plan(), "run_id": "r", "syncer_address": "127.0.0.1:29400"}
    monkeypatch.setattr(sh, "_remote_vars", lambda plan: "RUN=/r")
    scripts = [sh._syncer_start_script(plan), sh._syncer_status_script(plan),
               sh._syncer_stop_script(plan)]
    calls = []
    monkeypatch.setattr(sh, "_ssh", lambda plan, host, script: calls.append(script))
    monkeypatch.setattr(sh, "_syncer_host", lambda plan: "h")
    monkeypatch.setattr(sh, "load_plan", lambda path: (None, plan))
    sh.kill_syncer("x")
    monkeypatch.setattr(sh, "_all_hosts", lambda plan: [])
    sh._wait_for_syncer(plan, timeout_s=1)
    scripts += calls
    assert len(scripts) == 5
    for script in scripts:
        assert "EXPECTED_SYNCER_ARGV_FRESH=(" in script
        assert "EXPECTED_UNIT_ARGV_FRESH=(" in script
        syntax = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
        assert syntax.returncode == 0, syntax.stderr
    start = scripts[0]
    assert 'if [ -e "$CHECKPOINT_PATH" ]; then' in start
    assert '"$WRAPPER" "$EXIT_FILE" "${LAUNCH_SYNCER_ARGV[@]}"' in start


def test_start_script_launches_fresh_without_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(sh, "_learner_count", lambda plan: 1)
    script = "\n".join([
        f'RUN={tmp_path}', 'WRAPPER="$RUN/state/run-syncer"', 'EXIT_FILE="$RUN/state/syncer.exit"',
        'CHECKPOINT_PATH="$RUN/state/state.ckpt"',
        sh._syncer_expected_argv_lines(_plan()),
        'if [ -e "$CHECKPOINT_PATH" ]; then LAUNCH=("${EXPECTED_SYNCER_ARGV[@]}"); '
        'else LAUNCH=("${EXPECTED_SYNCER_ARGV_FRESH[@]}"); fi',
        'printf "%s\\n" "${LAUNCH[@]}"',
    ])
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout
    assert "--resume" not in out.split()
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "state.ckpt").write_bytes(b"x")
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout
    assert "--resume" in out.split()

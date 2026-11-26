"""The Miles trainer island's tape upload is observation only: an uploader whose stop()
raises after the training ended (a bad YETO_RL_TAPE_FINAL_FLUSH_S, say) leaves the
island's exit code to the training."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

from yeto.rl.adapters.miles import island_entry
from yeto.rl.engine import tape_upload

# The island process: the real TapeUploader over a directory store, the training a
# callable that ends with ``code`` (0: returns, else SystemExit(code)).
ISLAND = textwrap.dedent("""
    import sys
    from types import SimpleNamespace

    from yeto.rl.adapters.miles import island_entry
    from yeto.rl.engine import tape_upload
    from yeto.rl.engine.object_store import DirectoryObjectStore

    store, tape, code = sys.argv[1], sys.argv[2], int(sys.argv[3])

    def uploader(event_tape, *, learner_id=0, **_):
        return tape_upload.TapeUploader(DirectoryObjectStore(store), event_tape, island="trainer",
                                        incarnation=f"l{learner_id}", interval_s=999).start()

    def train():
        print("[rl] learner 0 finalized", flush=True)
        if code:
            raise SystemExit(code)

    tape_upload.trainer_uploader = uploader
    island_entry._run_with_trainer_tape(SimpleNamespace(event_tape=tape, learner_id=0), train)
""")


def _island(tmp_path, code: int, flush_s: str | None) -> subprocess.CompletedProcess:
    tape = tmp_path / "tape.jsonl"
    tape.write_text('{"event": "rl_learner_finalized"}\n')
    env = dict(os.environ)
    env.pop(tape_upload.FINAL_FLUSH_ENV, None)
    if flush_s is not None:
        env[tape_upload.FINAL_FLUSH_ENV] = flush_s
    return subprocess.run([sys.executable, "-c", ISLAND, str(tmp_path / "store"), str(tape), str(code)],
                          capture_output=True, text=True, env=env, timeout=120)


@pytest.mark.parametrize("code", [0, 11])
def test_an_uploader_that_fails_to_stop_leaves_the_trainings_exit_code(tmp_path, code):
    proc = _island(tmp_path, code, flush_s="soon")
    assert proc.returncode == code, proc.stderr
    assert ("[yeto-island] tape upload stop failed: ValueError: could not convert string to float: 'soon'"
            in proc.stdout)
    assert "Traceback" not in proc.stderr


@pytest.mark.parametrize("code", [0, 11])
def test_an_uploader_that_stops_uploads_the_tape_and_leaves_the_exit_code(tmp_path, code):
    proc = _island(tmp_path, code, flush_s=None)
    assert proc.returncode == code, proc.stderr
    assert "tape upload stop failed" not in proc.stdout
    stored = sorted((tmp_path / "store").rglob("*.jsonl"))
    assert b"".join(p.read_bytes() for p in stored) == (tmp_path / "tape.jsonl").read_bytes()


def test_stop_raising_in_process_keeps_the_trainings_exception(monkeypatch, capsys):
    """The training's own error (here its SystemExit) is what leaves, not the stop() error."""
    class Broken:
        def stop(self):
            raise ValueError("could not convert string to float: 'soon'")

    monkeypatch.setattr(tape_upload, "trainer_uploader", lambda *a, **k: Broken())

    def train():
        raise SystemExit(11)

    with pytest.raises(SystemExit) as raised:
        island_entry._run_with_trainer_tape(island_entry.argparse.Namespace(event_tape="t", learner_id=0), train)
    assert raised.value.code == 11
    assert "[yeto-island] tape upload stop failed: ValueError" in capsys.readouterr().out


def test_stop_does_not_swallow_keyboard_interrupt(monkeypatch):
    class Interrupted:
        def stop(self):
            raise KeyboardInterrupt

    monkeypatch.setattr(tape_upload, "trainer_uploader", lambda *a, **k: Interrupted())
    with pytest.raises(KeyboardInterrupt):
        island_entry._run_with_trainer_tape(island_entry.argparse.Namespace(event_tape="t", learner_id=0),
                                            lambda: None)

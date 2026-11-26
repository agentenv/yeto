"""A verl trainer island uploads its event tape to the run store while it trains, as a Miles
island does: a fleet's dashboard reads the trainer of a Modal or sky island from the store.
The upload is observation only: an uploader that cannot start leaves the training as it is."""

from __future__ import annotations

import subprocess
import sys
import types

import pytest

from yeto.rl.adapters.verl import island_entry
from yeto.rl.engine import tape_upload


class FakeUploader:
    def __init__(self, calls):
        self.calls = calls

    def stop(self):
        self.calls.append("stop")
        return True


@pytest.fixture
def trainer(monkeypatch):
    """``run_trainer`` with the store uploader and the verl process faked: the calls in order."""
    calls = []

    def uploader(event_tape, *, learner_id=0, **_kw):
        calls.append(("start", event_tape, learner_id))
        return FakeUploader(calls)

    monkeypatch.setattr(tape_upload, "trainer_uploader", uploader)
    monkeypatch.setattr(island_entry, "verl_root", lambda: ".")

    def popen(cmd, **_kw):
        if cmd == ["missing"]:
            raise FileNotFoundError(cmd[0])
        calls.append("train")
        return types.SimpleNamespace(stdout=iter(["step 1\n"]), wait=lambda: calls.append("exit") or 3)

    monkeypatch.setattr(island_entry, "subprocess",  # this module's Popen only, not the process-wide one
                        types.SimpleNamespace(Popen=popen, PIPE=subprocess.PIPE, STDOUT=subprocess.STDOUT))
    return calls


def test_the_tape_is_uploaded_while_the_trainer_runs_and_flushed_at_its_end(trainer, tmp_path, capsys):
    code = island_entry.run_trainer(["verl"], {}, tmp_path / "train.log", event_tape="~/tape.jsonl", learner_id=2)
    assert code == 3
    assert trainer == [("start", "~/tape.jsonl", 2), "train", "exit", "stop"]
    assert (tmp_path / "train.log").read_text() == "step 1\n" and "step 1" in capsys.readouterr().out


def test_the_uploader_stops_when_the_trainer_cannot_start(trainer, tmp_path):
    with pytest.raises(FileNotFoundError):
        island_entry.run_trainer(["missing"], {}, tmp_path / "train.log",
                                 event_tape="t.jsonl", learner_id=0)
    assert trainer[-1] == "stop"


def test_an_uploader_that_cannot_start_leaves_the_training_alone(monkeypatch, tmp_path, capsys):
    def broken(*_a, **_kw):
        raise OSError("no store credentials")

    monkeypatch.setattr(tape_upload, "trainer_uploader", broken)
    monkeypatch.setattr(island_entry, "verl_root", lambda: str(tmp_path))
    code = island_entry.run_trainer([sys.executable, "-c", "print('ok')"], {}, tmp_path / "train.log",
                                    event_tape="t.jsonl", learner_id=0)
    assert code == 0
    assert "[yeto-verl] tape upload off: OSError: no store credentials" in capsys.readouterr().out


def test_without_a_store_no_uploader_runs(monkeypatch, tmp_path):
    monkeypatch.delenv("YETO_RL_STORE_S3_URI", raising=False)
    monkeypatch.setattr(island_entry, "verl_root", lambda: str(tmp_path))
    assert tape_upload.trainer_uploader("t.jsonl", learner_id=0) is None
    assert island_entry.run_trainer([sys.executable, "-c", "pass"], {}, tmp_path / "train.log",
                                    event_tape="t.jsonl", learner_id=0) == 0


def test_an_uploader_that_fails_to_stop_leaves_the_trainers_exit_code(trainer, monkeypatch, tmp_path, capsys):
    """stop() raising (a bad YETO_RL_TAPE_FINAL_FLUSH_S, say) after the trainer finished: the
    island still exits with the trainer's code, and says the flush failed."""
    def broken_stop(self):
        self.calls.append("stop")
        raise ValueError("could not convert string to float: 'soon'")

    monkeypatch.setattr(FakeUploader, "stop", broken_stop)
    code = island_entry.run_trainer(["verl"], {}, tmp_path / "train.log", event_tape="t.jsonl", learner_id=0)
    assert code == 3 and trainer[-1] == "stop"
    assert ("[yeto-verl] tape upload stop failed: ValueError: could not convert string to float: 'soon'"
            in capsys.readouterr().out)

"""HangStackDumper: py-spy every Python process once a NCCL flight-recorder file appears."""
import os
import stat

from yeto.modal_runner import HangStackDumper


def _fake_proc(root, pids):
    for pid, cmd in pids.items():
        d = root / str(pid)
        d.mkdir(parents=True)
        (d / "cmdline").write_bytes(cmd.replace(" ", "\0").encode())
    (root / "self").mkdir()


def _fake_py_spy(tmp_path):
    exe = tmp_path / "py-spy"
    exe.write_text("#!/bin/sh\necho \"Thread main (stack of pid $3)\"\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return str(exe)


def test_from_env_is_opt_in(tmp_path):
    assert HangStackDumper.from_env({}, str(tmp_path), 0) is None
    d = HangStackDumper.from_env({"YETO_HANG_STACK_DUMP": "1",
                                  "TORCH_NCCL_DEBUG_INFO_TEMP_FILE": "/x/trace_"}, str(tmp_path), 0)
    assert d is not None and d.trace_prefix == "/x/trace_"


def test_triggered_matches_prefix_only(tmp_path):
    d = HangStackDumper(str(tmp_path / "out"), 0, trace_prefix=str(tmp_path / "nccl_trace_rank_"))
    assert d.triggered() == []
    (tmp_path / "other").write_text("")
    (tmp_path / "nccl_trace_rank_1").write_text("")
    assert d.triggered() == [str(tmp_path / "nccl_trace_rank_1")]


def test_dump_covers_ray_and_python_processes(tmp_path):
    proc = tmp_path / "proc"
    _fake_proc(proc, {11: "ray::MegatronTrainRayActor.train", 12: "/usr/bin/python3 -m x",
                      13: "/bin/bash -c sleep"})
    d = HangStackDumper(str(tmp_path / "out"), 1, py_spy=_fake_py_spy(tmp_path), proc=str(proc))
    assert [p for p, _ in d.python_pids()] == [11, 12]
    path = d.dump("test")
    text = open(path).read()
    assert os.path.basename(path) == "hang-stacks-rank1-1.txt"
    assert "pid 11: ray::MegatronTrainRayActor.train" in text and "stack of pid 11" in text
    assert "stack of pid 12" in text and "pid 13" not in text


def test_loop_dumps_twice_after_trigger(tmp_path):
    proc = tmp_path / "proc"
    _fake_proc(proc, {11: "ray::A"})
    prefix = tmp_path / "nccl_trace_rank_"
    d = HangStackDumper(str(tmp_path / "out"), 0, trace_prefix=str(prefix), poll_s=0.01,
                        repeat_s=0.01, py_spy=_fake_py_spy(tmp_path), proc=str(proc))
    (tmp_path / "nccl_trace_rank_0").write_text("")
    d._loop()
    assert d.dumps == 2
    assert sorted(os.listdir(tmp_path / "out")) == ["hang-stacks-rank0-1.txt", "hang-stacks-rank0-2.txt"]

"""tools/eval/health.py (S27 G-EU, VERL-PARITY #36): one health entry for both
backends; it picks the reader from the run directory and writes one summary."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tools.eval import health


def _run(tmp_path, *, args: str | None = None, verl_files: bool = False, islands: int = 2):
    run = tmp_path / "run"
    for k in range(islands):
        mirror = run / "tape-direct" / "yeto-x" / f"l{k}" / "rank0"
        mirror.mkdir(parents=True)
        if verl_files:
            (mirror / f"verl-plan-{k}.json").write_text("{}")
    if args is not None:
        (run / "args.txt").write_text(args)
    return run


def test_the_backend_comes_from_the_args_then_from_the_island_files(tmp_path):
    assert health.detect_backend(_run(tmp_path / "a", args="launch --rl-backend verl --seed 1"))[0] == "verl"
    assert health.detect_backend(_run(tmp_path / "b", args="launch --seed 1", verl_files=True))[0] == "verl"
    backend, why = health.detect_backend(_run(tmp_path / "c", args="launch --seed 1"))
    assert backend == "miles" and "no verl" in why
    assert health.detect_backend(_run(tmp_path / "d", args="launch --rl-backend miles", verl_files=True))[0] == "miles"


def test_the_verl_reader_defaults_come_from_the_run(tmp_path):
    run = _run(tmp_path, args="launch --total-steps 10 --fragments 4", islands=3)
    assert health.verl_defaults(run) == (3, 40)
    assert health.verl_defaults(_run(tmp_path / "e", islands=0)) == (2, None)


@pytest.mark.parametrize("backend,rows,overall,code", [
    ("verl", [{"check": "exit", "status": "PASS", "detail": "ok"}], "PASS", 0),
    ("verl", [{"check": "exit", "status": "FAIL", "detail": "rc=1"}], "FAIL", 1),
    ("miles", [{"name": "loss", "status": "WARN", "value": 1, "threshold": 2}], "WARN", 0),
])
def test_one_summary_whatever_the_reader(tmp_path, monkeypatch, capsys, backend, rows, overall, code):
    run = _run(tmp_path, args=f"launch --rl-backend {backend} --total-steps 2 --fragments 2")
    seen = {}

    def verl_check(path, learners, total):
        seen.update(path=path, learners=learners, total=total)
        counts = {s: sum(r["status"] == s for r in rows) for s in ("PASS", "WARN", "FAIL", "N/A")}
        return {"rows": rows, "counts": counts}

    def miles_main(argv):
        seen.update(argv=argv)
        out = argv[argv.index("--json") + 1]
        with open(out, "w") as handle:
            json.dump({"checks": rows, "overall": overall}, handle)

    readers = {"verl": SimpleNamespace(check=verl_check), "miles": SimpleNamespace(main=miles_main)}
    monkeypatch.setattr(health, "_load", lambda b: readers[b])
    out = tmp_path / "summary.json"
    assert health.main([str(run), "--json", str(out), "--no-table"]) == code
    summary = json.loads(out.read_text())
    assert summary["backend"] == backend and summary["overall"] == overall
    assert [c["status"] for c in summary["checks"]] == [r["status"] for r in rows]
    assert f"backend={backend} overall={overall}" in capsys.readouterr().out
    if backend == "verl":
        assert (seen["learners"], seen["total"]) == (2, 4)
    else:
        assert seen["argv"][0] == str(run) and "--no-table" in seen["argv"]


def test_the_summary_is_never_written_into_the_run_dir(tmp_path):
    run = _run(tmp_path, args="launch --rl-backend verl")
    with pytest.raises(SystemExit, match="inside the run dir"):
        health.main([str(run), "--json", str(run / "x.json")])

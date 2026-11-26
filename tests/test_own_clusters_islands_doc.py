"""docs/OWN_CLUSTERS.md, "Islands that lag, join or leave": every command of the section
parses, the ``yeto fleet island`` ones run in the doc's order on a seeded per-island fleet
(local store; the join checks' node health, placement and leases stand-ins), each dry run
writes nothing, and the flags, events and refusal the section shows are the code's."""

from __future__ import annotations

import re
import shlex
from pathlib import Path

from test_fleet_island_changes import fleet_store  # noqa: F401 - fixture
from test_island_join import Leases, Ssh, _inventory, stand_ins  # noqa: F401 - fixture
from yeto.launch import island_join as ij
from yeto.launch import island_straggler as st
from yeto.rl import fleet_islands

DOC = Path(__file__).resolve().parents[1] / "docs" / "OWN_CLUSTERS.md"
SECTION = "## Islands that lag, join or leave"


def section() -> str:
    text = DOC.read_text()
    start = text.index(SECTION)
    end = text.find("\n## ", start + len(SECTION))
    return text[start:] if end < 0 else text[start:end]


def commands() -> list[list[str]]:
    out = []
    for block in re.findall(r"```bash\n(.*?)```", section(), flags=re.DOTALL):
        out += [shlex.split(line) for line in block.replace("\\\n", " ").splitlines() if line.strip()]
    inline = re.findall(r"`(yeto island (?:stop|start) [^`]+)`", section())
    return out + [shlex.split(c) for c in inline]


def test_the_section_names_the_code_s_flags_and_events():
    text = section()
    assert "--rl-island-straggler-cycles" in text and f"`{st.DEFAULT_CYCLES}` is the default" in text
    for name in (st.EVENT, st.CLEARED_EVENT, st.UNAVAILABLE_EVENT, st.AVAILABLE_EVENT, st.WAIT,
                 st.SYNCER_TAPES.rstrip("/"), "controller restarted"):
        assert name in text
    assert f"{st.QUIET_POLLS} reads in a row" in text and f"every {st.STORE_POLL_S:g} s" in text
    assert f"at most {st.STORE_GETS_PER_POLL} chunks" in text and f"every {st.STORE_DISCOVER_POLLS} reads" in text
    assert f"up to {st.CLOCK_SKEW_S / 60:g} minutes before" in text
    for field in st.Straggler.__dataclass_fields__:
        if field not in ("island", "cleared", "cycles", "reason"):
            assert f"`{field}`" in text, field


def test_the_section_commands_run_in_order(fleet_store, stand_ins, tmp_path, monkeypatch, capsys):  # noqa: F811
    from yeto.cli import build_parser

    _cfg, _objects, store, run_cli, samplers = fleet_store
    argvs = commands()
    assert {a[1] for a in argvs} == {"fleet", "island"}
    for argv in argvs:
        build_parser().parse_args(argv[1:])
    monkeypatch.setattr("yeto.launch.ssh_trainer_island.SshOps", Ssh)
    monkeypatch.setattr(fleet_islands, "launch_dry_run", lambda member, **kw: None)
    monkeypatch.setattr(fleet_islands, "session_check", lambda *a, **kw: None)
    inventory = _inventory(tmp_path / "north.inventory.yaml", ["b300-4", "b300-5"])
    for argv in (a for a in argvs if a[:3] == ["yeto", "fleet", "island"]):
        rest = [str(samplers) if a == "north.samplers.json" else str(inventory) if a == "north.inventory.yaml"
                else a for a in argv[3:]]
        config = rest.index("--config")
        del rest[config:config + 2]
        before = store.read()["generation"]
        assert run_cli(*rest) == 0, (argv, capsys.readouterr().err)
        out = capsys.readouterr().out
        assert store.read()["generation"] == before + (0 if "--dry-run" in argv else 1)
        if argv[3] == "add":
            assert "nodes pass the join checks" in out


def test_the_refusal_the_section_shows_is_the_command_s(tmp_path, stand_ins, monkeypatch):  # noqa: F811
    from test_island_join import Checker

    stand_ins["checker"] = Checker(bad={"b300-5": "Xid 79 since the last check"})
    stand_ins["holders"] = {"b300-4": "other-run"}
    from yeto.launch.node_inventory import load_inventory

    nodes = load_inventory(_inventory(tmp_path / "inv.yaml", ["b300-4", "b300-5"]))
    out = ij.check(nodes, need=2, run_id="r", lease_store="s3://b/leases", ssh=Ssh())
    line = "fleet island: refused: island north's nodes cannot join: " + "; ".join(out.refusals)
    assert line in section()
    assert Leases.made

"""docs/OWN_CLUSTERS.md, "Islands that lag, join or leave": every command of the section
parses, the ``yeto fleet island`` ones run in the doc's order on a seeded per-island fleet
(local store), each dry run writes nothing, and the flags and events the section names are
the code's."""

from __future__ import annotations

import re
import shlex
from pathlib import Path

from test_fleet_island_changes import fleet_store  # noqa: F401 - fixture

from yeto.launch import island_straggler as st
from yeto.rl import fleet, fleet_islands

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
    for name in (st.EVENT, st.CLEARED_EVENT, st.UNAVAILABLE_EVENT, st.WAIT, st.SYNCER_TAPES.rstrip("/")):
        assert name in text
    assert f"{st.QUIET_POLLS} reads in a row" in text and f"every {st.STORE_POLL_S:g} s" in text
    for field in st.Straggler.__dataclass_fields__:
        if field not in ("island", "cleared", "cycles", "reason"):
            assert f"`{field}`" in text, field


def test_the_section_commands_run_in_order(fleet_store, capsys):  # noqa: F811
    from yeto.cli import build_parser

    cfg, _objects, store, run_cli, _samplers = fleet_store
    argvs = commands()
    assert {a[1] for a in argvs} == {"fleet", "island"}
    for argv in argvs:
        build_parser().parse_args(argv[1:])
    plan = fleet.plan_of(cfg)  # the island the section removes
    entry = dict(cfg["islands"][1]["samplers"][0], name="sglang-north")
    member, pools = fleet_islands.compile_island(cfg, plan, k=2, name="north", gpu="modal:1x2xh100", usd_per_h=8.0,
                                                 samplers=[entry])
    store.update(lambda d: fleet_islands.add(d, member, pools), note="add north")
    for argv in (a for a in argvs if a[:3] == ["yeto", "fleet", "island"]):
        rest = argv[3:]
        config = rest.index("--config")
        del rest[config:config + 2]
        before = store.read()["generation"]
        assert run_cli(*rest) == 0, (argv, capsys.readouterr().err)
        assert store.read()["generation"] == before + (0 if "--dry-run" in argv else 1)

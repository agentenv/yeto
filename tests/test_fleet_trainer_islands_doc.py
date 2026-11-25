"""docs/FLEET_TRAINER_ISLANDS.md: its commands run, in the doc's order, on a seeded
per-island fleet (local store, no cloud). Every ``--dry-run`` writes nothing, every
other command writes one generation, and the config keys it names are the compile's."""

from __future__ import annotations

import re
import shlex
from pathlib import Path

from test_fleet_island_changes import fleet_store  # noqa: F401 - the fixture
from yeto.launch.learner_ids import claim_key
from yeto.launch.pool import desired as dz
from yeto.rl import fleet

DOC = Path(__file__).resolve().parents[1] / "docs" / "FLEET_TRAINER_ISLANDS.md"


def doc_commands() -> list[list[str]]:
    blocks = re.findall(r"```bash\n(.*?)```", DOC.read_text(), flags=re.S)
    lines = [line.split("#", 1)[0].strip() for block in blocks for line in block.splitlines()]
    return [shlex.split(line) for line in lines if line]


def test_the_doc_names_the_compile_keys():
    text = DOC.read_text()
    for key in ("launch", "max_islands"):
        assert key in fleet.FLEET_KEYS and f"fleet.{key}" in text
    assert fleet.LAUNCH_PER_ISLAND in text and fleet.LAUNCH_SYNCED_GROUP in text


def test_the_doc_commands_run_in_order(fleet_store, capsys):  # noqa: F811
    cfg, objects, store, run_cli, samplers = fleet_store
    commands = doc_commands()
    islands = [c for c in commands if c[:3] == ["yeto", "fleet", "island"]]
    assert {c[3] for c in islands} == {"add", "move", "remove"}
    assert all("--dry-run" in c for c in islands[0::2]) and not any("--dry-run" in c for c in islands[1::2])
    from yeto.cli import build_parser

    for argv in commands:  # every command of the doc parses
        build_parser().parse_args(argv[1:])
    for argv in islands:
        rest = argv[3:]
        rest = [str(samplers) if a == "north.samplers.json" else a for a in rest]
        config = rest.index("--config")
        del rest[config:config + 2]  # the fixture's run config
        before = store.read()["generation"]
        assert run_cli(*rest) == 0, (argv, capsys.readouterr().err)
        out = capsys.readouterr().out
        if "--dry-run" in argv:
            assert store.read()["generation"] == before and "nothing changed" in out
        else:
            assert store.read()["generation"] == before + 1 and "written" in out
        if argv[3] in ("add", "move"):
            assert "passes its dry run" in out
    doc = store.read()
    trainers = dz.trainers(doc)
    assert "north" not in trainers and trainers["sglang"]["gpu"] == "aws:2xh100@us-east-2"
    assert trainers["sglang"]["learner_id"] == 1 and objects.read(claim_key(2)) is not None
    assert not [name for name in dz.sampler_pools(doc) if name.startswith("i2-")]

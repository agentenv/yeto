"""The configs shipped under examples/ pass the dependency rules (yeto.breakers.rules) without an
error: every run.*.json kit through the run up path (its pool and sampler spec followed), and every
sampler spec alone through the pool dry-run path. A new error here is either a bug in the example or a
rule that is too strict."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from yeto import breakers
from yeto.breakers import rules
from yeto.breakers.rules_facts import pool_sources
from yeto.launch.nc import cli as nc_cli

from test_k31_agentic_262k_kit import KITS, Kit
from test_k31_guide_examples_run import _filled

ROOT = Path(__file__).resolve().parents[1]
SPECS = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "examples").rglob("sampler-spec*.json"))


@pytest.mark.parametrize("names", KITS, ids=lambda k: k[0])
def test_every_run_kit_passes_the_rules(names, tmp_path, monkeypatch):
    monkeypatch.setattr(nc_cli.tunnel, "state_dir", lambda: tmp_path / "nc-state", raising=False)
    kit = Kit(names, tmp_path)
    run = {**kit.run, "trainer": {**kit.run["trainer"], "argv": kit.argv}}
    found = rules.check(breakers.resolved(run, strict=False), run_config=run)
    assert rules.errors(found) == [], [rules.format_finding(f) for f in found]


@pytest.mark.parametrize("path", SPECS)
def test_every_sampler_spec_passes_the_pool_rules(path, tmp_path):
    spec = _filled(json.loads((ROOT / path).read_text()), tmp_path)
    spec_file = tmp_path / "spec.json"
    spec_file.write_text(json.dumps(spec))
    sources = pool_sources({"spec": str(spec_file), "target": 1})
    found = rules.check(breakers.resolve(sources, strict=False), sources=sources)
    assert rules.errors(found) == [], [rules.format_finding(f) for f in found]


def test_the_kits_are_checked():
    assert KITS and SPECS


@pytest.mark.parametrize("names", KITS, ids=lambda k: k[0])
def test_the_kits_wait_out_one_agentic_round(names, tmp_path, monkeypatch):
    """The trainer's sample wait covers one agentic round plus its drain (no D4), and the stall
    clock, fed by the waiting trainer's heartbeat, does not cut that wait short (no D5)."""
    monkeypatch.setattr(nc_cli.tunnel, "state_dir", lambda: tmp_path / "nc-state", raising=False)
    kit = Kit(names, tmp_path)
    run = {**kit.run, "trainer": {**kit.run["trainer"], "argv": kit.argv}}
    found = rules.check(breakers.resolved(run, strict=False), run_config=run)
    assert not [f for f in found if f.rule in ("D4", "D5")], [rules.format_finding(f) for f in found]

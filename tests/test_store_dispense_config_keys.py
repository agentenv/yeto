"""A misspelled key of a store dispenser config (``prompts.dispense`` with
``backend: store`` in the sampler spec, ``sampler.dispense`` of a pool config) is
refused instead of silently dropped: a ``holdout_id`` typo would train on the
hold-out tasks, a ``namespce`` typo would share the default ledger with another run."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tests.test_pool_slot_name import BASE
from tests.test_store_dispense import FakeObjects, _open, _rows, _write
from yeto.launch.nc import cli, tasks
from yeto.rl.engine import store_dispense as sd

DOCS = Path(__file__).resolve().parents[1] / "docs"


@pytest.mark.parametrize("typo", ["holdout_id", "hold_out", "namespce", "lease", "max_redos", "label"])
def test_sampler_refuses_a_misspelled_dispense_key(tmp_path, typo):
    path = _write(tmp_path, _rows(3))
    with pytest.raises(sd.DispenseError, match=typo):
        _open(path, FakeObjects(), config={typo: ["t2"]})


def test_holdout_typo_does_not_train_on_the_holdout(tmp_path):
    path = _write(tmp_path, _rows(3))
    with pytest.raises(sd.DispenseError, match="unknown keys"):
        _open(path, FakeObjects(), config={"holdout_id": ["t2"]})


def test_every_documented_and_generated_key_is_accepted(tmp_path):
    hold = tmp_path / "hold.txt"
    hold.write_text("zz\n")
    generated = tasks.dispense_config("run/d", lease_s=60, max_redo=1, passes=1, seed=3,
                                      holdout_file=str(hold), labels="a", by_label=True)
    sd.check_store_config(generated)
    doc = (DOCS / "DISPENSER.md").read_text()
    start = doc.index("| Key |", doc.index('"backend": "store"'))
    table = doc[start:doc.index("\n\n", start)]
    documented = set(re.findall(r"^\| `(\w+)`(?: / `(\w+)`|, `(\w+)`)?", table, flags=re.M))
    keys = {k for row in documented for k in row if k}
    assert {"store", "holdout", "full_scan", "max_scan_s"} <= keys
    sd.check_store_config({k: None for k in keys})
    block = doc[doc.index('"prompts": {'):doc.index("```", doc.index('"prompts": {'))]
    sd.check_store_config(json.loads("{" + block + "}")["prompts"]["dispense"])


def test_pool_config_refuses_a_misspelled_sampler_dispense_key():
    cfg = dict(BASE, accounts=[7], sampler={"dispense": {"backend": "store", "namespace": "r/d"}})
    cli.pool_config(cfg, allowed={7})
    bad = dict(cfg, sampler={"dispense": {"backend": "store", "namespace": "r/d", "holdout_id": ["t"]}})
    with pytest.raises(ValueError, match="holdout_id"):
        cli.pool_config(bad, allowed={7})
    with pytest.raises((TypeError, ValueError), match="dispense"):
        cli.pool_config(dict(cfg, sampler={"dispense": "r/d"}), allowed={7})

"""A fleet of verl trainer islands on the dashboard (fleet-verl-trainer-islands 6.1): each island's
card names its backend, and a round metric its backend does not report (its identity's
``REPORTED_METRICS``: train-rollout KL, PPO KL for verl) reads "not reported by verl", never a
bare "n/a". A Miles fleet and a single-trainer run render as before."""

from __future__ import annotations

import re

import pytest
import test_dashboard_fleet_na as fleet_na
from test_dashboard_diloco import _smoke
from test_dashboard_fleet_islands import fleet_records, plan_doc

from yeto.dashboard.reducer import Reducer, unreported_metrics
from yeto.rl.engine import backend_identity
from yeto.rl.engine.backend_identity import METRIC_PPO_KL, METRIC_TRAIN_ROLLOUT_KL, ROUND_METRICS, reported_metrics


def _fleet(backend: str | None) -> Reducer:
    """The two-island fleet fixture, its trainers on ``backend`` (as their rl_engine_selected
    names it: verl writes rl_backend, Miles its miles_commit); None: no backend field (older tapes)."""
    r = Reducer(run="fleet-qa")
    for rec in fleet_records():
        if rec.get("event") == "rl_engine_selected" and backend == "verl":
            rec = dict(rec, rl_backend="verl", verl_commit="0123456789ab")
        elif rec.get("event") == "rl_engine_selected" and backend == "miles":
            rec = dict(rec, miles_commit="fedcba987654")
        r.feed(rec)
    assert r.load_fleet_plan(plan_doc(), "test")
    return r


def _text(html: str) -> str:
    return re.sub(r"<[^>]*>", " ", html)


def test_backends_declare_the_round_metrics_they_report():
    from yeto.rl.adapters.miles import identity as miles_identity
    from yeto.rl.adapters.verl import identity as verl_identity

    assert miles_identity.REPORTED_METRICS == ROUND_METRICS
    assert METRIC_TRAIN_ROLLOUT_KL not in verl_identity.REPORTED_METRICS
    assert METRIC_PPO_KL not in verl_identity.REPORTED_METRICS
    assert reported_metrics("verl") == verl_identity.REPORTED_METRICS
    assert reported_metrics("miles") == ROUND_METRICS
    # a backend not registered, or none known: every metric (a missing value is "not on the tape")
    assert reported_metrics("no-such-backend") == ROUND_METRICS == reported_metrics(None)


def test_the_card_reads_what_the_backend_declares(monkeypatch):
    assert unreported_metrics("verl") == {METRIC_PPO_KL: "not reported by verl",
                                          METRIC_TRAIN_ROLLOUT_KL: "not reported by verl"}
    assert unreported_metrics("miles") == {} == unreported_metrics(None)
    # the card asks the declaration, not the name: a backend that declares it reports PPO KL
    monkeypatch.setattr(backend_identity, "reported_metrics",
                        lambda name: frozenset({METRIC_PPO_KL}) if name == "verl" else ROUND_METRICS)
    assert unreported_metrics("verl") == {METRIC_TRAIN_ROLLOUT_KL: "not reported by verl"}


def test_verl_fleet_cards_name_their_backend_and_what_it_does_not_report():
    r = _fleet("verl")
    cards = {c["id"]: c for c in r.overview(now=r.max_ts + 30)["islands"]}
    for k in ("0", "1"):
        assert cards[k]["backend"] == "verl" and cards[k]["backend_label"] == "verl"
        assert set(cards[k]["unreported"]) == {METRIC_TRAIN_ROLLOUT_KL, METRIC_PPO_KL}
    samplers = [c for c in cards.values() if c["role"] == "sampler"]
    assert samplers and all(c["unreported"] == {} and c["backend_label"] is None for c in samplers)


@pytest.mark.parametrize("backend", ["miles", None])
def test_a_miles_fleet_card_is_unchanged(backend):
    r = _fleet(backend)
    cards = r.overview(now=r.max_ts + 30)["islands"]
    assert all(c["unreported"] == {} and c["backend_label"] is None for c in cards)
    assert {c["backend"] for c in cards if c["role"] == "trainer"} == {backend}


def test_verl_fleet_page_says_not_reported_by_verl_and_has_no_bare_na(tmp_path):
    out = fleet_na._page(tmp_path, _fleet("verl"))
    nodes = out["rNodes"]
    rows = [tr for tr in re.findall(r"<tr[^>]*>.*?</tr>", nodes) if re.search(r">trainer [01]<", tr)]
    assert len(rows) == 2
    for tr in rows:
        assert re.search(r"H100 × \d+ · verl", _text(tr)) and '—<div class="ns">not reported by verl</div>' in tr
    for key in ("rKpis", "rCostK", "rNodes", "aTrainer", "aSamplers", "rFleet"):
        assert not re.search(r"\bn/a\b", _text(out[key])), key
    for pick in ("i0", "i1"):
        (tmp_path / pick).mkdir()
        picked = fleet_na._page(tmp_path / pick, _fleet("verl"), pick=pick)
        for key in ("rKpis", "rNodes", "aTrainer", "aSamplers"):
            assert not re.search(r"\bn/a\b", _text(picked[key])), (pick, key)


def test_a_miles_fleet_page_names_no_backend(tmp_path):
    nodes = fleet_na._page(tmp_path, _fleet("miles"))["rNodes"]
    assert "verl" not in nodes and "not reported by" not in nodes


def test_verl_trainer_cards_in_the_drawing_name_verl(tmp_path):
    out = _smoke(tmp_path, _fleet("verl"))
    subs = [t for t in out["texts"] if t.endswith(" · verl")]
    assert len(subs) == 2 and all("H100" in t for t in subs)
    assert not any(re.search(r"\bn/a\b", t) for t in out["texts"])
    (tmp_path / "m").mkdir()
    miles = _smoke(tmp_path / "m", _fleet("miles"))
    assert not any("verl" in t for t in miles["texts"])

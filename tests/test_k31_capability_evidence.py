"""The customer K3.1 path's mechanisms on the k31 lineage (capability evidence).

``examples/k31/run.*.json`` pass ``--rl-allow-unverified-mechanism`` for four
mechanisms. The flags stay (the customer plan golden is byte for byte), and an
allowance for a mechanism the pin declares is a no-op: it widens nothing and marks no
artifact as unverified.

* ``losses:score_centering``: declared from FULL_D on (k31 GPU evidence);
* ``features:no_grpo_std_normalization``: declared on every pin (its evidence is not
  pin-bound);
* ``features:rollout_logprobs_as_old`` and ``loss_aggregations:token``: undeclared on
  k31 (K31_REVERIFY: no k31 GPU run shows the CP split of the rollout log-probs, nor a
  paired per-token grad_norm), so their allowance still applies.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from yeto.rl.adapters.miles import entry, pins

ROOT = Path(__file__).resolve().parents[1]
CUSTOMER = ("losses:score_centering", "features:rollout_logprobs_as_old",
            "features:no_grpo_std_normalization", "loss_aggregations:token")
STILL_UNVERIFIED = ("features:rollout_logprobs_as_old", "loss_aggregations:token")


def _allowances(path: Path) -> list[str]:
    argv = json.loads(path.read_text())["trainer"]["argv"]
    return [argv[i + 1] for i, token in enumerate(argv) if token == "--rl-allow-unverified-mechanism"]


def test_the_customer_run_files_keep_their_four_allowances():
    for name in ("run.agentic-262k.json", "run.baremetal-b300-262k.json"):
        assert sorted(_allowances(ROOT / "examples" / "k31" / name)) == sorted(CUSTOMER), name


def test_score_centering_is_declared_on_the_customer_pin_with_k31_evidence():
    head = pins.MILES_K31_NEXT9_COMMIT
    assert "score_centering" in entry.declared_by_dimension(head)["losses"]
    assert "score_centering" in entry.declared_by_dimension(pins.MILES_K31_FULL_D_COMMIT)["losses"]
    assert "score_centering" not in entry.declared_by_dimension(pins.MILES_K31_FULL_C_COMMIT).get("losses", set())
    for ref in entry.MILES_DECLARED_EVIDENCE["losses:score_centering"]:
        assert ref.lineage == "k31" and entry.evidence_covers(ref, head)
        assert "2026-11-" in ref.run_id and "outside the repo" in ref.judgment_path


def test_a_declared_mechanism_s_allowance_is_a_no_op_on_the_customer_pin():
    head = pins.MILES_K31_NEXT9_COMMIT
    assert entry.undeclared_allowance(CUSTOMER, head) == STILL_UNVERIFIED
    assert set(entry.needs_k31_reverification(head)) >= set(STILL_UNVERIFIED)
    # the run's selection record lists only the allowances still undeclared on its pin
    launch = SimpleNamespace(placement=SimpleNamespace(kind="colocated"), absorbed_flags=None)
    from yeto.rl.engine.algorithm import AlgorithmSpec

    event = entry.selection_event(launch=launch, algorithm=AlgorithmSpec(), miles_commit=head,
                                  unverified_mechanisms=CUSTOMER, outer_sync=False)
    assert tuple(event["rl/unverified_mechanisms"]) == STILL_UNVERIFIED


def test_the_capabilities_widen_only_by_the_undeclared_allowances(monkeypatch):
    monkeypatch.setattr(pins, "MILES_NEXT_COMMIT", pins.MILES_K31_NEXT9_COMMIT)
    caps = entry.miles_capabilities("sha256:" + "0" * 64, unverified_mechanisms=CUSTOMER)
    assert caps.unverified_mechanisms == frozenset(STILL_UNVERIFIED)
    assert "score_centering" in caps.losses

"""Regenerate tests/fixtures/shape_aws_runpod_snapshot.json.

The snapshot pins the exact `--clouds aws,runpod` plan the shaper produced
BEFORE the cloud-signal registry refactor, so the refactor can be checked
for behavior preservation field by field. Re-run this ONLY when a
deliberate planner behavior change is intended, and say so in the commit.

    python tests/fixtures/gen_shape_snapshot.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import test_shape_plan as t  # noqa: E402

from yeto.shape import plan as plan_mod  # noqa: E402


def _offerings(regions, gpus, cache, clouds=("aws",)):
    out = [o for o in t.OFFERINGS + t.RUNPOD_OFFERINGS if not gpus or o.gpu in gpus]
    out = [o for o in out if o.cloud in clouds]
    if regions is not None:
        out = [o for o in out if o.cloud != "aws" or o.region in regions]
    return out


def main() -> None:
    plan_mod.list_offerings = _offerings
    plan_mod.model_weights_gb = lambda model, override, cache: override or 66.0
    cases = {}
    for name, kw in {
        "high_stock_budget_500": dict(budget=500.0, stock={("H100", 8): 9, ("B200", 8): 9}),
        "medium_stock_min_score_5": dict(
            budget=5000.0, stock={("H100", 8): 6, ("B200", 8): 0}, min_score=5, gpus=["H100"]
        ),
        "unfetchable_stock": dict(budget=500.0, stock={}),
    }.items():
        stock = kw.pop("stock")
        fake = t.FakeAws(t.QUOTAS, t.SCORES, t.CODES)
        result = t._shape(
            fake, clouds=("aws", "runpod"), **_signals_kw(stock), **kw
        )
        d = plan_mod.to_json_dict(result, "gemma4", kw["budget"], "lora", "org/data")
        cases[name] = d
    out = ROOT / "tests" / "fixtures" / "shape_aws_runpod_snapshot.json"
    out.write_text(json.dumps(cases, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out}")


def _signals_kw(stock: dict) -> dict:
    """Pre-refactor: runpod_providers; post-refactor: signals={'runpod': ...}."""
    if "runpod_providers" in plan_mod.build_shape.__code__.co_varnames:
        return {"runpod_providers": t.FakeRunPod(stock)}
    return {"signals": {"runpod": t.FakeRunPod(stock)}}


if __name__ == "__main__":
    main()

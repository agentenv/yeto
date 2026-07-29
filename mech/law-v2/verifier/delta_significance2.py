#!/usr/bin/env python3
"""Jackknife SEs for the signed-alignment/hump models vs C1sat-H7kin/H7."""
from __future__ import annotations

import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "zoo"))
import fit_zoo as fz  # noqa: E402
import attack_experiments as ax  # noqa: E402
import signed_align_probe as sp  # noqa: E402


def main() -> None:
    df = fz.load()
    refs = ax.reference_models()
    signed = {m.name: m for m in sp.make_models()}
    models = [refs["C1sat-H7kin"], refs["H7-rho-floor-S"],
              signed["Vsigned-align-M"], signed["Vhump-phi"]]
    resids = {}
    for m in models:
        pooled, _, resid = fz.loco(m, df, fz.SEED)
        resids[m.name] = resid
        print(f"{m.name:16s} LOCO={pooled:.3f}")
    out = {}
    for a in ["Vsigned-align-M", "Vhump-phi"]:
        for b in ["C1sat-H7kin", "H7-rho-floor-S"]:
            d, se = fz.jackknife_delta(resids[a], resids[b], df)
            out[f"{a} - {b}"] = dict(delta=d, jk_se=se)
            print(f"{a} - {b}: delta={d:+.3f} bits, jk SE={se:.3f} "
                  f"({abs(d)/se:.1f}x)")
    with open(HERE / "delta_significance2.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()

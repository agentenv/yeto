#!/usr/bin/env python3
"""law-v2 VERIFIER: nested model-selection cross-validation.

The zoo's league picks its winner by pooled LOCO RMSE over the same 12
campaigns used to score it -- winner selection is therefore in-sample at the
campaign level (and the H4-H7/C1sat forms were themselves designed after
inspecting those LOCO residuals; that human-level leakage cannot be undone
here).  This script measures the *model-pick* component of the optimism:

  for each outer campaign fold:
     inner LOCO over the remaining 11 campaigns, all 21 zoo models
     select the model with the lowest inner pooled RMSE
     fit it on all 11 campaigns and predict the outer campaign

The pooled outer RMSE is a selection-honest counterpart of the league's
0.331; the gap is the selection optimism.

Run: .venv/bin/python mech/law-v2/verifier/nested_selection_cv.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "zoo"))
import fit_zoo as fz  # noqa: E402


def main() -> None:
    df = fz.load()
    zoo = fz.make_zoo()
    fz.add_own_designs(zoo)
    fz.add_theory_candidates(zoo)

    campaigns = sorted(df["campaign"].unique())
    outer_resid = np.full(len(df), np.nan)
    picks = {}
    t0 = time.time()
    for oc in campaigns:
        inner_df = df[df["campaign"] != oc]
        # inner LOCO league on 11 campaigns
        best_name, best_rmse = None, np.inf
        for m in zoo:
            pooled, _, _ = fz.loco(m, inner_df.reset_index(drop=True),
                                   fz.SEED)
            if pooled < best_rmse:
                best_rmse, best_name = pooled, m.name
        # fit the selected model on all 11 campaigns, predict outer
        m = next(x for x in zoo if x.name == best_name)
        trp = fz.pack(inner_df)
        tep = fz.pack(df[df["campaign"] == oc])
        rng = np.random.default_rng(fz.SEED)
        th = fz.fit(m, trp, rng)
        pred = fz.predict(m, th, trp, tep)
        r = df[df["campaign"] == oc]["y"].to_numpy() - pred
        outer_resid[df.index[df["campaign"] == oc].to_numpy()] = r
        picks[oc] = dict(selected=best_name, inner_rmse=float(best_rmse),
                         outer_rmse=float(np.sqrt(np.mean(r ** 2))))
        print(f"[{time.time()-t0:7.1f}s] outer={oc:9s} pick={best_name:20s} "
              f"inner={best_rmse:.3f} outer={picks[oc]['outer_rmse']:.3f}",
              flush=True)

    assert not np.isnan(outer_resid).any()
    pooled = float(np.sqrt(np.mean(outer_resid ** 2)))
    out = dict(pooled_outer_rmse=pooled, picks=picks)
    print(f"\nnested-selection pooled RMSE = {pooled:.3f} bits "
          f"(league winner in-sample selection: 0.331)")
    with open(HERE / "nested_selection_cv.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()

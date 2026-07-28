# V2 registration pack

This branch contains four prospective designs and their deterministic,
CPU-only gate-feasibility simulations. It also contains an evidence-package
manifest audited against the newer frozen paper text. The H200 nodes were
offline throughout this work.

## Registration ledger

| Program | Registration commit | Future scientific cells | Principal gate simulation |
|---|---|---:|---|
| v14 transfer matrix | `3b14e87fd5e26f2814c9ea0c8459690e4c5a02d9` | 160 | `P_eval=1.000`; weakest upward-pair power `0.86795`; asymmetry `0.91335` |
| v15 multi-seed scale panel | `500b7897aa0d34f4e6a5f0930702dff4c171240e` | 180 | 7B absolute `0.99810`; 7B relative `0.99900`; independent verdicts |
| v16 second-family redesign | `9bd1ccc43172e4b72c3b252fbde7526470e8a56f` | 612 | observed-family-noise `P_eval=0.82470` |
| v17 reproduce and overturn | `547c4b99fabe1162666722483e9a9269225b0ada` | adaptive + 35 fixed full-budget runs | survival-alternative power `0.82370`; tuned-null overturn probability `0.99990` |

The v17 count is adaptive because only preregistered successive-halving
survivors advance. Its fixed full-budget component is 14 Phase-A runs plus 21
Phase-B confirmation runs.

## Files

Each registration has a machine JSON contract, a readable Markdown contract,
and a gatesim JSON report under `experiment-specs/`. The shared simulator is
`experiment-specs/v2pack-gatesim.py`, SHA-256
`ec23cf6e762d7b65f8740d770646e34698cc9d1a8ad2cb6a98fcb6a0b4b5c5d5`.

Reproduce and byte-verify all reports with:

```bash
python3 experiment-specs/v2pack-gatesim.py --verify
```

The simulator intentionally has no write or launch mode. It validates gate
evaluator behavior and power under declared alternatives; it does not generate
scientific evidence.

The evidence inventory is available as:

- `experiment-specs/v2-evidence-package-manifest.json`;
- `experiment-specs/v2-evidence-package-manifest.md`.

It identifies the missing completed G9 joint readout, the stale G9 archive
substitutes, missing canonical analyzers and Lean sources, mechanism-record
gaps, and the exact 2,238-versus-2,244 G3 bootstrap-path adjudication.

## Authority boundary

Every registration says `DESIGN_AND_GATESIM_ONLY` and `NO_LAUNCH_AUTHORITY`.
This pack creates no result root, attempt directory, launch manifest, node
controller, queue, or H200 command. A future launch requires a separate,
explicit pre-outcome authority plus the frozen analyzers and materialized
inputs required by each contract. No gatesim probability is a scientific
result.

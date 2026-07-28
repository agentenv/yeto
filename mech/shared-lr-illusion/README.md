# Shared-LR illusion analysis

This directory contains a CPU-only, post-hoc descriptive audit of sealed outer
momentum data. It does not launch training, change a scientific gate, or claim
that the shared-LR estimand was preregistered.

Artifacts:

- `table.md`: complete human-readable table, result paragraph, eligibility
  audit, source hashes, and caveats;
- `summary.md`: the requested one-paragraph count summary;
- `table.tex`: paper-ready landscape `longtable` (requires `booktabs`,
  `longtable`, `threeparttablex`, and `pdflscape`);
- `subsection.tex`: short paper subsection draft;
- `table.csv`: flat machine-readable estimates;
- `results.json`: unrounded estimates, grids, fit status, bootstrap-valid
  counts, exclusions, source hashes, raw-snapshot validation, and 60
  frozen-fit parity checks; and
- `analyze.py`: deterministic reproducer.

The reproducer requires Python 3.10+ and NumPy. Pass explicit paths to the
sealed artifacts rather than relying on mutable campaign directories:

```sh
python mech/shared-lr-illusion/analyze.py \
  --two-param-root /path/to/two-param-analysis \
  --g3-readout /path/to/g3-readout.json \
  --g4c-readout /path/to/g4c-readout.json \
  --g6-readout /path/to/g6-readout.json \
  --g8-readout /path/to/g8-readout.json \
  --output-dir mech/shared-lr-illusion
```

The script fails closed on incomplete eta-by-seed grids, unpaired arms, source
schema changes, raw snapshot/hash mismatches, or point fits that do not
reproduce the frozen readouts. It traces all 72 pilot and 460 v3 seed-level CSV
rows back to the raw banked results and verifies all 1,557 entries in the
recorded raw-source hash lists. It uses the original paired training-seed
bootstrap pattern, 10,000 refits, campaign RNG engines, and campaign RNG seeds.
The new gain/flip estimands themselves are explicitly post-hoc.

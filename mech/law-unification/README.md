# Reproducing the law-unification audit

This directory is self-contained apart from its declared Python packages. It reads only the frozen files in `sources/`; it launches no training and makes no network calls.

From the repository root:

```bash
uv run --no-project -p python3.13 --with numpy --with scipy --with matplotlib \
  python mech/law-unification/analyze.py
```

The script verifies every source SHA-256, reconstructs the G4C/G6/G8/G12 eta-star intervals with the original paired seed-resampling seeds, assembles both ledgers, runs OLS/WLS and nested sharing tests, applies the verdict rule frozen in `INCLUSION.md`, and regenerates the CSV/JSON/TeX/PNG/PDF/writeup artifacts. It also writes the requested one-line handoff to `/private/tmp/h200-law-note.md`; pass `--law-note PATH` to change that location.

The primary estimate is equal-point OLS in natural-log space. Fixed-effect WLS uses `SE(log eta) = [log(CI_high)-log(CI_low)]/(2*1.959964)` only for nondegenerate intervals with zero invalid bootstrap refits. This intentionally prevents singleton 7B intervals or qualified conditional intervals from receiving infinite or misleading weight.

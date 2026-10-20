"""yeto RL efficiency evaluation (S26 E0.1 / E0.2).

Eight arms per model preset: {bare, yeto} x {verl, miles} x {sync, async}. The
bare arms run the framework's own entry point from the same image (no yeto
code on their path); the yeto arms run the same framework through yeto. Every
arm is generated from one :class:`.spec.EffSpec`, so model, data, order seed,
reward, hyperparameters, max tokens, card type and count and image are equal
by construction (:func:`.spec.control_fingerprint`).

    python -m tools.eval eff-plan qwen3-0.6b [--arm ARM]
    python -m tools.eval eff-run qwen3-0.6b --arm bare-verl-sync --run-id ID [--dry-run]
    python -m tools.eval eff-parse RUN_DIR
    python -m tools.eval eff-report RUN_DIR ... --out report.md

* :mod:`.spec` -- presets, arms, the per-arm command plans;
* :mod:`.parse` -- framework logs and yeto tapes -> one per-round record format;
* :mod:`.metrics` -- per-run summary (throughput, segments, cost, off-policy);
* :mod:`.report` -- Markdown report with yeto-vs-bare overhead and range verdicts;
* :mod:`.modal_eff` -- the Modal harness (bare arms and yeto+verl);
  yeto+Miles runs through the yeto launcher (``tools/eval/miles_local.sh``).

Raw data is written to ``<run dir>/raw/`` before anything is computed;
``parsed.json`` and the report are derived from it and can be rebuilt.
"""

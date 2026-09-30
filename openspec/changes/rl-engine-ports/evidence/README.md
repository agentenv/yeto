# Evidence (trimmed)

Raw GPU evidence for rl-engine-ports R0, trimmed before merge to keep the repo small:

- `*.parquet` dataset shards were removed (regenerable from GSM8K).
- Logs/JSON(L) over 64 KB and 400 lines keep the first 150 and last 250 lines, with a
  `[truncated: ...]` marker in between.
- Reports (`report.md`), scripts, small JSON summaries and hashes are unchanged.

The untrimmed evidence is kept out of tree (`rl-engine-ports-evidence-full.tar.gz`).

# Archive listing and narrow extraction manifest

Host: `c@65.19.161.135` (`dev16`)

The archive was completely listed before extraction.  The completed listing
marker predates the extraction manifest and extraction-complete marker:

| event/artifact | UTC timestamp | size | SHA-256 |
|---|---:|---:|---|
| archive `evac2-heuristic-shockley.tar.zst` | 2026-07-27 20:35:16.753285537 | 225,427,003,969 bytes | not rescanned (source archive is immutable/read-only) |
| complete archive table of contents | 2026-07-29 01:09:09.941959897 | 784,601 bytes / 8,892 lines | `7960593425e5ac2ada860bcc112c463a574c957b85111d658a0b96abaeb8ccb7` |
| seven-member extraction list | 2026-07-29 01:14:27.973899966 | 559 bytes / 7 lines | `1d48e80902082ecb57deb8ba98afebd1dd5a7bf29acbd1fd643e7280fce0386a` |
| extraction complete | 2026-07-29 01:20:27.021183973 | exactly 7 regular files | checksum-file SHA-256 `31c2cd3499f36e5e2bbae75558131414360b198b32b3ed73123f70264c7c0b9b` |

The sorted relative paths found below the extraction root compare byte-for-byte
equal to `archive-members.txt`; no other archive member was extracted.

| extracted checkpoint member | bytes | SHA-256 |
|---|---:|---|
| `data/yeto-results-v8/v8-t20-corrected-mu0p80-e2-seed811/attempt-1/work/m4/state.ckpt` | 1,076,137,912 | `a29e3a10682e2cf97dbbdecbf3116b943198ca1271623768b872ce64eea7de22` |
| `data/yeto-results-v8/v8-t20-corrected-mu0p95-e2-seed801/attempt-1/work/m4/state.ckpt` | 1,076,137,912 | `54dfda98c2ad0b2983e67c55808b7af408381fd46169aef9d3c395c5c49c809f` |
| `data/yeto-results-v8/v8-t20-mu0-mu0-e2-seed801/attempt-1/work/m4/state.ckpt` | 1,076,137,912 | `1181ba52bf4f168e604f93d3c0b1ae0d4264199710196ca44b8be92f350fd2a5` |
| `data/yeto-results-v8/v8-t20-mu0-mu0-e2-seed811/attempt-1/work/m4/state.ckpt` | 1,076,137,912 | `4ae4fc7663947b6f017151a1a86c2d58183ee6538d18caf191151e409b801e09` |
| `data/yeto-results-v8/v8-t20-raw-mu0p80-e2-seed801/attempt-1/work/m4/state.ckpt` | 1,076,137,912 | `ce61ff2103c922b9eccf5d1901ebd673697eeca483be03df76b98c534c4e5919` |
| `data/yeto-results-v8/v8-t20-raw-mu0p95-e2-seed801/attempt-1/work/m4/state.ckpt` | 1,076,137,912 | `6f47c2bdbd3187be9e93a147750d686d8388ce42be5121b3911fd137384af6e0` |
| `data/yeto-results-v8/v8-t20-raw-mu0p95-e2-seed811/attempt-1/work/m4/state.ckpt` | 1,076,137,912 | `cd31fb656fd3ed06a330e3202b620d71a7a84a3e95d36569f5612af79d103fdc` |

Extraction root:
`/mnt/nvme1/yeto-mech-discriminator-20260728/checkpoints/archive/`.
The source archive and capped trees were not modified.

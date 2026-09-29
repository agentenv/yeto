# G3 rerun 3 results (two Modal H100! islands, local syncer on :29410, strict-avg; launcher-returned tapes)

| run | window (UTC) | criteria 1-4 | criterion 5 (checker) | verdict |
| --- | --- | --- | --- | --- |
| tis | 19:54:40 to 20:09:08 | all PASS (3 steps with 2 responders and 0 stale; same sha; v0-v3 publication hashes equal across islands; 3 finite rounds each) | PASS | **PASS** |
| icepop [0.5, 5] | 20:10:53 to 20:34:09 | all PASS | checker FAIL: the streamed log line for island l0 step 0 (launch.log:9053) lost its `[algo1a-g3c-icepop-l0-modal]` prefix to interleaving, so the prefix regex does not attribute it | see note |

Note on icepop criterion 5. This is a fact check, not a new criterion. The unprefixed line comes
from `MegatronTrainRayActor pid=3062`, and all 259 other lines of that pid carry the l0 prefix. The
line contains train/tis 0.99950, train/tis_abs 0.010649 and train/tis_clipfrac 0.0. With it, both
islands have all three keys for steps 0-2. The pre-declared checker result stays FAIL and is
recorded as such; whether the prefix-less line counts is left to the coordinator. The IcePop
verdict is therefore "criteria 1-4 PASS, criterion 5 FAIL by the checker, satisfied by the pid
attribution above".

Resources: 2x H100 x (14.5 + 23.3) min = ~1.26 H100 h, ~ $5 (estimate). Afterwards
`modal container list` shows no algo1a container, port 29410 is closed, and the watchdogs of both
runs are stopped.

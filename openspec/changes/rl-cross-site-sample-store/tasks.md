# Tasks

Codes follow `infra-drafts/SESSION25/GOAL.md` §3 (v2). CPU-first tasks need no NPU or PPU. Effort is in sub-agent person-days and is an estimate.

## N3. Cross-site store (OSS)

- [x] N3.1 `OssStore` (design D1) and sharded buffer with numbered index objects (design D2, D3). Acceptance: CPU tests on a local directory and a fake S3 server; old single-file buffers readable. CPU-first: yes. Estimate: 3 d
- [x] N3.2 Release commit order and reader check on object stores (design D4); W4A8 releases use the same check. Acceptance: CPU test that a reader never loads a partial release. CPU-first: yes. Estimate: 1.5 d
- [x] N3.3 Trainer scheduler accepts any reachable cut store; reachability check at launch (design D5). Prerequisite of stage 1, because the trainer is not on Modal. Acceptance: CPU tests; Modal-volume runs unchanged. CPU-first: yes. Estimate: 1 d
- [x] N3.4 Multipart upload settings for releases of about 300 GB (design D6). Acceptance: CPU test with a fake S3 server and a large sparse file. CPU-first: yes. Estimate: 0.5–1 d
- [x] N3.6 Head backlog view of a sharded buffer: the head reads only the merged index, so with the scheduler on, samples that the trainer has not merged yet do not count. Wire it with the ssh trainer binding (N7.3). CPU-first: yes. Done in branch s25-n7-trainer: `UnmergedShards` (read-only) counted by `CommittedBacklog` in the sky and SSH bindings; tests/test_rl_ssh_trainer_island.py.
- [ ] N3.5 Measure OSS transfer: upload and download one W4A8 release (about 301 GB) from the submitting server and from one cluster node; later from one PPU machine. Acceptance: numbers in the review document. Needs: OSS keys from the user; cluster nodes (no cards used). Estimate: 0.5 d

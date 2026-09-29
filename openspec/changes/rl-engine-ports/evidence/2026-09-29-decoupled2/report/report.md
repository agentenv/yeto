# Miles RL LM benchmark: Qwen/Qwen3-0.6B

Rounds: 8; seeds: 17; held-out prompts: 8; samples/prompt: 1

## Quality

| arm | M | runs | reward | pass@1 | delta vs native | delta vs single | delta vs strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| yeto-decoupled-m2 | 2 | 1 | 0.6250 +/- 0.0000 | 0.6250 | - | - | - |

## Systems

| arm | GPUs | train s | artifact-ready s | eval s | traj/s | action tok/s | active GPU-s | active % | util avg/min % | GPU-h | cost | hook s | final s | H | PULL-to-PUSH s | BCAST queue s | sync ms | filter generated | filter dropped | replacements | fragment payload MB | KL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yeto-decoupled-m2 | 2 | 407.7 | 412.1 | 95.8 | 1.26 | 293.0 | 341.5 | 0.419 | 15.3/15.3 | 0.253 | - | 36.68 | 21.26 | 6.75 | 31.624 | 18.201 | 33231.88 | 0.0 | 0.0 | 0.0 | 449.118 | - |

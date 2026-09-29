# Miles RL LM benchmark: Qwen/Qwen3-0.6B

Rounds: 8; seeds: 19; held-out prompts: 8; samples/prompt: 1

## Quality

| arm | M | runs | reward | pass@1 | delta vs native | delta vs single | delta vs strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| yeto-decoupled-m2 | 2 | 1 | 0.6250 +/- 0.0000 | 0.6250 | - | - | - |

## Systems

| arm | GPUs | train s | artifact-ready s | eval s | traj/s | action tok/s | active GPU-s | active % | util avg/min % | GPU-h | cost | hook s | final s | H | PULL-to-PUSH s | BCAST queue s | sync ms | filter generated | filter dropped | replacements | fragment payload MB | KL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yeto-decoupled-m2 | 2 | 467.4 | 473.9 | 86.4 | 1.10 | 240.1 | 299.0 | 0.320 | 12.8/12.6 | 0.284 | - | 37.59 | 21.75 | 6.75 | 28.479 | 16.659 | 29681.44 | 0.0 | 0.0 | 0.0 | 454.164 | - |

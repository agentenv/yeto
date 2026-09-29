# Miles RL LM benchmark: Qwen/Qwen3-0.6B

Rounds: 1; seeds: 17; held-out prompts: 8; samples/prompt: 1

## Quality

| arm | M | runs | reward | pass@1 | delta vs native | delta vs single | delta vs strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| yeto-decoupled-m2 | 2 | 1 | 0.8750 +/- 0.0000 | 0.8750 | - | - | - |

## Systems

| arm | GPUs | train s | artifact-ready s | eval s | traj/s | action tok/s | active GPU-s | active % | util avg/min % | GPU-h | cost | hook s | final s | H | PULL-to-PUSH s | BCAST queue s | sync ms | filter generated | filter dropped | replacements | fragment payload MB | KL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yeto-decoupled-m2 | 2 | 331.7 | 341.7 | 115.5 | 0.19 | 34.3 | 64.5 | 0.097 | 2.3/1.9 | 0.216 | - | 11.35 | 11.35 | 1.00 | 0.015 | - | 296.88 | 8.0 | 0.0 | 0.0 | 307.823 | 0.00064 |

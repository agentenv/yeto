# Miles RL LM benchmark: Qwen/Qwen3-0.6B

Rounds: 8; seeds: 20; held-out prompts: 8; samples/prompt: 1

## Quality

| arm | M | runs | reward | pass@1 | delta vs native | delta vs single | delta vs strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| yeto-decoupled-m2 | 2 | 1 | 0.8750 +/- 0.0000 | 0.8750 | - | - | - |

## Systems

| arm | GPUs | train s | artifact-ready s | eval s | traj/s | action tok/s | active GPU-s | active % | util avg/min % | GPU-h | cost | hook s | final s | H | PULL-to-PUSH s | BCAST queue s | sync ms | filter generated | filter dropped | replacements | fragment payload MB | KL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yeto-decoupled-m2 | 2 | 469.6 | 475.5 | 71.0 | 1.09 | 228.4 | 282.9 | 0.301 | 12.6/12.4 | 0.281 | - | 31.37 | 16.67 | 6.75 | 29.607 | 15.889 | 30813.50 | 0.0 | 0.0 | 0.0 | 449.118 | - |

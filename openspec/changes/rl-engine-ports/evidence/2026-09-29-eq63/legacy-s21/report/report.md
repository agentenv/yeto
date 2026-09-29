# Miles RL LM benchmark: Qwen/Qwen3-0.6B

Rounds: 8; seeds: 21; held-out prompts: 8; samples/prompt: 1

## Quality

| arm | M | runs | reward | pass@1 | delta vs native | delta vs single | delta vs strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| yeto-decoupled-m2 | 2 | 1 | 0.7500 +/- 0.0000 | 0.7500 | - | - | - |

## Systems

| arm | GPUs | train s | artifact-ready s | eval s | traj/s | action tok/s | active GPU-s | active % | util avg/min % | GPU-h | cost | hook s | final s | H | PULL-to-PUSH s | BCAST queue s | sync ms | filter generated | filter dropped | replacements | fragment payload MB | KL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yeto-decoupled-m2 | 2 | 534.5 | 543.5 | 104.7 | 0.96 | 190.2 | 401.0 | 0.375 | 16.4/15.9 | 0.326 | - | 24.22 | 11.54 | 6.75 | 22.045 | 23.209 | 22566.31 | 64.0 | 0.0 | 0.0 | 449.118 | 0.00067 |

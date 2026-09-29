# Miles RL LM benchmark: Qwen/Qwen3-0.6B

Rounds: 8; seeds: 19; held-out prompts: 8; samples/prompt: 1

## Quality

| arm | M | runs | reward | pass@1 | delta vs native | delta vs single | delta vs strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| yeto-decoupled-m2 | 2 | 1 | 0.6250 +/- 0.0000 | 0.6250 | - | - | - |

## Systems

| arm | GPUs | train s | artifact-ready s | eval s | traj/s | action tok/s | active GPU-s | active % | util avg/min % | GPU-h | cost | hook s | final s | H | PULL-to-PUSH s | BCAST queue s | sync ms | filter generated | filter dropped | replacements | fragment payload MB | KL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yeto-decoupled-m2 | 2 | 516.8 | 525.4 | 118.3 | 0.99 | 217.5 | 402.1 | 0.389 | 17.6/17.6 | 0.320 | - | 25.38 | 12.84 | 6.75 | 22.127 | 23.023 | 23104.75 | 64.0 | 0.0 | 0.0 | 449.118 | 0.00062 |

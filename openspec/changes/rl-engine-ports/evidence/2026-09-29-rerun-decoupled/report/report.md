# Miles RL LM benchmark: Qwen/Qwen3-0.6B

Rounds: 8; seeds: 17; held-out prompts: 8; samples/prompt: 1

## Quality

| arm | M | runs | reward | pass@1 | delta vs native | delta vs single | delta vs strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| yeto-decoupled-m2 | 2 | 1 | 0.6250 +/- 0.0000 | 0.6250 | - | - | - |

## Systems

| arm | GPUs | train s | artifact-ready s | eval s | traj/s | action tok/s | active GPU-s | active % | util avg/min % | GPU-h | cost | hook s | final s | H | PULL-to-PUSH s | BCAST queue s | sync ms | filter generated | filter dropped | replacements | fragment payload MB | KL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yeto-decoupled-m2 | 2 | 504.9 | 511.5 | 106.4 | 1.01 | 236.6 | 313.3 | 0.310 | 12.3/12.1 | 0.310 | - | 40.89 | 24.76 | 6.75 | 31.402 | 16.122 | 33806.94 | 0.0 | 0.0 | 0.0 | 454.164 | - |

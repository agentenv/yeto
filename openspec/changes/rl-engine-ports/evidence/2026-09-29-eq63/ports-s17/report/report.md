# Miles RL LM benchmark: Qwen/Qwen3-0.6B

Rounds: 8; seeds: 17; held-out prompts: 8; samples/prompt: 1

## Quality

| arm | M | runs | reward | pass@1 | delta vs native | delta vs single | delta vs strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| yeto-decoupled-m2 | 2 | 1 | 0.6250 +/- 0.0000 | 0.6250 | - | - | - |

## Systems

| arm | GPUs | train s | artifact-ready s | eval s | traj/s | action tok/s | active GPU-s | active % | util avg/min % | GPU-h | cost | hook s | final s | H | PULL-to-PUSH s | BCAST queue s | sync ms | filter generated | filter dropped | replacements | fragment payload MB | KL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yeto-decoupled-m2 | 2 | 544.0 | 550.4 | 90.6 | 0.94 | 219.6 | 303.9 | 0.279 | 11.8/11.6 | 0.327 | - | 28.22 | 12.93 | 6.75 | 35.504 | 17.481 | 36250.56 | 0.0 | 0.0 | 0.0 | 454.164 | - |

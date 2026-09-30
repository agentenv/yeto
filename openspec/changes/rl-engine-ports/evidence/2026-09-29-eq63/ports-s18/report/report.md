# Miles RL LM benchmark: Qwen/Qwen3-0.6B

Rounds: 8; seeds: 18; held-out prompts: 8; samples/prompt: 1

## Quality

| arm | M | runs | reward | pass@1 | delta vs native | delta vs single | delta vs strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| yeto-decoupled-m2 | 2 | 1 | 0.6250 +/- 0.0000 | 0.6250 | - | - | - |

## Systems

| arm | GPUs | train s | artifact-ready s | eval s | traj/s | action tok/s | active GPU-s | active % | util avg/min % | GPU-h | cost | hook s | final s | H | PULL-to-PUSH s | BCAST queue s | sync ms | filter generated | filter dropped | replacements | fragment payload MB | KL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yeto-decoupled-m2 | 2 | 468.4 | 474.5 | 89.7 | 1.09 | 255.3 | 302.1 | 0.322 | 12.7/12.5 | 0.285 | - | 34.04 | 14.13 | 6.75 | 31.086 | 16.784 | 32003.12 | 0.0 | 0.0 | 0.0 | 454.164 | - |

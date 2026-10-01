# Miles RL LM benchmark: Qwen/Qwen3-0.6B

Rounds: 4; seeds: 17; held-out prompts: 8; samples/prompt: 1

## Quality

| arm | M | runs | reward | pass@1 | delta vs native | delta vs single | delta vs strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| yeto-decoupled-m2 | 2 | 1 | 0.7500 +/- 0.0000 | 0.7500 | - | - | - |

## Systems

| arm | GPUs | train s | artifact-ready s | eval s | traj/s | action tok/s | active GPU-s | active % | util avg/min % | GPU-h | cost | hook s | final s | H | PULL-to-PUSH s | BCAST queue s | sync ms | filter generated | filter dropped | replacements | fragment payload MB | KL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yeto-decoupled-m2 | 2 | 607.8 | 618.7 | 136.4 | 0.42 | 101.2 | 274.9 | 0.226 | 12.2/12.0 | 0.376 | - | 70.75 | 43.57 | 3.25 | 42.768 | 37.826 | 43896.75 | 32.0 | 0.0 | 0.0 | 423.887 | 0.00069 |

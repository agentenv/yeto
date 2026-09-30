# Miles RL LM benchmark: Qwen/Qwen3-0.6B

Rounds: 3; seeds: 17; held-out prompts: 8; samples/prompt: 1

## Quality

| arm | M | runs | reward | pass@1 | delta vs native | delta vs single | delta vs strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| yeto-federated-m2 | 2 | 1 | 0.8750 +/- 0.0000 | 0.8750 | - | - | - |

## Systems

| arm | GPUs | train s | artifact-ready s | eval s | traj/s | action tok/s | active GPU-s | active % | util avg/min % | GPU-h | cost | hook s | final s | H | PULL-to-PUSH s | BCAST queue s | sync ms | filter generated | filter dropped | replacements | fragment payload MB | KL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yeto-federated-m2 | 2 | 396.2 | 405.7 | 121.5 | 0.48 | 113.4 | 171.7 | 0.217 | 9.3/8.9 | 0.254 | - | 0.00 | 0.00 | - | - | - | 32349.67 | 24.0 | 0.0 | 0.0 | - | 0.00061 |

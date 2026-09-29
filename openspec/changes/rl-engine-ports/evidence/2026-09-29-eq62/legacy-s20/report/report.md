# Miles RL LM benchmark: Qwen/Qwen3-0.6B

Rounds: 3; seeds: 20; held-out prompts: 8; samples/prompt: 1

## Quality

| arm | M | runs | reward | pass@1 | delta vs native | delta vs single | delta vs strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| yeto-federated-m2 | 2 | 1 | 0.7500 +/- 0.0000 | 0.7500 | - | - | - |

## Systems

| arm | GPUs | train s | artifact-ready s | eval s | traj/s | action tok/s | active GPU-s | active % | util avg/min % | GPU-h | cost | hook s | final s | H | PULL-to-PUSH s | BCAST queue s | sync ms | filter generated | filter dropped | replacements | fragment payload MB | KL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yeto-federated-m2 | 2 | 401.7 | 411.3 | 123.8 | 0.48 | 112.3 | 196.3 | 0.244 | 9.7/9.4 | 0.258 | - | 0.00 | 0.00 | - | - | - | 35803.33 | 24.0 | 0.0 | 0.0 | - | 0.00061 |

# Miles RL LM benchmark: Qwen/Qwen3-0.6B

Rounds: 3; seeds: 17; held-out prompts: 8; samples/prompt: 1

## Quality

| arm | M | runs | reward | pass@1 | delta vs native | delta vs single | delta vs strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| yeto-federated-m2 | 2 | 1 | 0.3750 +/- 0.0000 | 0.3750 | - | - | - |

## Systems

| arm | GPUs | train s | artifact-ready s | eval s | traj/s | action tok/s | active GPU-s | active % | util avg/min % | GPU-h | cost | hook s | final s | H | PULL-to-PUSH s | BCAST queue s | sync ms | filter generated | filter dropped | replacements | fragment payload MB | KL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yeto-federated-m2 | 2 | 794.4 | 800.8 | 102.2 | 0.24 | 59.4 | 151.7 | 0.095 | 3.3/3.3 | 0.470 | - | 0.00 | 0.00 | - | - | - | 169054.00 | 0.0 | 0.0 | 0.0 | - | - |

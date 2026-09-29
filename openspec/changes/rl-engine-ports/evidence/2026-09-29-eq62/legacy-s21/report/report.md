# Miles RL LM benchmark: Qwen/Qwen3-0.6B

Rounds: 3; seeds: 21; held-out prompts: 8; samples/prompt: 1

## Quality

| arm | M | runs | reward | pass@1 | delta vs native | delta vs single | delta vs strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| yeto-federated-m2 | 2 | 1 | 0.7500 +/- 0.0000 | 0.7500 | - | - | - |

## Systems

| arm | GPUs | train s | artifact-ready s | eval s | traj/s | action tok/s | active GPU-s | active % | util avg/min % | GPU-h | cost | hook s | final s | H | PULL-to-PUSH s | BCAST queue s | sync ms | filter generated | filter dropped | replacements | fragment payload MB | KL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yeto-federated-m2 | 2 | 405.2 | 415.2 | 121.9 | 0.47 | 112.7 | 182.1 | 0.225 | 9.5/9.5 | 0.259 | - | 0.00 | 0.00 | - | - | - | 33462.00 | 24.0 | 0.0 | 0.0 | - | 0.00063 |

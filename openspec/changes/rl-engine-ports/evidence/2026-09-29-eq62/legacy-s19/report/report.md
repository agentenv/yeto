# Miles RL LM benchmark: Qwen/Qwen3-0.6B

Rounds: 3; seeds: 19; held-out prompts: 8; samples/prompt: 1

## Quality

| arm | M | runs | reward | pass@1 | delta vs native | delta vs single | delta vs strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| yeto-federated-m2 | 2 | 1 | 0.6250 +/- 0.0000 | 0.6250 | - | - | - |

## Systems

| arm | GPUs | train s | artifact-ready s | eval s | traj/s | action tok/s | active GPU-s | active % | util avg/min % | GPU-h | cost | hook s | final s | H | PULL-to-PUSH s | BCAST queue s | sync ms | filter generated | filter dropped | replacements | fragment payload MB | KL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yeto-federated-m2 | 2 | 380.8 | 390.1 | 135.6 | 0.50 | 120.7 | 182.0 | 0.239 | 10.2/10.1 | 0.249 | - | 0.00 | 0.00 | - | - | - | 32979.33 | 24.0 | 0.0 | 0.0 | - | 0.00062 |

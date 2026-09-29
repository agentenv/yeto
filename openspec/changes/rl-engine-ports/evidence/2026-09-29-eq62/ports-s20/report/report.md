# Miles RL LM benchmark: Qwen/Qwen3-0.6B

Rounds: 3; seeds: 20; held-out prompts: 8; samples/prompt: 1

## Quality

| arm | M | runs | reward | pass@1 | delta vs native | delta vs single | delta vs strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| yeto-federated-m2 | 2 | 1 | 0.7500 +/- 0.0000 | 0.7500 | - | - | - |

## Systems

| arm | GPUs | train s | artifact-ready s | eval s | traj/s | action tok/s | active GPU-s | active % | util avg/min % | GPU-h | cost | hook s | final s | H | PULL-to-PUSH s | BCAST queue s | sync ms | filter generated | filter dropped | replacements | fragment payload MB | KL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yeto-federated-m2 | 2 | 603.0 | 608.8 | 75.8 | 0.32 | 75.6 | 129.9 | 0.108 | 4.1/3.9 | 0.356 | - | 0.00 | 0.00 | - | - | - | 124007.67 | 0.0 | 0.0 | 0.0 | - | - |

# Miles RL LM benchmark: Qwen/Qwen3-0.6B

Rounds: 1; seeds: 17; held-out prompts: 8; samples/prompt: 1

## Quality

| arm | M | runs | reward | pass@1 | delta vs native | delta vs single | delta vs strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| yeto-federated-m2 | 2 | 1 | 0.8750 +/- 0.0000 | 0.8750 | - | - | - |

## Systems

| arm | GPUs | train s | artifact-ready s | eval s | traj/s | action tok/s | active GPU-s | active % | util avg/min % | GPU-h | cost | hook s | final s | H | PULL-to-PUSH s | BCAST queue s | sync ms | filter generated | filter dropped | replacements | fragment payload MB | KL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yeto-federated-m2 | 2 | 406.8 | 412.9 | 77.6 | 0.16 | 28.0 | 58.7 | 0.072 | 1.4/1.4 | 0.248 | - | 0.00 | 0.00 | - | - | - | 182221.00 | 0.0 | 0.0 | 0.0 | - | - |

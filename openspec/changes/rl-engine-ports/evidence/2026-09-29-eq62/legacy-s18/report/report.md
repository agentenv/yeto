# Miles RL LM benchmark: Qwen/Qwen3-0.6B

Rounds: 3; seeds: 18; held-out prompts: 8; samples/prompt: 1

## Quality

| arm | M | runs | reward | pass@1 | delta vs native | delta vs single | delta vs strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| yeto-federated-m2 | 2 | 1 | 0.6250 +/- 0.0000 | 0.6250 | - | - | - |

## Systems

| arm | GPUs | train s | artifact-ready s | eval s | traj/s | action tok/s | active GPU-s | active % | util avg/min % | GPU-h | cost | hook s | final s | H | PULL-to-PUSH s | BCAST queue s | sync ms | filter generated | filter dropped | replacements | fragment payload MB | KL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yeto-federated-m2 | 2 | 413.6 | 423.5 | 126.8 | 0.46 | 113.5 | 182.3 | 0.220 | 9.3/9.1 | 0.265 | - | 0.00 | 0.00 | - | - | - | 33936.67 | 24.0 | 0.0 | 0.0 | - | 0.00062 |

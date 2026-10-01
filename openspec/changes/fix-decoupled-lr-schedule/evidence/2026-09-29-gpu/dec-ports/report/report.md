# Miles RL LM benchmark: Qwen/Qwen3-0.6B

Rounds: 4; seeds: 17; held-out prompts: 8; samples/prompt: 1

## Quality

| arm | M | runs | reward | pass@1 | delta vs native | delta vs single | delta vs strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| yeto-decoupled-m2 | 2 | 1 | 0.8750 +/- 0.0000 | 0.8750 | - | - | - |

## Systems

| arm | GPUs | train s | artifact-ready s | eval s | traj/s | action tok/s | active GPU-s | active % | util avg/min % | GPU-h | cost | hook s | final s | H | PULL-to-PUSH s | BCAST queue s | sync ms | filter generated | filter dropped | replacements | fragment payload MB | KL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yeto-decoupled-m2 | 2 | 496.5 | 504.8 | 114.3 | 0.52 | 123.2 | 212.5 | 0.214 | 8.1/8.0 | 0.308 | - | 79.46 | 51.29 | 3.25 | 42.595 | 28.836 | 45184.12 | 0.0 | 0.0 | 0.0 | 423.887 | - |

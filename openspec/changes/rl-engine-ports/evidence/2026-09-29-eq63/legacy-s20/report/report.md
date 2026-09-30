# Miles RL LM benchmark: Qwen/Qwen3-0.6B

Rounds: 8; seeds: 20; held-out prompts: 8; samples/prompt: 1

## Quality

| arm | M | runs | reward | pass@1 | delta vs native | delta vs single | delta vs strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| yeto-decoupled-m2 | 2 | 1 | 0.5000 +/- 0.0000 | 0.5000 | - | - | - |

## Systems

| arm | GPUs | train s | artifact-ready s | eval s | traj/s | action tok/s | active GPU-s | active % | util avg/min % | GPU-h | cost | hook s | final s | H | PULL-to-PUSH s | BCAST queue s | sync ms | filter generated | filter dropped | replacements | fragment payload MB | KL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yeto-decoupled-m2 | 2 | 516.9 | 526.3 | 127.6 | 0.99 | 225.4 | 422.5 | 0.409 | 18.9/18.3 | 0.323 | - | 24.56 | 11.50 | 6.75 | 22.827 | 24.395 | 23466.81 | 64.0 | 0.0 | 0.0 | 454.164 | 0.00063 |

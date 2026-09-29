# Miles RL LM benchmark: Qwen/Qwen3-0.6B

Rounds: 8; seeds: 17; held-out prompts: 8; samples/prompt: 1

## Quality

| arm | M | runs | reward | pass@1 | delta vs native | delta vs single | delta vs strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| yeto-decoupled-m2 | 2 | 1 | 0.5000 +/- 0.0000 | 0.5000 | - | - | - |

## Systems

| arm | GPUs | train s | artifact-ready s | eval s | traj/s | action tok/s | active GPU-s | active % | util avg/min % | GPU-h | cost | hook s | final s | H | PULL-to-PUSH s | BCAST queue s | sync ms | filter generated | filter dropped | replacements | fragment payload MB | KL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yeto-decoupled-m2 | 2 | 688.6 | 698.3 | 141.4 | 0.74 | 176.8 | 427.4 | 0.310 | 14.2/14.2 | 0.422 | - | 35.50 | 22.18 | 6.77 | 26.025 | 18.706 | 29948.47 | 64.0 | 0.0 | 0.0 | 444.072 | 0.00063 |

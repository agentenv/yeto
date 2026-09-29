# Miles RL LM benchmark: Qwen/Qwen3-0.6B

Rounds: 8; seeds: 18; held-out prompts: 8; samples/prompt: 1

## Quality

| arm | M | runs | reward | pass@1 | delta vs native | delta vs single | delta vs strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| yeto-decoupled-m2 | 2 | 1 | 0.6250 +/- 0.0000 | 0.6250 | - | - | - |

## Systems

| arm | GPUs | train s | artifact-ready s | eval s | traj/s | action tok/s | active GPU-s | active % | util avg/min % | GPU-h | cost | hook s | final s | H | PULL-to-PUSH s | BCAST queue s | sync ms | filter generated | filter dropped | replacements | fragment payload MB | KL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yeto-decoupled-m2 | 2 | 529.2 | 542.9 | 131.6 | 0.97 | 224.3 | 422.2 | 0.399 | 18.3/18.0 | 0.331 | - | 25.26 | 12.33 | 6.75 | 23.336 | 24.384 | 24084.38 | 64.0 | 0.0 | 0.0 | 449.118 | 0.00065 |

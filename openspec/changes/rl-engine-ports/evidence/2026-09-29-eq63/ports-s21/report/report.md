# Miles RL LM benchmark: Qwen/Qwen3-0.6B

Rounds: 8; seeds: 21; held-out prompts: 8; samples/prompt: 1

## Quality

| arm | M | runs | reward | pass@1 | delta vs native | delta vs single | delta vs strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| yeto-decoupled-m2 | 2 | 1 | 0.6250 +/- 0.0000 | 0.6250 | - | - | - |

## Systems

| arm | GPUs | train s | artifact-ready s | eval s | traj/s | action tok/s | active GPU-s | active % | util avg/min % | GPU-h | cost | hook s | final s | H | PULL-to-PUSH s | BCAST queue s | sync ms | filter generated | filter dropped | replacements | fragment payload MB | KL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yeto-decoupled-m2 | 2 | 472.8 | 478.6 | 73.4 | 1.08 | 225.7 | 290.2 | 0.307 | 12.6/12.4 | 0.283 | - | 30.80 | 15.37 | 6.75 | 29.927 | 16.585 | 31062.06 | 0.0 | 0.0 | 0.0 | 454.164 | - |

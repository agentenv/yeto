# 2.1 pg_clipfrac：compile 与 eager 离线对照（CPU，2026-10-09，未上卡）

## 结论
**不能复现。** 在 CPU 上，`compute_policy_loss` 的 compile 与 eager 两种模式给出的 clipfrac 逐位相同，并且都等于手算结果。eps_clip 等于 eps_clip_high（包括两个参数是同一个 Python 浮点对象）和不等于两种情形都一样。所以 CPU 上没有证据表明 fork 有缺陷。2.2 按"不是已确认的 fork 缺陷"处理，不改 fork。

## 做了什么
- 代码：Miles fork `ddce20992c9e`（当前 pin `MILES_NEXT_COMMIT`）的 `miles/backends/training_utils/loss_hub/math_utils.py:254` `compute_policy_loss`（`@torch.compile(dynamic=True)`）。eager 用 `_torchdynamo_orig_callable` 取原函数。
- 环境：`/home/michael/work/miles-next-venv`，torch 2.13.0+cu130，CPU（inductor CPU 后端）。
- 张量：n=4096，`ppo_kl ~ N(0, 0.01)`，`A ~ N(0,1)`，seed 0；float32 与 bfloat16 各一组。
- 情形：eps (0.001, 0.001)、(0.001, 10)、(0.2, 0.2)、(0.2, 0.28)；相等时分别用同一对象和不同对象传参。另用一个新的 compiled 对象按"先不等、后相等"的顺序调用，排除 dynamo 特化顺序的影响。
- 重跑：`PYTHONPATH=/home/michael/work/miles-algosup OMP_NUM_THREADS=1 /home/michael/work/miles-next-venv/bin/python check_clipfrac_compile.py`，输出见 `output.txt`。

## 关键数字（float32）
| 情形 | 手算 | eager | compiled | A>0 且 ratio>1+hi 的比例 |
|---|---|---|---|---|
| eq 0.001/0.001 | 0.452393 | 0.452393 | 0.452393 | 0.228516 |
| neq 0.001/10 | 0.223877 | 0.223877 | 0.223877 | 0 |

eq 与 neq 的差正好是上界贡献，与 2026-09-29 的离线手算（grpo-knobs evidence）一致。pg_loss 在 compile 与 eager 之间不逐位相同（融合运算，最大差 2.4e-7），clipfrac 逐位相同。

## 没有排除的解释（g1e 中 A 组第 2 步 pg_clipfrac 与 B 组逐位相同）
以下都是推测，未验证：
1. CUDA 上的 inductor 代码生成与 CPU 不同。本次只测了 CPU，不能排除 GPU 上的特化问题。
2. g1e 的 A 组实际传入 Miles 的 `eps_clip_high` 不是 0.001（例如参数默认值或翻译路径把它改成了别的值）。
3. 记录的 pg_clipfrac 来自与 grad_norm 不同的微批或步（聚合路径差异）。

## 对后续项的影响
- 2.2：不改 fork，按否定结论勾选。
- 5.3：条件不成立，上卡时记为不适用。
- 5.2（clip_higher 在新 pin 上重跑）的判据里已有"clipfrac 与离线重算一致"。GPU 上的残余疑点由 5.2 检查：如果 5.2 中 clipfrac 与离线重算不一致，再回到本项查 CUDA 路径。

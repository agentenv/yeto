# RL 舰队看板（`yeto dashboard`）

只读看板：把 learner 磁带、syncer 磁带、controller journal 与 head 侧 `fleet.jsonl`
用同一个 reducer（`yeto/dashboard/reducer.py`）归约成视图，两种出口：

- **实时**：`yeto dashboard serve`，stdlib HTTP，只绑 127.0.0.1、只接受 GET，经 SSH 隧道访问；
- **离线**：`yeto dashboard export`，单个自包含 HTML（视图 JSON 内联，不加载字体/CDN，打开时无网络请求）。

界面按 mock v7：告警卡（严重/警告/提示，点击定位岛/round/指标）→ 按岛叠加训练曲线（12 个指标 tab）
→ syncer round 表（派生，可“仅异常”）→ 岛健康卡（点击展开同页岛明细：进度/policy、cell 表、
E1 事务、最近事件、Ray 嵌入）→ 成本效率表。页面没有任何改变运行状态的操作，只有“复制 CLI 命令”。

W&B 仍是曲线归档，不是看板的数据源。

## 实时：head 上起服务，笔记本经隧道看

```bash
# head 上（run 名即 --cluster-prefix）
yeto dashboard serve --run <run> [--tapes 额外磁带或目录...] [--port 8787] [--budget 300] [--head <head-host>]
# 打印：ssh -N -L 8787:127.0.0.1:8787 <head-host>  then open http://127.0.0.1:8787/

# 笔记本上
ssh -N -L 8787:127.0.0.1:8787 <head-host>
# 浏览器打开 http://127.0.0.1:8787/ ，页面每 5 s 轮询 /api/overview
```

`--run <run>` 自动跟随 `~/.yeto/runs/<run>/events/*.jsonl`（回显重组的 learner 磁带）、
`~/.yeto/runs/<run>/fleet.jsonl` 与 head 上的 syncer 磁带 `~/yeto-output/yeto-tape.jsonl`（存在时）。
`--tapes` 可追加文件或目录（目录向下两层收集 `*.jsonl`；同目录只有一个 `rl-island-N.jsonl` 时
`journal.jsonl` 归到岛 N）。服务每 `--poll` 秒（默认 2）按字节偏移增量读取，只消费完整行，
重复读同一偏移不重复计数；偏移只在内存里，与 W&B 转发的 `OffsetStore` 互不影响。

`--host` 只接受回环地址（127.0.0.1 / ::1 / localhost）；`--host 0.0.0.0` 会拒绝启动。
127.0.0.1 仍可被同机其他用户访问：本工具假定 head 单用户，不做鉴权。

端点（全部 GET；其它方法 405）：

| 端点 | 内容 |
|---|---|
| `/` | v7 页面（与导出同一份静态资源） |
| `/api/overview` | 运行、全局状态、告警、成本摘要、岛卡片、叠加曲线序列、round 标记 |
| `/api/islands/<id>` | 岛明细：卡片、指标序列、资源采样、cell 表、E1 事务、最近事件、`ray_embed` |
| `/api/rounds?only_bad=1` | syncer 派生 round 表 |
| `/api/fleet` | `fleet.jsonl` 生命周期与 `cost_tick`、价目表来源、成本视图 |
| `/api/events?island=&type=&after=&limit=` | 事件分页（`after` 为上一页 `cursor`） |

## 离线导出

```bash
yeto dashboard export --run <run> -o run.html
yeto dashboard export --tapes ~/s1-runs/<run>/pulled -o run.html --budget 1000
```

头部显示“离线导出 · 生成于 …”。离线页的“现在”取磁带里最新时间戳（不是打开时刻），所以已结束运行
的心跳年龄不会随时间膨胀；对同一组磁带，导出内联的 overview 与以离线时钟运行的 `/api/overview`
相同（`tests/test_dashboard_serve.py`）。实时服务用墙钟计算年龄。Ray iframe 在导出中只显示占位与隧道命令。

## ssh_harness 运行：远端磁带镜像

```bash
yeto dashboard mirror --target user@learner-host --remote ~/yeto-output/rl-island-0.jsonl \
    --local ~/dash/<run>/rl-island-0.jsonl --island 0 &
yeto dashboard serve --tapes ~/dash/<run>
```

镜像用 `ssh <target> tail -c +<已镜像字节+1> -F <remote>`，断线后指数退避（1 s 起，最大 30 s）重连，
从本地镜像末尾的字节继续，不丢不重；断线/恢复写入旁路文件 `<mirror>.source.jsonl`
（`dashboard_source_lost` / `dashboard_source_restored`），看板据此给该岛“磁带来源丢失”告警。
Modal 岛继续走 `TapeCollector` 回显重组，不需要镜像。

## 成本（估算，非账单）

成本 = 价目表单价（$/GPU·h）× GPU 数 × 墙钟（`island_ready` 到 `island_lost`/`island_stop`/现在，重启
后重新累计）。默认价目表 `yeto/dashboard/prices.example.json` 是**示例值，非账单**；用 `--prices my.json`
（格式同示例，键为 `cloud:GPU`，`*:GPU` 作通配）换成自己的。不在表里的岛显示“未定价”而不是 $0；
本地岛按 $0 计且不参与效率排名。`--budget` 给出预算上限后，累计达 60%/80% 产生警告/严重告警，
并显示预计触达时间。没有 `fleet.jsonl` 的运行（旧运行、未经 FleetController 的运行）成本面板显示“无数据”。

head 侧 `fleet.jsonl` 由 `FleetController` 写入（`runs/<run>/fleet.jsonl`）：`island_ready`（含首次与重启）、
`island_lost`、`island_stop`、每 5 分钟一条 `cost_tick`。价目表与预算可用环境变量
`YETO_DASHBOARD_PRICES` / `YETO_DASHBOARD_BUDGET_USD` 传给 launcher。

## 告警规则与阈值

| 规则 | 默认阈值 | 级别 |
|---|---|---|
| 心跳/最后事件超时 | >60 s 警告，>300 s 严重 | 0/1 |
| 连续 missed | ≥2 轮警告，≥5 轮严重 | 0/1 |
| quorum 时延 | 近 7 轮中位 ≥ 3× 基线（≥10 轮后） | 1 |
| grad_norm 尖峰 / NaN | > 5× 前 8 轮中位；NaN/Inf 严重 | 1/0 |
| clipfrac / KL | clip > 0.2 提示；KL > 0.1 警告 | 2/1 |
| 预算 | 60% / 80% | 1/0 |
| RECOVERY_REQUIRED（journal phase 或磁带 `rl_reconfiguration`） | — | 0 |
| ssh 镜像来源丢失 | — | 1 |

用 `--thresholds t.json` 覆盖（键见 `yeto/dashboard/alerts.py:DEFAULT_THRESHOLDS`，未知键报错）。

## 缺失数据

磁带里没有的字段一律显示“无数据”，不画 0。目前 ports 路径的 learner 磁带还没有 `rl_heartbeat`、
`rl_resource_sample`、`train_metrics{}`、reward 分位数、response length、截断比例等（见
`openspec/changes/yeto-fleet-dashboard/tasks.md` 的接口请求）；对应曲线 tab 与卡片字段显示“无数据”，
发射后无需改看板即可出现。tok/s 在没有 `tok_per_s` 时由 `action_tokens / (rollout_seconds + train_seconds)` 推导。
cell 表在没有 `rl_cell_snapshot` 时由 journal 的 `gpu_pool` 记录派生（标注来源）。

# 真机运行失败记录（2026-09-23）

本文记录 2026-09-23 在真实云上跑任务组 8 时，每一次启动失败的原因、责任归属和处理方式。机器是 Linux 开发机，SkyPilot 0.13.0，head 控制器多数放在 Nebius eu-north1，任务 8.7 的那次放在 AWS us-east-1。

## 处理方式说明

| 标记 | 含义 |
|---|---|
| **代码修复** | 改了 `yeto/` 下的源码，并补了能复现该问题的测试 |
| **手工处理** | 在本机或云控制台做了一次性操作，代码没改，下次还会遇到 |
| **未处理** | 已定位，但没改代码，也没绕过 |

所有代码修复目前都在工作区里，**尚未提交**。本文共记录 24 个问题，其中 14 个做了代码修复。

## 按运行排列

### 运行 1：yeto-nb81（Nebius，1 卡 H100）

head 起来了，learner 没起来。两个原因，任一个都足以失败。

1. **head 上缺 Nebius 项目配置。** 本 change 的 bug。
   - 现象：head 上的控制器报"eu-north1 没有 project_id"。
   - 原因：Nebius 一个项目绑定一个区域，项目 ID 写在 `~/.sky/config.yaml`。本机提交前的检查读得到这个文件，但 head 只挂载了 `~/.nebius`，没挂这个配置文件。
   - 处理：**代码修复**。fleet 含 Nebius 时，head 额外挂载 `~/.sky/config.yaml`（`yeto/launcher.py` 的 `head_cloud_credentials`）。测试：`test_head_gets_sky_config_when_fleet_uses_nebius`。
2. **syncer 一启动就退出。** main 上已有的 bug，不是本 change 引入的。
   - 现象：`--resume checkpoint does not exist or is not a file`。
   - 原因：2026-08-27 的提交 d4c72cd 让 syncer 在带 `--resume` 但检查点不存在时报错退出。而 SFT 和 RL 的 syncer 启动命令都无条件带 `--resume`，所以任何全新运行都会失败，和用哪朵云无关。
   - 处理：**代码修复**。改成由 shell 判断，检查点存在时才加 `--resume`（`yeto/launcher.py` 的 `_resume_if_exists`）。重启时命令会重新求值，所以续跑不受影响。测试：`test_syncer_resumes_only_when_a_checkpoint_exists`。

### 运行 2：yeto-nv1（Nebius + Verda，各 1 卡 H100）

syncer 正常启动，两台 learner 都没起来。

3. **head 上的 sky 没启用 Nebius。** 本 change 的 bug。
   - 现象：`Task requires nebius which is not enabled`。
   - 原因：head 只安装了 `skypilot[aws,gcp]`。任务 3.6 改了 `pyproject.toml` 的 launcher extra，但漏了 head 自己的安装命令。缺少 nebius 的 SDK 包，sky 就判定 Nebius 未启用。
   - 处理：**代码修复**。head 改装 `skypilot[aws,gcp,runpod,nebius,verda]`（`yeto/cli.py` 的 `HEAD_SETUP_PIP`），测试断言已同步更新。
4. **Verda 起不了机器。** 云侧和 sky 侧的限制。
   - 现象：`Failed to acquire resources in all zones in FIN-02`。
   - 原因：两件事叠在一起。其一，Verda 现在所有 location 都没有 1 卡 H100。其二，sky 的 Verda 目录只有 11 行，缺全部 8 卡机型和全部 H200，而 Verda 现在有货的只有 FIN-02 的 1 卡 H200 和 FIN-03 的 8 卡 H100。sky 目录里的 1 卡 H100 机型名是 `1H100.80S.32V`，Verda 在售的却是 `1H100.80S.30V`，可能已经过时。
   - 处理：**未处理**。规划器已经不用 sky 的 Verda 目录（设计 D2），但启动仍然走 sky。要让 Verda 可用，需要刷新 sky 的 Verda 目录并让 head 也使用它。这超出本 change 的设计，待决定。

### 运行 3：yeto-nb2（Nebius，1 卡 H100）

learner 起来了，也连上了 syncer，但一读数据就退出。

5. **模型的 chat template 不支持精确 loss mask。** 我选错了参数，不是代码 bug。
   - 现象：`ExactAssistantMaskError`，Qwen3-0.6B 的 chat template 里没有 `{% generation %}`。
   - 原因：Yeto 默认使用精确的 assistant loss mask，需要模板里有这个标记。
   - 处理：**手工处理**。启动时加 `--assistant-mask-mode legacy`。代码行为是对的，报错信息也写明了怎么办。

### 运行 4：yeto-md1（Modal，1 卡 H100）

head 没起来。

6. **Nebius 公网 IP 配额用满。** 云侧限制，由问题 7 触发。
   - 现象：`vpc.ipv4-address.public.count (limit 3, requested 4)`。
   - 原因：Nebius 租户默认只允许 3 个公网 IPv4。当时运行 3 留下的孤儿 H100 还占着一个。
   - 处理：**手工处理**。删除孤儿机器后重试。要同时开更多机器，需要向 Nebius 申请提高配额。
7. **`yeto down` 在 head 模式下漏删 learner。** 已有的 bug，不是本 change 引入的，但会造成真实花费。
   - 现象：回收运行 3 后，它的 H100 learner 仍在 Nebius 上运行，从 07:49 空跑到 08:03。
   - 原因：head 模式下，learner 由 head 上的 sky 创建，只登记在 head 那边。`yeto down` 在本机对 learner 调用 `sky down`，本机的 sky 不认识它，只报"集群不存在"，然后照样删掉 head，learner 就没人管了。正常结束的运行不受影响，因为 head 自己会回收 learner。只有中途失败后手动回收的运行才会泄漏。
   - 处理：**手工处理加未处理**。孤儿机器已用 Nebius CLI 手动删除，代码没改。之后每次回收，我都在 Nebius 上直接核对实例列表。

### 运行 5：yeto-md2（Modal，1 卡 H100）

head 和 syncer 都起来了，Modal 岛没起来。

8. **head 上没装 Modal SDK。** 本 change 的 bug，启动前就发现了。
   - 原因：Modal 岛由 head 上的控制器通过 Modal SDK 部署和启动，但 head 不装 `modal` 包。
   - 处理：**代码修复**。fleet 含 Modal 时，head 额外安装 `modal>=1.0`（`yeto/cli.py`）。测试：`test_head_installs_the_modal_sdk_when_the_fleet_has_a_modal_island`。
9. **Modal 岛被当成 sky 云。** 本 change 的 bug。
   - 现象：`Cloud 'modal' is not a valid cloud`。
   - 原因：为了复用运行脚本和环境变量，代码对 Modal 岛也调用了普通 learner 的任务构建函数，而这个函数会设置 `infra=modal` 的 sky 资源。单元测试把 sky 整个 mock 掉了，而且从来没有对 Modal 岛本身构建过任务，所以没发现。
   - 处理：**代码修复**。SFT 和 RL 的任务构建函数对 Modal 岛都跳过 sky 资源设置（`yeto/launcher.py`）。测试：`test_modal_island_task_never_asks_sky_for_modal_resources`，其中的假 sky 会像真 sky 一样拒绝 `modal`。
10. **错误清理代码掩盖了真实报错。** 本 change 的 bug。
    - 现象：`UnboundLocalError: local variable 'modal_ops'`。
    - 原因：失败路径要停止 Modal 应用，用到了 `modal_ops`，但这个变量要到构建完各岛之后才初始化。
    - 处理：**代码修复**。改为在构建任何岛之前初始化。

### 运行 6：yeto-md3（Modal，1 卡 H100）

路由问题已解决，Modal 开始构建镜像，部署时失败。

11. **Modal 镜像构建步骤顺序不对。** 本 change 的 bug。
    - 现象：`An image tried to run a build step after using image.add_local_* to include local files`。
    - 原因：Modal 规定，以非复制方式加入本地目录后，不能再有任何构建步骤。我们在加入代码目录之后又设置了环境变量。原有单元测试把这个错误顺序写成了断言。
    - 处理：**代码修复**。先设环境变量，加入代码目录作为最后一步（`yeto/modal_runner.py` 的 `build_image`），测试断言同步更正。

### 运行 7：yeto-md4（Modal，1 卡 H100）

训练成功，完成了 2 整轮同步，结果见 `docs/CLOUDS.md` 的 Run log。回收时发现一个问题。

12. **`yeto down` 声称已停止 Modal 应用，实际没停。** 本 change 的 bug。
    - 现象：`yeto down` 打印"Modal app yeto-yeto-md4: stopped"，但 `modal app list` 显示它仍是 deployed。任务数为 0，所以没有 GPU 在计费。
    - 原因：Modal 1.5.5 的 `modal app stop` 在非交互终端下会要求确认，拿不到就中止。代码设置了 `check=False` 并且不看输出，所以失败被吞掉，外层照样打印成功。
    - 处理：**代码修复**。加上 `--yes`，命令失败或输出中止时抛出异常，`yeto down` 会将其报告为"停止失败"（`yeto/modal_runner.py` 的 `stop_app`）。测试：`test_stop_app_confirms_and_surfaces_failures`。md4 的应用已用修好的代码停掉。

### 之后的运行：规划器、Verda 和 Modal 手动接入（yeto-vd1/vd2、yeto-mj1 至 mj4）

13. **Nebius 容量接口始终返回 HTTP 400。** 本 change 的 bug。
    - 原因：请求的 `pageSize` 是 1000，接口上限是 200。所有 Nebius 形状因此都退化为假定分数。
    - 处理：**代码修复**。page size 改为 200，并正确翻页（`yeto/shape/providers.py`）。测试：`test_nebius_advice_request_respects_the_page_size_limit_and_pages`。
14. **大机型售罄时，规划器把同型号的小机型也一起淘汰了。** main 上已有的规则，经本 change 的非 AWS 云暴露出来。
    - 现象：Verda 的 1 卡 H200 明明有货，规划器却给出"没有可行方案"。
    - 原因："被更大的同型号机器取代"这条规则在查库存之前执行。它原本是为 AWS 设计的，因为 AWS 按每天的配置数计费。
    - 处理：**代码修复**。非 AWS 云改为在有货的形状里判断取代关系，AWS 不变（`yeto/shape/plan.py`）。测试：`test_sold_out_fat_node_does_not_hide_an_in_stock_thin_one`。
15. **启动过程中改了源码，head 拒绝运行。** 我的操作失误。
    - 现象：`ProvenanceError: Yeto source SHA256 mismatch`。
    - 原因：Yeto 在启动时计算源码哈希，head 上再校验一次。我在 vd1 启动后修改了代码。
    - 处理：**手工处理**。重新启动。此后有运行在启动中时不再改源码。
16. **Verda 规划显示有货，申请时却没有。** 云侧限制。
    - 原因：Verda 的库存变化很快，而且库存信号只有"有或无"两种状态。
    - 处理：**未处理**。按用户要求，Verda 暂不使用。
17. **head 模式下的 syncer 不计外部座位。** main 上已有的 bug，2026-07-03 引入 head 模式时就存在。
    - 现象：手动接入的 learner 被拒，报"learner id 1 is outside configured range 0..1"。
    - 原因：head 上的 syncer 只按 `--gpu` 的岛数设置 learner 数。head 模式是默认模式，所以 `--external-learners` 在默认模式下从来没能用过。
    - 处理：**代码修复**（`yeto/cli.py` 的 `cmd_head`）。测试：`test_head_syncer_counts_external_learner_seats`。
18. **默认 quorum 为 1 时，两个岛抢同一步。** 这不是 bug，是参数语义。
    - 现象：Modal 岛只参与了第 1 到 4 步。
    - 处理：**手工处理**。5.3 改用 `--quorum 2` 重跑。
19. **Modal 镜像把 agent 的 worktree 也打包进去了。** 本 change 的 bug。
    - 现象：`...pyc was modified during build process`，构建中止。
    - 原因：上传仓库目录时没有排除 `.claude`，缓存目录的忽略规则也只匹配仓库根目录。
    - 处理：**代码修复**。排除 `.claude` 和各级缓存目录（`MODAL_WORKDIR_IGNORE`）。测试：`test_island_image_skips_agent_worktrees_and_nested_caches`。
20. **手动接入时源码哈希不一致。** 我的操作失误，同第 15 条。
    - 原因：运行脚本生成后，我又改了 `modal_runner.py`。
    - 处理：**手工处理**。用当前源码重新启动。
21. **`yeto down` 在 head 模式下漏删 learner，前两次修复都不对。** 对应第 7 条问题。
    - 第一次修复：在 head 上提交一个 sky 作业来删除 learner。作业没有输出，删除也没发生，代码却打印了成功。
    - 第二次修复：改为通过 SSH 在 head 上执行，并逐个核对输出。但 SSH 会话里的 `python3` 是系统 Python，没有 sky，这一点通过只读检查发现。
    - 第三次问题：head 端和控制器同时在删 learner，sky 返回 500。代码随后照样删掉了 head，learner 成为孤儿。
    - 处理：**代码修复**。现在优先使用 head 上的 miniconda Python；最多重试 3 次；只要有 learner 没得到确认，就保留 head 并返回非零退出码。测试：`test_down_head_run_retries_then_keeps_the_head_when_unconfirmed`、`test_down_head_run_succeeds_after_a_retry`、`test_head_down_requires_a_confirmation_per_learner`、`test_head_down_script_downs_each_learner`。**这个最终版本还没有在真机上经历一次真正需要它的回收。**
    - 本次会话一共手动删除了 4 台孤儿 H100，分别来自 nb2、mj1、mj3 和 mj4。

### 任务 8.7 混合 fleet（yeto-mix87、yeto-mix87b）

22. **本 AWS 账号开不出任何 L4 机器。** 云侧配额，不是代码问题。
    - 现象：`yeto-mix87` 的 `aws:1xl4@us-east-1` 岛在五个可用区全部报
      `VcpuLimitExceeded ... current vCPU limit of 0`。
    - 原因：us-east-1 的 "Running On-Demand G and VT instances" 配额是 0，
      G/VT 的 spot 配额也是 0，L4 所在的 g4/g5/g6 全系列都开不了。查了四个配额：
      G-and-VT 0、G-and-VT-spot 0、P 96、standard-OD 256。P 系列有配额，
      但 p3 是 V100（sm_70，不支持 bf16），p4d/p5 需要容量预留。
    - 处理：**手工处理**。改用 `nebius:1xh100@eu-north1,modal:1xh100` 两个学习者岛，
      AWS 留在 fleet 里做 head/syncer（m6i.2xlarge，纯 CPU 走 standard 配额）。
      三朵云仍在同一次启动里。要逐字复跑原文命令，得先申请 G-and-VT 配额。
    - 附带确认了一件好事：AWS 岛开不出来时，launcher 把已经部署好的 Modal 岛
      主动拆掉了，`yeto down` 之后三边零残留。失败路径是对的。

23. **`yeto down` 对已经停掉的 Modal 应用报"停止失败"。** 本 change 的小瑕疵，未处理。
    - 现象：正常跑完的运行里 head 已经停了 Modal 应用，随后 `yeto down` 打印
      `Modal app stop failed: ... App is already stopped.`，但继续正常回收，退出码 0。
    - 原因：第 12 条的修复让 `stop_app` 在命令失败时抛异常，但没有把
      "应用已经是 stopped" 当成成功。
    - 处理：**未处理**。每次成功的 Modal 或混合 fleet 运行都会打印这行误导性的噪音。
      当时另一个 session 的 `yeto-rl84f` 正在跑，改源码会影响它重启时的 provenance
      校验，所以没有动代码。

24. **sky 岛的训练日志不进 head job 日志，集群一拆就没了。** 已有的可观测性缺口，未处理。
    - 现象：8.7 那次运行保存下来的日志里，Modal 岛（learner 1）有 135 条
      `loss/token` 记录，Nebius 岛（learner 0）只有 4 行，全是 setup 阶段的，
      一条训练指标都没有。
    - 原因：两类岛的日志路径不对称。Modal 岛的 stdout 由 Modal SDK 直接回流到
      head job，整段都在；sky 岛的训练日志留在它自己的集群上，head 只转发了开头
      几行，要看得上集群跑 `sky logs`。回收之后就取不回来了。
    - 影响：不影响判断岛有没有在训练（tape 里每步的 responders、finalize 记录和
      job 状态都够），但拿不到两个岛的 loss 曲线做逐步对照——而 8.4 正是靠逐步
      对照才发现 gnorm 全零的。多云运行时这个缺口会更难受，因为跨云的数值差异
      正是最该盯的东西。
    - 处理：**未处理**。绕过办法是回收前先 `yeto logs <prefix>` 存一份，或者在
      head 的控制器里把 sky 岛的日志也转发进 job stdout。


> 以下三段来自 8.4 的独立 worktree，合并进主线时条目由 13-23 重编号为 25-35，
> 内部交叉引用已相应修正。

### 任务 8.4 的 RL 岛：yeto-rl84a 至 rl84e（Modal，8 卡 H100）

用户把 8.4 的环境从 CyberGym 改成 MATH-500 加数学答案奖励。这组运行在独立的 git worktree 里进行，前四次失败，第五次（rl84e）跑通。

25. **RL 岛的 setup 被当成镜像构建步骤。** 本 change 的 bug，启动前读代码发现。
    - 原因：`build_image` 用 `run_commands` 把 sky 任务的 `setup` 烘进镜像。但 setup 要读仓库里的 Miles bundle，而仓库目录只在容器启动时挂载，构建时读不到。
    - 处理：**代码修复**。setup 改为在容器里、run 脚本之前执行，和 sky 在节点上执行 setup 的方式一致（`yeto/modal_runner.py` 的 `island_main`）。测试：`test_rl_setup_runs_in_the_container_before_the_run_script`，并更正了 `test_rl_island_uses_the_digest_image`。
26. **默认 Miles 镜像拉不下来。** 环境限制。
    - 原因：`ghcr.io/agentenv/miles` 是私有仓库，本机没有 GitHub token。
    - 处理：**手工处理**。改用 Docker Hub 上公开的上游镜像 `radixark/miles:v0.1.0`，按摘要固定（`--rl-image docker:radixark/miles@sha256:cd40db92…eaa09`，2026-08-18 构建，与 MILES_BASE_COMMIT 同期）。Modal 镜像摘要仍等于 `--rl-image`，符合 spec；和默认镜像不同，属于偏离。Miles 和 SGLang 本身仍由 setup 装到固定提交。
27. **镜像自带的 Miles 仓库 origin 不对。** 已有 bug，换镜像后暴露（rl84a）。
    - 现象：`Miles origin mismatch: expected https://github.com/agentenv/miles, got https://github.com/radixark/miles.git`。
    - 原因：镜像里已有 `~/miles`，setup 只在目录不存在时才 clone，于是沿用了镜像的 origin，运行时校验拒绝。
    - 处理：**代码修复**。setup 在 fetch 前把 origin 设为固定仓库（`yeto/launcher.py` 的 `make_miles_island_task`）。测试：`test_miles_task_checks_out_exact_commit_and_builds_multinode_ray` 中新增的断言。
28. **Ray head 没开 dashboard，Miles 找不到 head 节点。** 已有 bug，sky 路径同样会触发（rl84b）。
    - 现象：`Failed to make request to http://127.0.0.1:8265/api/v0/nodes`。
    - 原因：learner 总是给 Miles 传 `--pin-rollout-manager-to-head`，Miles 通过 Ray 的 state API 列节点，而这个 API 由 dashboard 提供。launcher 却用 `--include-dashboard=false` 启动 Ray。SSH harness 路径用的是 `=true`，所以之前没暴露。
    - 处理：**代码修复**。改为 `--include-dashboard=true`。测试：同上一条测试中新增的断言。
29. **Miles 启动 SGLang router 超时。** Miles 的限制，在 Modal 上必现（rl84c、rl84d）。
    - 现象：`Server at 172.20.x.x:3xxx not ready after 30s`。
    - 原因：Miles 用 `spawn` 子进程启动 router，子进程要重新 import `miles.utils.http_utils`，它会连带导入 Megatron-Bridge。在 Modal 的 H100 容器上，热缓存时也要 26 到 28 秒，冷启动时 55 秒；而 Miles 写死只等 30 秒。独立启动 router CLI 只要 2 秒（用 Modal sandbox 实测）。先试过在容器里预热 import（rl84d），没用，已撤回。
    - 处理：**代码修复**。Modal RL 岛设置 `YETO_RL_EXTERNAL_ROUTER=1`，learner 自己启动独立 router，等待上限 300 秒，再把地址交给 Miles；Miles 看到 `sglang_router_ip` 已设置就不再自己启动（`yeto/rl/learner.py` 的 `start_external_sglang_router`，`yeto/launcher.py` 的 `build_modal_island_config`）。sky 岛行为不变。测试：`test_external_router_starts_and_hands_miles_its_address`，以及 `test_launch_auto.py` 中 Modal 配置测试新增的断言。
30. **`--apply-chat-template-kwargs` 没有传到 RL 岛。** 已有 bug。
    - 原因：CLI 接受这个参数，但 `make_miles_island_task` 不转发。Qwen3 需要 `{"enable_thinking": false}`，否则 1024 token 的回答几乎都截断在思考阶段。
    - 处理：**代码修复**。测试：`test_miles_island_forwards_chat_template_kwargs`。
31. **MATH-500 的格式 Yeto 读不了。** 数据集适配，不算 bug。
    - 原因：MATH-500 的列是 `problem` 和 `answer`，而且只有 `test` 一个 split；RL 数据准备只认 `messages/prompt/input`，并固定读 `train`。
    - 处理：**代码修复**。`problem`/`question` 可作为提示词，`answer` 可作为 label；Hub 数据集没有 `train` 且只有一个 split 时用那个 split（`yeto/rl/learner.py`）。新增奖励函数 `yeto/rl/math_reward.py:reward_func`，用 Miles 自带的数学判分。测试：`test_math_500_rows_become_miles_prompt_rows`、`test_prompt_data_uses_the_only_split_when_there_is_no_train`、`test_math_reward_grades_the_boxed_answer_after_thinking`。
32. **RL 的 head 模式运行被登记成错误的名字。** 已有 bug。
    - 现象：`[yeto] run 'CYBERGYM_REWARD_VIEW' submitted`。
    - 原因：`cmd_launch_head` 里转发 CyberGym 环境变量的循环复用了变量名 `name`，覆盖了运行名。于是运行登记表里的状态和 head job id 被写到了一个叫 `CYBERGYM_REWARD_VIEW` 的运行上。
    - 处理：**代码修复**。测试：`test_rl_launch_records_the_run_under_its_own_name`。

### 任务 8.4 重跑：yeto-rl84f（Modal，8 卡 H100）

33. **`gnorm="0.0000"` 是显示假象，但适配器确实没在学。** 上一轮据此没有勾选 8.4，判断依据不成立。
    - 现象：syncer 日志三步都打 `gnorm="0.0000"`。
    - 实情：tape 里的全精度值是 `4.161e-06`、`3.697e-06`、`1.103e-06`，并非零。日志那行用的是 `{gnorm:.4}`（`syncer/src/server.rs:3070`），把 4e-06 格式化成了 `0.0000`。**tape（同文件 3701 行）写的是原始值，排查时读 tape 不要读日志行。**
    - 但量级说明训练没有真正发生：3.67M 个 LoRA 参数、`lr-pg_0=6.67e-06`，一次 Adam 步的 delta 范数应约 `sqrt(3.67e6)*6.67e-6 = 0.013`，实测小约 3000 倍，属 fp32 往返噪声。Megatron 侧 `train/grad_norm` 三步精确为 `0.0`，两边一致。
    - **根因已查明，见下面第 35 条。** 排查中排除过：`grad_norm` 漏赋值（`multi_lora=False`、`debug_disable_optimizer=False`，走的是 `optimizer.step()` 真实返回值）；钩子时机（`after_local_train` 在 `actor_model.train()` 之后）；`apply_trainable_state` 破坏优化器引用（它是 `copy_()` 原地写入 `main_param`）；`bridge.py` 的 delta 算法；采样退化（温度 1.0、top_k -1，非贪心）；LoRA 参数 `requires_grad` 被关掉（实测 224/224 全为 True）。
    - **排查时不要被这几个数误导**：`rollout/advantages` 和 `train/pg_loss` 约等于 `2^-29`，但组内中心化让优势的全局均值恒为 0，GRPO 在 `ratio=1` 时 loss 值也天然≈0，**而梯度不该为 0**；`ess_ratio=1.0`、`pg_clipfrac=0.0` 同理。
    - **核心矛盾**：奖励是严格二值的（`yeto/rl/math_reward.py`），`rollout/raw_reward = 0.609375 = 39/64`。退化组只贡献 0 或 4，39 是奇数，**必然至少一组内奖励有差异**，优势不可能全为零。所以「优势退化」解释不了梯度为零。
    - 建议的下一步：单卡复现，直接断言 LoRA 的 A/B 参数 `requires_grad`、反向后 `.grad` 是否为 None、优化器步前后 B 矩阵是否变化。注意容器里的 Miles 是 setup 阶段从 GitHub 按固定 commit clone 的，改本机 `/home/michael/miles` 不影响容器。

34. **`--pipeline` 对 strict-avg 不适用，不是「不需要调大」。** 澄清而非 bug。
    - `strict-avg` 强制 `--fragments 1`，而 `--pipeline` 被钳制到 `--fragments`（`yeto/cli.py:586-591`），所以恒为 1，调它无效。
    - 实测同步阻塞约占整轮 30%（扣掉本来就要付的 SGLang 权重发布后约 27%），其中 syncer 自身合并只有 48–60 毫秒，其余是 14.7 MB delta 的跨公网往返和两端导出/应用/哈希。这个开销值得回收，手段是换 decoupled 预设并提高 `--fragments`。
    - **`sync_ms` 不是 WAN 耗时**，它等于 `merge_seconds`（`syncer/src/server.rs:3698`），只是合并计算。WAN 时间混在 `quorum_ms` 里，而后者被岛的 rollout+训练主导，要拆开只能用岛侧时间戳。

### 定位第 33 条的根因：gnz1（Modal，1 卡 H100）

35. **跨 dtype 的 `.data` 交换销毁了 Megatron DDP 的梯度累加 hook，LoRA 训练全程是空操作。** 本 change 集成路径的 bug，单卡复现后定案。
    - **现象**：`train/grad_norm` 恒为 `0.0`，适配器只在 fp32 噪声量级抖动（见第 33 条）。
    - **直接证据**（探针打在真实 `optimizer.step()` 前后）：224 个 LoRA 参数全部 `requires_grad=True`；`param.grad` **有值**，`lora_grad_l2 = 2.79e-02`；但 `main_grad` 全为零，`grad_added_to_main_grad=False`；优化器侧 `n_with_grad=0`、`step_returned=(True, 0.0, None)`、步后所有 `delta=+0.000000e+00`。
    - **结论**：反向传播是健康的，loss mask 不是全零，**优势也不是全零**——梯度确实产生了，只是从未进入 Megatron DDP 的 grad buffer。Megatron 的 `_copy_model_grads_to_main_grads` 读 `main_grad` 而非 `param.grad`，于是把零拷进 fp32 master，`grad_norm` 精确为 0，优化器步彻底空转。
    - **根因**：`miles/backends/megatron_utils/trainable_state.py:502`
      ```python
      parameter.data = parameter.main_param.view(parameter.shape)
      ```
      把 **fp32** 张量赋给 **bf16** Parameter 的 `.data`。PyTorch 的 `VariableHooks::set_data` 在 dtype 不一致时执行 `grad_accumulator_.reset()`，销毁 AccumulateGrad 节点；Megatron DDP 在 `__init__` 时注册在该节点上的 hook 随之失效，下次反向会新建一个没有 hook 的 accumulator。第 507 行的还原是同样的跨 dtype 赋值，救不回来。
    - **最小复现**（本机 CPU，torch 2.8.0，零成本）：
      ```
      no data swap        : hook_fired=1  main_grad=4.0
      bf16 param <-> fp32 : hook_fired=0  p.grad=4.0  main_grad=0.0
      bf16 param <-> bf16 : hook_fired=1  main_grad=4.0
      ```
      同 dtype 交换无害，**是跨 dtype 这一点触发的**。
    - **为什么每轮都坏**：`_optimizer_masters_as_model_parameters` 在 `_collective_adapter_tensors`（export）和 `apply_trainable_state` 两条路径都用，且都在第一次 train step 之前的 `MilesPolicySync._initialize` 中跑过，所以从第 0 步就坏、每轮重新坏一次。
    - **A/B 的附带现象**：`linear_in`（A）`grad=0.0`、`linear_out`（B）`grad≈2.4e-3`。B 初始为零时 A 本来就没梯度；因为 B 永远不动，A 也就永远拿不到梯度，适配器被钉死在初始点。
    - **归属**：代码住在 Miles 仓库，但 `trainable_state.py` 整个是 Yeto 的 external policy-sync 集成层（满篇 `yeto_rl_*` 参数），纯 Miles 的 LoRA 路径不经过它，所以 Miles 自己的 e2e 测试测不出来。**实质是本 change 这条集成路径的 bug。**
    - **修复障碍**：容器里的 Miles 是 setup 阶段从 GitHub 按固定 commit clone 的，**只改本机 `/home/michael/miles` 不生效**，修复必须推到 Miles 仓库并更新 pin，或走 `yeto/rl/vendor/` 的 bundle 补丁。
    - **探针注入通道**（以后还能用）：`RayTrainGroup._actor_handles` + `handle.__ray_call__.remote(fn)`。yeto 整个 worktree 通过 `modal.Image.add_local_dir` 挂进容器，所以 yeto 侧新增的模块能被 Ray 按模块名反序列化，绕开了「Miles 从 GitHub 固定 commit clone」这个障碍。
    - 诊断用改动（未提交，可删）：新增 `yeto/rl/_gradprobe.py`，`yeto/rl/miles.py` 加 `_install_grad_probe()` 及两处调用。
    - 花费：约 1.5 美元（单卡跑一次即定案，本机 CPU 复现闭合因果链）。

## 成功的运行

**yeto-md4（Modal，1 卡 H100，syncer 在 Nebius head 上）** 也跑通了：16 个外层步，2 整轮同步。每步梯度范数和 Nebius 那次逐步一致。冷启动后，跨公网同步每步 123 到 338 毫秒。它不对应任何一个任务：8.4 要求 8 卡 RL 岛，8.7 要求和 AWS 岛混跑，5.3 要求手动接入模式。

**yeto-nb3（Nebius，1 卡 H100）** 完成了任务 8.1：16 个外层步，也就是 2 整轮同步，最终检查点已写入，训练正常结束。head 在结束时自己回收了 learner。详情见 `docs/CLOUDS.md` 的 Run log。

## 为了跑起来做的环境配置

以下都是本机上的一次性操作，**代码没有改**。换一台机器需要重做，或者按 `docs/CLOUDS.md` 的说明由用户自己完成。

- **Nebius token 文件。** 用户建好了 nebius CLI 的服务账号 profile，但 SkyPilot 读的 `~/.nebius/NEBIUS_IAM_TOKEN.txt` 和 `NEBIUS_TENANT_ID.txt` 不存在。我从 profile 导出了这两个文件。服务账号的 whoami 里没有 tenant，所以 tenant ID 取自 profile 配置，而不是文档里写的 `iam whoami` 命令。**这说明 `docs/CLOUDS.md` 的 Nebius 命令只适用于用户账号，不适用于服务账号。**
- **sky 的 Nebius 项目配置。** 新建 `~/.sky/config.yaml`，写入 eu-north1 的项目 ID。
- **Verda `config.json`。** Verda CLI 写的是 `~/.verda/credentials`（INI 格式），SkyPilot 和 Yeto 读的是 `~/.verda/config.json`。我做了一次格式转换。
- **全量测试的 PATH。** 集成测试要用 cargo 编译 debug 版 syncer，而 cargo 不在默认 PATH 里，需要加上 `~/.cargo/bin`。

## 已知但未处理的问题

- **sky 的 Verda 目录太旧**（问题 4）。Verda 基本无法通过 sky 启动。
- **`yeto down` 在 head 模式下漏删 learner**（问题 7 和 21）。代码已修，但最终版本还没有在真机上验证过。
- **Modal 手动接入需要手工生成运行脚本。** 启动日志只打印 MLX 的接入命令，没有 Modal 的。
- **head 挂载了整个 `~/.nebius` 目录。** 其中包含 nebius CLI 的 `config.yaml`，里面有服务账号私钥。SkyPilot 只需要两个 txt 文件。这符合 spec 的写法，但挂载范围比必要的宽。
- **回收后的云端核验报错。** 报 `'<' not supported between instances of 'StatusVersion'`，可能是 sky 0.13 的兼容问题。它不影响回收本身，但也就没有了核验。
- **RL 集成测试同样撞上 `--resume` 问题。** `tests/test_rl_integration.py` 自己拼 syncer 命令，对全新的检查点路径无条件带 `--resume`。这是 main 的测试问题，没有改。

## 安全事项

- **Nebius 服务账号私钥泄露到了会话记录里。** 我读取 `~/.nebius/config.yaml` 时写错了过滤规则，私钥被完整打印出来。应当轮换这把密钥。
- **仓库根目录有未被 gitignore 的 `private.pem` 和 `public.pem`。** 不要提交它们。
- **`~/.modal.toml` 权限是 644。** 本机其他用户都能读，建议改成 600。

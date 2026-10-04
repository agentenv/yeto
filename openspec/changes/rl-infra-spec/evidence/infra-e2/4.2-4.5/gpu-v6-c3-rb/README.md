# G-4.4 正常重建（RESTORED）：C3-rebuild 首跑与补跑

- 首跑 app ap-CBXIKEsGz8049UxYgz1WXA，补跑 app ap-KLss8iQRDzDDM1EAwxGMdw，都已 stopped/0。
- 两次的磁带侧判据全部通过：只初始化一次，重建事件 1 次且 policy_version=3，重新发布给全体成员且 hash == cut，外层不重放，第 3 轮样本哈希与游标 == 基线 B1，优化器更新不重复。首跑还从事务日志拿到：阶段序列、结果 RESTORED、generation=1、local_step=3（`g44.json`）。
- 两次都没有拉回 cut manifest：首跑的状态包在 8 KiB exec 上限处被截断，补跑的分块整包拉取过慢。
- **cut manifest 字段的来源：C3-rebuild-old 运行的 manifest**（`../gpu-v6-c3-rbold/cut-manifest.truncated-8k.json`，其中 ledger、outer、progress 段完整：local_step=3、settled=true、carried_over=0、ready_unconsumed=0）。
  - 保存路径与正常重建相同：`rebuild_wiring.make_trainer_rebuilder` 在任何重建尝试之前调用 `trainer.save_cut`（`yeto/rl/engine/miles_adapter/rebuild_wiring.py:148` → `trainer.py:401`）。
  - 两条路径只在之后的 `rebuild_same_shape`（`trainer_rebuild.py:195`）中不同；REBUILD_OLD 的注入位于 fork 的 `create_training_models`（`trainer_rebuild.py:132-141`）。
  - 按主 agent 裁定，不再补跑。

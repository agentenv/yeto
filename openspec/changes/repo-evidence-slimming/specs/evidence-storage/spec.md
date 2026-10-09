## ADDED Requirements

### Requirement: evidence 目录只提交摘要与清单
openspec/changes/*/evidence/ 下 SHALL 只提交摘要、判读结果（小 json）、文档与归档清单。原始日志与原始数据（日志、jsonl、parquet、压缩包等）SHALL 上传到 Modal Volume `yeto-evidence-archive`，按仓库内原相对路径存放，并在该 change 的 `evidence/ARCHIVE-MANIFEST.tsv` 记录路径、字节数、sha256 与 Volume 位置。

#### Scenario: 新实验产生原始日志
- **WHEN** 开发者把实验原始日志放进 evidence 目录
- **THEN** `.gitignore` 使其不被跟踪，开发者上传 Volume 并在清单中登记，只提交摘要

#### Scenario: 测试夹具
- **WHEN** 某个 evidence 文件被 tests/ 或代码读取
- **THEN** 该文件可以留在仓库，并在 `.gitignore` 中显式放行

### Requirement: 提交前检查
仓库 SHALL 提供只依赖标准库的提交前检查脚本，拒绝单文件超过 500 KB（白名单除外）、疑似密钥和个人邮箱。

#### Scenario: 提交大文件
- **WHEN** 暂存区含有超过 500 KB 且不在白名单的文件
- **THEN** 检查失败并列出文件

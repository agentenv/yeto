## Why

仓库当前树 134 MB 中约 87% 是 openspec/changes/*/evidence/ 下的原始日志与数据（审计见 infra-drafts/REPO-HEALTH-AUDIT.md）。克隆和检出变慢，原始日志里还混有一个个人 gmail。需要把原始数据移出当前树，并防止以后再提交进来。

## What Changes

- 用户决定（2026-10-09）：
  1. 不改写历史。只从当前树删除，之后的提交里不再有这些文件。
  2. 外部归档放 Modal Volume `yeto-evidence-archive`（v1），按仓库内原相对路径存放。
  3. 删掉当前树里的个人 gmail，替换为 `<redacted-email>`。本机路径、用户名不在本次范围。
- 被 tests/、yeto/、scripts/、tools/ 代码读取的 evidence 文件留在仓库，不动。
- 只被 .md 文档引用且大于 100 KB 的文件、以及没有任何引用的文件：上传 Volume 并校验后，从当前树删除。
- 每个 change 的 evidence 目录写一份 `ARCHIVE-MANIFEST.tsv`，记录已归档文件。文档里的链接不改，靠清单查找。
- 防复发：`.gitignore` 忽略 evidence 下的原始日志类型；新增标准库提交前检查脚本 `scripts/precommit_check.py` 与 `.pre-commit-config.yaml`；不改 CI。

## Capabilities

### New Capabilities
- `evidence-storage`：evidence 目录只提交摘要、判读结果与清单，原始数据放 Modal Volume。

## Impact

- openspec/changes/*/evidence/ 大量文件删除（历史中仍可取回）。
- 新文件：.gitignore 规则、scripts/precommit_check.py、.pre-commit-config.yaml、docs/EVIDENCE_STORAGE.md。
- 正在写 evidence 的分支（如 s18-aru-stage3）合入后，原始日志会被忽略，需改为上传 Volume。

# evidence 存放约定与提交前检查

## evidence 只提交摘要

`openspec/changes/<change>/evidence/` 只提交：摘要（`*summary*`）、判读结果（小 json）、文档（`*.md`）、清单（`*manifest*`、`ARCHIVE-MANIFEST.tsv`），以及被测试读取的夹具。

原始日志与原始数据（`*.log`、`*.jsonl`、`*.parquet`、`*.gz`、`*.part` 等）由 `.gitignore` 忽略，放到 Modal Volume `yeto-evidence-archive`，路径与仓库内相对路径一致：

```bash
modal volume put yeto-evidence-archive openspec/changes/<change>/evidence/<file> openspec/changes/<change>/evidence/<file>
```

然后在 `openspec/changes/<change>/evidence/ARCHIVE-MANIFEST.tsv` 追加一行（制表符分隔）：

```
path	bytes	sha256	volume	volume_path
```

取回：`modal volume get yeto-evidence-archive <volume_path> <本地路径>`。2026-10-09 之前提交的原始文件也可以从 git 历史检出。

新增的测试夹具如果被 `.gitignore` 忽略，在 `.gitignore` 末尾用 `!路径` 显式放行。

## 提交前检查

`scripts/precommit_check.py` 只用标准库，检查暂存文件：

- 单文件大于 500 KB 拒绝。确需提交的文件写进 `.precommit-allow-large`（每行一个路径或通配）。
- 15 类简单密钥正则（AWS、GitHub、HF、W&B、Modal、OpenAI/Anthropic、私钥、岛 HMAC、URL 内凭据、Bearer、Nebius、Verda 等）。
- 个人邮箱（gmail、outlook、qq 等），请替换为 `<redacted-email>`。

启用方式（二选一）：

```bash
pre-commit install                      # 已装 pre-commit 时，读取 .pre-commit-config.yaml
printf '#!/bin/sh\nexec python3 scripts/precommit_check.py\n' > .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit
```

手动检查某些文件：`python3 scripts/precommit_check.py 文件...`。CI 不运行此检查。

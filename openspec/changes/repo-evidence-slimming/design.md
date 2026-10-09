## Context

审计按路径前缀匹配，分类偏宽。本次重新精确分类后再处理。

## Decisions

### 分类方法
对每个已跟踪的 evidence 文件：
- 静态：在 tests/、yeto/、scripts/、tools/ 的文本里查找完整路径、change 内相对路径、evidence 内相对路径、"父目录/文件名"、全仓唯一的文件名；以及 evidence 下子目录名以路径片段形式出现（视为整个目录可能被 glob/遍历读取）。
- 动态：运行所有提到 evidence 的本机安全测试，用 Python 审计钩子记录实际打开或列出的 evidence 路径。
- 任一命中即"被代码读取"，保留。
- 文档引用：在全部 .md 里查同样的路径键。

### 处理
- 被代码读取：保留，不裁剪。
- 只被文档引用且 ≤100 KB：保留。
- 只被文档引用且 >100 KB、或无引用：归档后删除。
- 删除前必须确认 Volume 中同路径文件大小一致。

### 归档位置与清单格式
- Volume：`yeto-evidence-archive`（Modal Volume v1），路径 = 仓库内原相对路径（如 `openspec/changes/<change>/evidence/...`）。
- 每个 change 的 evidence 目录下 `ARCHIVE-MANIFEST.tsv`，制表符分隔，表头：
  `path	bytes	sha256	volume	volume_path`
  `path` 为仓库内原相对路径。
- 取回：`modal volume get yeto-evidence-archive <volume_path> <本地路径>`，或从 git 历史检出。

### 防复发
- `.gitignore` 忽略 evidence 下 `*.log *.jsonl *.parquet *.gz *.part *.out *.err *.tar *.zst *.npy *.pt *.bin`，放行 `*summary*`、`*manifest*`、`*.md`、`ARCHIVE-MANIFEST.tsv`；仍被测试读取的现有文件逐条放行。
- `scripts/precommit_check.py`（仅标准库）：单文件 >500 KB 拒绝（白名单 `.precommit-allow-large`）、15 类密钥正则、个人邮箱检查（允许 noreply 与示例域名）。
- 不改 CI（PM 决定不管 CI）。

## Risks

- 静态匹配漏掉动态拼接路径 → 用测试动态追踪补齐；被删文件仍可从 git 历史与 Volume 取回。
- Volume 存储费用：数据约 100 MB 量级，费用很小，已在 gpu-spend.md 预登记估算。

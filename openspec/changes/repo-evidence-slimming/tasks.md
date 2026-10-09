## 1. 分类
- [x] 1.1 静态 + 动态精确分类，产出保留/归档清单

## 2. 归档
- [x] 2.1 gpu-spend.md 预登记 Volume 存储费用
- [x] 2.2 上传到 Modal Volume `yeto-evidence-archive`，逐文件比对大小，抽查 sha256 ≥50 个
- [x] 2.3 写各 change 的 ARCHIVE-MANIFEST.tsv，从当前树删除已校验文件

## 3. 个人邮箱
- [x] 3.1 当前树个人 gmail 替换为 `<redacted-email>`

## 4. 防复发
- [x] 4.1 .gitignore 规则，git ls-files 前后对比被测试读取文件仍被跟踪
- [x] 4.2 scripts/precommit_check.py、.pre-commit-config.yaml、docs/EVIDENCE_STORAGE.md

## 5. 验证
- [x] 5.1 删除前后跑读取 evidence 的本机安全测试，结果不变差
- [x] 5.2 记录工作树大小前后对比

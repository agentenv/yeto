# F-E1 第四次（A10G，代码 37155d8，镜像 2cc5cc52… / Miles e3a11ab3）：通过（冒烟，不计入 task）

- 指纹运行 fe1r5fp：`sha256:38169f69…`；manifest 镜像内 miles e3a11ab3、launcher 拉取 digest 2cc5cc52… 已核对。
- F-E1 fe1r5（3×A10G，T1R1S1↔T1R2S0；第 1 轮 train 时 up、第 3 轮 train 时 down，容器内触发器提交）。journal（`fe1r5/elastic-state/reconfig/journal.jsonl`）：
  - up1：start `engine:inference-engine-all-0-0-00001` done → VERIFYING → weight_admission → COMMITTED（成员 00000、00001）→ **SUCCEEDED**；
  - down1：QUIESCING → stop 00001 done → COMMITTED（成员 00000）→ **SUCCEEDED**；
  - epochs：config_epoch 2，members `engine:inference-engine-all-0-0-00000`。
- 判据（§9.9）：up、down 都 SUCCEEDED ✔；成员均为 fork cell id ✔；旧成员 00000 全程在役不变 ✔ → **通过**。
- 费用：指纹 ≤$0.5（09:10:07–09:23:06），正式 ≤$0.69（09:23:43–09:36:11，3×A10G×$1.10）。另 037d4f5 上作废的指纹运行 ≤$0.49。app 均 stopped/0。

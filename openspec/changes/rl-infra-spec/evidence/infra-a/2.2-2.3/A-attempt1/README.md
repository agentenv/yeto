失败在 Modal deploy 之前（本机用户线程 3863/4096，RuntimeError: can't start new thread），没有创建 app，也没有占用 GPU。修复：arm.sh 启动前检查线程数，低于 3200 才启动。

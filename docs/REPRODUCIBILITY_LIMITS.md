# 复现边界

本仓库提供三层证据：保存结果的 CPU 复算、原权重的重评估入口、从头训练协议。第一层通过，不自动证明后两层已完成。模型权重因体积不入 Git；`provenance/checkpoints.json` 登记历史核验的 SHA256，新的运行须重验实际下载文件。

原始实验源文件包含历史服务器绝对路径；`scripts/prepare_runtime.py` 写入隔离目录并记录改写前后 SHA。新提交稿图 3 组合与图 4 交换新增逐样本预测和 GPU 源码；图 3b/c 各条件分母 3,200，图 4d 为 4,116。两个 map fit 不等于两个独立 backbone。Ouro 的 frozen checkpoint 断言不可在新权重上直接忽略。跨硬件与不同浮点实现可能改变边界样本预测。

附录的逐层读出只使用冻结 final head，是读出诊断；不能把逐层最大类直接解释为内部语义步骤。Dense J 的 8-head 子集失败结论限定于 seeds 3/5/7、2,067 例、最终一个 F 调用、无 residual attenuation 的 pattern-only 替换；保存记录也含 attenuation 条件，它们不能并入这项失败结论。

仓库还保存旧 v66 范围的 PCA、KG 等研究材料，但它们不是当前提交稿证据；旧说明仅在 `docs/legacy_v66/`。此次没有从头跑完整 GPU 训练，不能保证不同初始化产生相同论文数值。

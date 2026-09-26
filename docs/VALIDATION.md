# 2026-09-26 提交稿验证

- 以提交 zip 中的 `main.tex` 为范围，锁定 21 页 PDF 和 15 个实际引用图形；见 `provenance/submission_manifest.json`。工作区中更晚的 reviewer revision 不在本次版本内。
- `python reproduce.py` 成功生成全部 15 个同名 PDF；图 1 为原作者概念图，14 个数据图由保存数值重绘。非论文中间图也可出现在输出目录。
- `python scripts/audit_submission.py` 从五个 backbone×两个 fit 的逐样本 `npz` 复算两次组合；按同一 3,200 例群体核对附录表格。目标交换从 4,116 例预测重算两个方向和五个 backbone，匹配保存汇总与正文 45.4%／60.9%。
- 同一核算从逐层预测确认 seeds 3/5/7 的主要循环末读出为 0、2、4、6、8；从 2,067 例的原始预测枚举每个 fit、每个 query scope 的 255 个非空 head 子集，最大 one-hop pattern-transfer 准确率均为 0，而 dense J 完整控制均超过 99.8%。这只验证保存的预测，不是新 GPU 重放。
- 旧版已完成的 `scripts/audit_results.py`、`scripts/audit_parity_phase.py` 继续覆盖图机制、Ouro、N8 长程、Parity 相位和监督对照；运行报告写入 `outputs/`。
- 尚未在这次改版中完整重训全部 GPU backbones/controllers，也未宣称新提交稿的所有 GPU 前向在另一台机器逐例重放；权重重评估需原 SHA 对应文件和合适 GPU。

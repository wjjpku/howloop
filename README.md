# PaperExperiment — ICLR 2027 提交稿复现

对应 2026-09-26 的匿名提交稿 **One Set of Weights, Many Algorithms: How Looped Transformers Route Computation Across Loops**。版本以 [`paper/main.tex`](paper/main.tex)、[`paper/submission.pdf`](paper/submission.pdf) 和 [`provenance/submission_manifest.json`](provenance/submission_manifest.json) 固定：21 页、15 个引用图形。

## 一键复算保存结果

Python 3.12，CPU 即可：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-plot.txt
python scripts/verify_integrity.py
python reproduce.py
python scripts/audit_submission.py
python scripts/audit_results.py
python scripts/audit_parity_phase.py
```

重绘图在 `outputs/figures/`，逐样本统计在 `outputs/audit/`，Parity 相位核算在 `outputs/parity_phase/`。图 1 是原稿概念图；其他 14 个数据图由保存的数值重新绘制。逐图数据与绘图入口见 [`provenance/figures.json`](provenance/figures.json)。

## 提交稿实验

| 部分 | 复现内容 | 位置 |
|---|---|---|
| 图 2 | N10 D8L8/D8L6 原生轨迹 | [`experiments/n10`](experiments/n10) |
| 图 3a | 冻结 F，从同一 h6 控制一步／两步目标 | [`experiments/n10`](experiments/n10) |
| 图 3b,c | 两次调用 F、连续应用 J_one/J_two，五个 backbone × 两个 fit | [`experiments/composition`](experiments/composition) |
| 图 4b,c | 跨图 pattern/output patch、steering pattern transfer、Ouro 对照 | [`experiments/graph_mechanism`](experiments/graph_mechanism)、[`experiments/ouro_letter`](experiments/ouro_letter) |
| 图 4d | one-hop／two-hop target pattern exchange，4,116 例 | [`experiments/target_exchange`](experiments/target_exchange) |
| 图 5、6 | Parity 长度与时序；另一个 N8 图 continuation 群体 | [`experiments/parity_phase`](experiments/parity_phase)、[`experiments/long_range`](experiments/long_range) |
| 图 7 | Ouro final-only／stepwise 监督与四循环 steering | [`experiments/ouro_antonym`](experiments/ouro_antonym) |

细节、分母和训练入口见 [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)、[`docs/RUNNING.md`](docs/RUNNING.md)。图 3 的 4,110／3,200 和图 4d 的 4,116 是三个不同评估群体。旧版 PCA、KG 和 S1/S2 图不在这份提交稿内；旧资料仍保存，但不计入本版的通过条件。

保存结果的核算与全部图形重绘已通过；全套 GPU 训练**未重新跑完**。原始权重不入仓库，身份与获取办法见 [`provenance/checkpoints.json`](provenance/checkpoints.json) 和 [`docs/RUNNING.md`](docs/RUNNING.md)。验证记录和具体限制见 [`docs/VALIDATION.md`](docs/VALIDATION.md)、[`docs/REPRODUCIBILITY_LIMITS.md`](docs/REPRODUCIBILITY_LIMITS.md)。

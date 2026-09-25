# PaperExperiment — v66 复现仓库

对应论文 **One Set of Weights, Many Algorithms: How Looped Transformers Route Computation Across Loops**，冻结版本见 [`paper/main.tex`](paper/main.tex)。本仓库覆盖正文和附录中的全部实验类别、17 个论文图形文件、原始结果及复现入口。

## 最快复现：从保存结果重绘与核算

Python 3.12，CPU 即可；不需要服务器或模型权重。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-plot.txt
python scripts/verify_integrity.py
python reproduce.py
python scripts/audit_results.py
python scripts/audit_parity_phase.py
```

- `outputs/figures/`：与 `paper/figures/` 同名的 **17 个论文图形文件**，另有中间组合图。
- `outputs/audit/`：从逐样本记录重新汇总的统计、计数、配对检查与置信区间。
- `outputs/parity_phase/`：三组 parity 相位拟合及 moving-block bootstrap。
- 图 1 是作者提供的概念图，保留原始矢量 PDF；它不属于实验数据图。其余图形均由脚本从保存的数值输入重建。图 4 的示意图也有绘制代码。

## 实验定位

| 论文位置 | 实验 | 源码、结果与入口 |
|---|---|---|
| §3，图 2 | N10 D8L8 / D8L6 原生轨迹，12 / 5 个模型 | [`experiments/n10`](experiments/n10)，`n10-backbone` / `n10-native` |
| §4，图 3a，附录 A | 同一个 h6 的一步、两步目标控制 | [`experiments/n10/code`](experiments/n10/code)，`n10-controller` / `n10-evaluate` |
| 图 3b，附录 A | token-mean PCA；按图划分 fit/display | [`experiments/pca`](experiments/pca)，`pca` |
| §5，图 4，图 S1，附录 A | N10 语义 patch、block/restore、pattern exchange；五模型 screening 与 A/C 新确认 | [`experiments/graph_mechanism`](experiments/graph_mechanism)，`graph-confirmation` |
| §5，图 4，附录 B | Ouro letter-walk、固定 16 heads 的路由与恢复 | [`experiments/ouro_letter`](experiments/ouro_letter)，`ouro-semantic` / `ouro-restore` |
| §6.1，图 5a、6、S2，附录 C | Parity 长度外推、热图与相位拟合 | [`experiments/parity_phase`](experiments/parity_phase)，`parity-backbone` / `parity-controller` / `parity-diagnostics` / `parity-long` |
| §6.1，图 5b，附录 D | **N8** D8L8，12 backbones × 2 J，1–40 次循环 | [`experiments/long_range`](experiments/long_range)，`n8-backbone` / `n8-controller` / `graph-long` |
| §6.2，图 7a–c，附录 E | Ouro 反义词，final-only / stepwise，固定四循环 | [`experiments/ouro_antonym`](experiments/ouro_antonym)，`ouro-final-controller` / `ouro-stepwise-controller` |
| §6.3，图 7d，附录 F | KG 四种控制器，长度 4→32 curriculum | [`experiments/kg`](experiments/kg)，`kg` |

逐图输入及脚本见 [`provenance/figures.json`](provenance/figures.json)。逐实验的配置和边界见 [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)。论文中的筛选条件、分母和种子数量不能跨实验混用。

## 使用权重重新评估 / 从头训练

请先读 [`docs/RUNNING.md`](docs/RUNNING.md)。实验环境与绘图环境分开：历史 GPU 运行环境完整清单在 [`provenance/a100_environment_freeze.txt`](provenance/a100_environment_freeze.txt)，核心依赖在 `requirements-experiments.txt`。

```bash
# 把历史源码中的机器路径转换为当前机器的独立工作目录；不会启动训练。
python scripts/prepare_runtime.py --work /your/data/paper-v66 --mode replay

# 先查看大小；下载需要原始私密服务器的 SSH 访问权。
python scripts/fetch_checkpoints.py --work /your/data/paper-v66 --group n10
python scripts/fetch_checkpoints.py --work /your/data/paper-v66 --group n10 --download

# 默认仅打印命令，检查后加 --gpu 0 --execute 才运行。
python scripts/run_experiment.py n10-evaluate --work /your/data/paper-v66 --seed 6 --hop 1 --fit 1
```

模型权重不进入 Git；`provenance/checkpoints.json` 记录实际核验的路径、大小和 SHA256，下载入口逐个验哈希。小型固定数据划分、Ouro tokenizer/模型实现、通用语料的已采样 token 数组和绘图输入已收录。

## 验证范围

本次整理实际完成了全部数据图重绘、关键逐样本结果重新汇总、三组相位拟合与 bootstrap、主 D8L6 模型和控制器的 CPU 前向/梯度冒烟检查。具体结果见 [`docs/VALIDATION.md`](docs/VALIDATION.md)。**没有把全套 GPU 训练重新跑一遍，不能据此保证从头训练逐位得到论文数值。**

完整训练的历史限制保留在 [`docs/REPRODUCIBILITY_LIMITS.md`](docs/REPRODUCIBILITY_LIMITS.md)，尤其是 KG 原始控制器 RNG 未记录、Ouro checkpoint 绑定，以及跨硬件浮点差异。保存结果重绘、原权重重评估、从头训练是三个不同层级。

`experiments/` 与 `vendor/` 保留历史代码和证据，历史绝对路径只用于溯源；使用 `scripts/prepare_runtime.py` 生成可运行副本，避免直接运行带旧机器路径的历史脚本。新运行不会修改仓库内的论文或原始结果。

# 运行指南

## 1. CPU：从保存结果复算提交稿

在仓库根目录使用 Python 3.12：

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

`outputs/figures/` 包含与提交稿同名的 15 个 PDF；图 1 是概念矢量图，其余 14 个由数值数据和绘图脚本生成。图 3 的数值直接取两个 fit 的逐样本预测；图 4 的新增 target-exchange 数值取存档预测重算。`outputs/audit/submission.json` 保存新表格审计。

`paper/` 是原样提交的 PDF 和独立 LaTeX 工程。若要试编译，在其副本中运行 `latexmk -pdf main.tex`；要替换重绘图，也只替换副本里的同名文件。重绘不承诺字体、PDF 字节和排版与提交包完全一致。

## 2. 原权重重评估

需要原权重，按 `provenance/checkpoints.json` 的 SHA256 核验；仓库不含模型权重、SSH 凭据。Linux GPU 环境参照 `requirements-experiments.txt` 和 `provenance/a100_environment_freeze.txt`。先在**空目录**生成隔离运行副本：

```bash
python scripts/prepare_runtime.py --work /your/data/paper-submission --mode replay
python scripts/fetch_checkpoints.py --work /your/data/paper-submission --group n10
python scripts/fetch_checkpoints.py --work /your/data/paper-submission --group n10 --download
```

下载命令仅在加 `--download` 时读取授权服务器。以下命令默认只打印，不启动 GPU；实际运行时显式加 `--gpu 0 --execute`，设备编号按机器情况选择：

```bash
python scripts/run_experiment.py n10-evaluate --work /your/data/paper-submission --seed 6 --hop 1 --fit 1
python scripts/run_experiment.py target-exchange --work /your/data/paper-submission
python scripts/run_experiment.py composition --work /your/data/paper-submission
python scripts/run_experiment.py graph-confirmation --work /your/data/paper-submission --seed 6
python scripts/run_experiment.py graph-long --work /your/data/paper-submission
python scripts/run_experiment.py parity-diagnostics --work /your/data/paper-submission --seed 2
python scripts/run_experiment.py ouro-semantic --work /your/data/paper-submission
```

`target-exchange` 在 `paper_strengthening_20260925/graph` 产出模型预测；`composition` 依赖这些文件中的 first-step 对照，因此顺序不能颠倒。两项都对 A–E×两个 fit 运行，并保留采样锁、checkpoint SHA 和计算设备记录。完整表格需再运行各实验目录的分析脚本。其他各实验的训练和评估入口见 `scripts/run_experiment.py`；旧版 `pca` 和 `kg` 命令仍在，但**不属于提交稿复现流程**。

## 3. 从头训练

在新的空目录执行 `prepare_runtime.py --mode train`，然后按实验清单依次训练 backbone、冻结它、训练 J、运行评估。每个新重复须保存 seed、数据排除集、模型 SHA 和选择规则；新权重不是原始 checkpoint。N10 L6 五 seeds、L8 十二 seeds，N8 十二 seeds×两 J，Parity 三 seeds，以及 Ouro 两类 backbone／J 是不同训练族；不要把 smoke test 或命令生成当作完整重训。

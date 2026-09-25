# 运行说明

## 三个复现层级

1. **保存结果 → 统计与图表**：只需仓库和CPU，README四条命令覆盖所有数据图、关键统计和相位拟合。
2. **原始权重 → 重新评估**：需要下载对应模型权重；沿用原始数据锁，结果应在声明的数值精度内吻合。CPU完整重评估范围见VALIDATION。
3. **从头训练 → 统计重复试验**：使用原训练协议，不假设重新训练能逐位产生同一权重。需要充足GPU时间；KG等历史run的限制见REPRODUCIBILITY_LIMITS。

## 独立工作目录和环境

使用Linux、Python3.10、CUDA环境运行历史实验最接近原配置。核心版本在`requirements-experiments.txt`，完整实测环境在`provenance/a100_environment_freeze.txt`。PyTorch CUDA wheel选择应适配主机驱动；不要在绘图venv里混装另一版numpy。

```bash
python scripts/prepare_runtime.py --work /your/data/paper-v66 --mode replay
```

- `--mode replay` 会提供冻结评估所需的原backbone说明。
- `--mode train` 用于新的空目录；不复制历史完成标志或运行目录，避免新训练被旧summary跳过。
- 原源码保存在仓库中。runtime副本替换`/data/wujiaju`根路径和解释器路径；letter-walk trainer额外加入checkpoint停止参数，以保留原500步学习率计划、在第200步保存后停止。改写前后SHA写入runtime_manifest。
- 配置文件里历史PID、GPU、status只是历史记录，不表示当前有实验运行。

## 取得权重

```bash
python scripts/fetch_checkpoints.py --work /your/data/paper-v66 --group n10
python scripts/fetch_checkpoints.py --work /your/data/paper-v66 --group n10 --download
```

group可选n10/parity/graph/ouro/kg/all。默认仅列出数量、大小与路径；`--download`才通过SSH读取。`--host`可指定另一台保存相同绝对路径的授权归档服务器。已有文件须哈希匹配，下载写入`.partial`后验证，再原子改名，不覆盖不同内容的文件。

`ouro`包括原始Ouro模型权重及微调模型，较大；code/config/tokenizer已在仓库。仅绘图和统计重算不需要下载权重。没有该私密服务器权限的人仍可复现保存结果，或按训练协议另行训练；仓库不包含服务器凭据。

## 原权重重新评估

以下命令默认打印工作目录与准确命令；确认设备有足够显存后，加`--gpu 0 --execute`。GPU编号由使用者显式指定，不自动占用多卡。

```bash
python scripts/run_experiment.py n10-native --work /your/data/paper-v66 --seed 6
python scripts/run_experiment.py n10-evaluate --work /your/data/paper-v66 --seed 6 --hop 1 --fit 1
python scripts/run_experiment.py graph-confirmation --work /your/data/paper-v66 --seed 6
python scripts/run_experiment.py graph-confirmation --work /your/data/paper-v66 --seed 4
python scripts/run_experiment.py pca --work /your/data/paper-v66
python scripts/run_experiment.py graph-long --work /your/data/paper-v66
python scripts/run_experiment.py parity-long --work /your/data/paper-v66
python scripts/run_experiment.py parity-diagnostics --work /your/data/paper-v66 --seed 2
python scripts/run_experiment.py ouro-semantic --work /your/data/paper-v66
python scripts/run_experiment.py ouro-restore --work /your/data/paper-v66
```

主控制实验需运行hop1/2×fit1/2；完整backbone重复则对seeds6/3/4/5/7各运行。`graph-confirmation`有意锁定原checkpoint/hash和A/C选择，适用于原权重重评估。若换成新训练权重，必须重新执行discovery head selection与图排除检查并建立新的selection lock，不能静默复用论文head和原hash。

N10完整五模型controller+discovery+confirmation流程在runtime的`n10_migration_20260923/code/worker_local.py`：`--shard 0`和`--shard 1`分别覆盖注册模型，串行执行也可以。该脚本使用当前backbone动态计算哈希并先在discovery选head。它不是A/C独立新确认脚本，两个群体分别报告。

## 从头训练

在新的空目录prepare `--mode train`。N10数据锁已经包含，不要再次执行make_locks覆盖它们。

```bash
python scripts/run_experiment.py n10-backbone --work /your/data/new-run --seed 6 --loops 6
python scripts/run_experiment.py n10-controller --work /your/data/new-run --seed 6 --hop 1 --fit 1
# 完成hop1/2 × fit1/2后导出统一的控制器包
python scripts/run_experiment.py n10-export --work /your/data/new-run --seed 6

python scripts/run_experiment.py n8-backbone --work /your/data/new-run --seed 100
python scripts/run_experiment.py n8-controller --work /your/data/new-run --seed 100 --fit 1

python scripts/run_experiment.py parity-backbone --work /your/data/new-run --seed 2
python scripts/run_experiment.py parity-controller --work /your/data/new-run --seed 2

python scripts/run_experiment.py ouro-letter-backbone --work /your/data/new-run
python scripts/run_experiment.py ouro-letter-controller --work /your/data/new-run
python scripts/run_experiment.py ouro-final-backbone --work /your/data/new-run
python scripts/run_experiment.py ouro-stepwise-backbone --work /your/data/new-run
python scripts/run_experiment.py ouro-final-controller --work /your/data/new-run
python scripts/run_experiment.py ouro-stepwise-controller --work /your/data/new-run

python scripts/run_experiment.py kg --work /your/data/new-run --controller lora_r48 --seed 0
```

同样需加`--gpu 0 --execute`才执行。完整范围：N10 L6五seeds、L8十二seeds；N8 seeds100–111×两J；parity seeds0–2；KG四种controller。KG入口需原backbone，Ouro从头SFT需原预训练Ouro权重；不意味着重新预训练2.6B基础模型。

Ouro历史J脚本有冻结backbone SHA断言。要复现原J拟合，使用已核验的原backbone；对重新训练产生的新权重进行统计重复试验时，应先建立新checkpoint身份并修改新runtime中的断言，记录修改，不能称为原权重复现。letter-walk原backbone保存于update200，但原学习率按500步计划衰减。启动器使用`--steps 500 --stop-after 200`，在保存checkpoint200后停止，保留原训练的验证early-stop逻辑。直接改成`--steps 200`会改变学习率，不能作为相同协议。

KG历史控制器种子未记录。新入口强制传递显式seed（命令建议0），保存每阶段权重；论文表格由已存原始结果精确重算。KG原backbone生成逻辑见`vendor/remote/kg-fj-affine-resffn-m64-code-5f49178/experiments/kg_fj_length/`，该锚点优先用哈希锁定权重，未声称已经端到端重训验证。

训练命令和日志写入工作目录`logs/`。完整训练耗时取决于设备；本仓库没有提供未经测量的总耗时承诺。Ouro FP32主权重/优化器通常需要大显存，历史运行使用A100-80GB。

## CPU主模型检查

```bash
python scripts/smoke_graph.py --checkpoint /path/to/best.pt --controller /path/to/best_controller.pt
```

它检查配置、20个样本的批处理/逐例预测、有限输出、冻结backbone不收梯度、controller收到有限非零梯度，不把smoke结果当全测试集准确率。

## 重建论文

`paper/`本身是自包含LaTeX工程。安装TeX Live后可在该目录运行`latexmk -pdf main.tex`。若要替换为重绘图，请先复制`paper/`到输出目录，再仅复制17个同名PDF进入副本的`figures/`；不要修改冻结的原稿作为验证方式。本次任务验证图表和实验代码，未额外改写论文。

# 本次整理的实际验证（2026-09-25）

## 已执行

| 检查 | 结果 | 证据 |
|---|---|---|
| v66论文身份 | 冻结源文件与当前v66 SHA256一致 | `provenance/paper_version.json` |
| 绘图 | 新建Python3.12 venv、按固定依赖安装，全部图形入口成功；17个论文PDF全部生成 | `provenance/validation/clean_environment_figures.json` |
| 图形外观 | 与原图并排渲染检查；轨迹、PCA、机制、长程、监督与补充图布局和数值吻合 | 原稿`paper/figures/`；复现输出`outputs/figures/` |
| D8L6机制 | A/C新确认与五模型历史screening，从events重新汇总；bootstrap和原summary一致 | `provenance/validation/saved_results_audit.json` |
| Ouro机制 | semantic64与restore64逐样本重新汇总，计数/置信区间/配对差值一致 | 同上 |
| 长程曲线 | Graph 12×3×40×8192计数、Parity 101长度的逐例布尔结果核算一致 | 同上 |
| Ouro监督 | 四条件512个`(k,sequence,expected)`严格配对；step0/500计数一致 | 同上 |
| KG容量 | m32最终评估行与图7d全部16个格一致 | 同上 |
| Parity相位 | 三backbones×两条件，从diagonal-band数据重做Fourier phase、unwrap及5000次bootstrap，拟合与区间一致 | `provenance/validation/parity_phase.json` |
| 原权重CPU冒烟 | 20例，有限输出、逐例/批次预测一致、backbone无梯度、J非零有限梯度 | `provenance/validation/graph_cpu_smoke.json` |
| 原权重CPU完整评估 | N10 A、one-hop、fit1；512图×10起点；raw/full/shuffle×pre/post，3072条graph-cluster记录与历史记录逐条完全相同 | `provenance/validation/full_cpu_replay_comparison.json` |
| 隔离runtime | 在空目录生成运行副本；23个阶段命令构造成功 | `provenance/validation/runtime_commands.json` |
| 代码完整性 | 2227个Python文件语法解析通过；未检出所检查的常见token/private-key模式 | `provenance/validation/static_checks.json` |
| 权重身份 | 92个文件逐个读取核验SHA256；不是只复制旧manifest | `provenance/checkpoints.json` |

本机初次venv安装遇到Python系统证书链配置问题，使用已有certifi CA文件后安装成功，没有禁用TLS校验。该机器问题不写入通用requirements。

## 未执行

- 没有完整重训全部backbones与controllers。
- 没有在此次整理中运行全套GPU机制评估、全部Ouro生成或KG训练。
- 23个阶段是命令构造检查，不等价于23个完整实验运行成功。
- 本次CPU完整权重评估只覆盖A/one-hop/fit1；其他拟合的结果重算来自历史逐样本文件。

GitHub Actions配置会在push时运行CPU完整性、重绘、结果汇总和相位重算。是否通过以仓库实际run状态为准，不以存在工作流文件代替检查结果。

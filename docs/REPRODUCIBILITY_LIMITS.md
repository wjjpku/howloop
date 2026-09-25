# 可复现性边界与历史缺口

## 已具备独立复现条件

- 保存的数值输入、逐样本结果、bootstrap和绘图代码均在仓库；CPU可重绘全部数据图及重新汇总主要表格。固定selection/confirmation锁与原始checkpoint身份可校验。
- 17个论文图形文件有逐项mapping。图1是人工提供的概念图；不把复制它称为实验重算。图4示意图由绘图代码重建。字体在不同系统可能有替换，因此不承诺PDF字节完全相同；数值和数据划分有独立检查。
- 92个模型/控制器/预训练权重文件实际读取并核验SHA256，清单位于`provenance/checkpoints.json`；文件本身不进入Git。原权重重新评估需授权SSH访问或另行取得相同SHA文件。

## 从头训练的限制

1. **未重新运行全套GPU训练。** 本次任务的验证是保存结果重算、干净绘图环境、运行入口检查、主N10原权重CPU冒烟及一组完整评估。其余GPU入口未执行的事实逐项保留。
2. **KG原始控制器RNG和阶段权重缺失。** 四family的历史results未记录controller seed；原run未保存controller checkpoints。可精确重算论文中保存的表格，但无法保证重新训练复刻那一次随机实现。新入口显式设置seed并保存每阶段权重，属于新的重复试验。
3. **Ouro机制代码绑定原backbone/J。** 冻结评估保留SHA断言；新backbone训练成功不意味着哈希应相同。新checkpoint上的实验需要新的身份、日志与选择记录，不能删除断言后仍称原结果重现。
4. **历史Ouro训练不能补造图排除保证。** 保存了评估pairs、训练生成代码和通用语料token缓存；原backbone没有可验证的完整graph-exclusion清单。只报告论文已有的评估群体关系。
5. **数值精度依赖后端。** 同一checkpoint在GPU联合branch评估和单branch评估中曾有3个graph-call计数差异，来源为接近决策边界的浮点差异，已保留`experiments/long_range/results/overlap_recheck.json`。不要替换成更高分结果来“修正”复现。
6. **原机器路径只作溯源。** 历史manifest和源快照保留原路径；实际运行使用prepare_runtime的隔离副本。数据图入口没有个人路径依赖。没有把“只有机器路径的代码集合”当成完全可移植入口。
7. **预训练来源与分发权限。** 此仓库维持私密；没有授予未明确提供的研究代码或语料新许可。第三方Ouro模型实现沿用其原始归属，见NOTICE。若未来公开仓库，需要单独决定权重/语料的分发方式。

## 协议不能合并的地方

- N10 D8L6机制与N8 D8L8 continuation分开。
- 原生轨迹、图3a的4,110例、PCA的5,120例、机制配对的各自eligibility分母分开。
- graph训练排除512图与主8192 broad-distribution曲线分开。
- parity旧64例/长度与新128例/长度分开，缺测长度不插值。
- 两个J fits不是两个独立backbone。
- Ouro final-only/stepwise比较同时改变了训练prompt；J训练见过请求深度5–8。
- KG长度8/16/24/32已包含在curriculum内，不是未见长度外推。

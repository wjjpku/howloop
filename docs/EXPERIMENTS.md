# ICLR 2027 提交稿实验清单（2026-09-26）

范围由 `paper/main.tex` 和 `provenance/submission_manifest.json` 固定。A–E 对应 D8L6 seed 6/3/4/5/7；图形清单见 `provenance/figures.json`。数值输入、筛选集和模型身份不能跨行混用。

| 正文与附录 | 核心实验 | 证据与代码 |
|---|---|---|
| 图 2、Graph Traversal | N10 D8L8/D8L6 原生读出轨迹，final-only backbone | `experiments/n10`；`plots/fig2.py` |
| 图 3a | 同一 h6 施加 rank-48 `D+AB+b`，分别预测一步、两步目标；4,110 个互异标签样本 | `experiments/n10/figure_work/source`；`plots/fig3a.py` |
| 图 3b,c、Graph Traversal 附录 | 冻结 J_one/J_two 两次组合；512 图×10 起点，经 0–4 hop collision exclusion 后每条件 3,200 例；A–E×2 fits | `experiments/composition/{run.py,analyze.py,*_fit*.npz,summary.json}`；`plots/fig3_submission.py` |
| 图 4b,c、Graph Traversal/Ouro Letter-Walk | 跨图 pattern/output patch；同输入 steered↔unsteered pattern transfer；head-output restoration | `experiments/graph_mechanism`、`experiments/ouro_letter`；`plots/fig4_submission.py` |
| 图 4d、Graph Traversal 附录 | J_two→J_one 与 J_one→J_two：L1、L2、both 的 post-softmax pattern 交换；4,116 例，五 backbones×2 fits | `experiments/target_exchange/graph/*_fit*.npz`、`graph_summary.json`、`graph_matrix.py` |
| 图 5a、图 6、Parity 附录 | n1–20 backbone、n20–40 J；seed2 长度和读出时序，另有 seeds0/1 验证 | `experiments/parity_phase`、`data/plot/parity`、`plots/fig5.py`、`plots/fig6.py` |
| 图 5b、Graph Continuation 附录 | 单独的 N8 D8L8，12 backbones×2 J，1–40 loops | `experiments/long_range`、`plots/fig5.py` |
| 图 7、Ouro Antonym Cancellation 附录 | stepwise/final-only 的原生读出与四循环 k5–8 控制；各一个 rank128 J | `experiments/ouro_antonym`、`data/plot/ouro_fixed4.json`、`plots/fig7_submission.py` |

图 3b/c 与图 4d 是本次相对旧 v66 仓库的新增必备证据。两次组合的目标以固定的 `u=f_G^8(s)` 定义，不以第一次模型输出重定锚点；同一 hidden-state 序列直接进入第二个 J 和 F。4,110／3,200／4,116 分别属于目标选择、两次组合、目标交换三个群体。图 4 的 pattern patch 保留 receiver 的 values；output patch 复制 source 的 head output。图 5b 的 N8 cohort 不能与图 4 的 N10 D8L6 机制结果合并。

旧稿的 PCA、KG 容量和补充图 S1/S2 不属于这份 21 页提交稿。历史数据和脚本仍保留在仓库，旧说明备份于 `docs/legacy_v66/`，但不计入当前提交稿复现通过条件。

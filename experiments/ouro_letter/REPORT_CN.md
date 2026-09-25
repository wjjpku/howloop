# Ouro letter-walk：三组机制确认实验

## 结论

在一个固定的、经过 letter-walk 任务训练的 Ouro-2.6B checkpoint 和一个固定 affine J 上，三组独立 64 对样本支持一条完整但有边界的机制论证：attention pattern 干预与 head output 干预产生不同的语义结果；L34.H5 是有效的局部恢复位置；后 16 层的 patterns 能传递大部分 J 带来的准确率增益。这里的输出是 z=αV（o_proj 之前），不是单独的 V。没有新增模型训练，也没有把这些结果当成独立 backbone 复现。

![机制证据](ouro_mechanism_evidence.png)

**统一图注。** 三组实验使用不同的 64 对样本，主指标为全词表首答案 token argmax；误差线为 95% Wilson 区间。a：在共享前七条路径边、但第八个接收人不同的 source/base 图对上，移植 source patterns 主要产生 base 图反事实答案，移植 source outputs 主要产生 source 答案。均在第 2–4 次 call 的共享层 32–47、全部 heads 上操作。b：在另一组 64 对样本上，用同图不同起点运行的 Q 替换 L34 的 Q，再逐个恢复干净运行的 head output。分母为原本正确而被打错的 55 例；H0 是其余 15 个 heads 中恢复最多的一个。c：同一输入的 with-J/raw 双向 pattern patch；rescue 将 with-J patterns 放进 raw，damage 将 raw patterns 放进 with-J。层、call 范围与 a 相同。层和 head 编号从 0 开始。a 的完整姓名反事实匹配为 53/64，首 token 匹配为 52/64，差异来自一次合法的 Emma 分词路径。

## 1. Pattern 改变读取路径，output 传递路径相关内容

设计重点不是随意换两张图。source 与 base 都是十人单环；从 source 起点出发，前七条边相同，第八条边不同；base 提问使用另一个起点。排除 base 答案、source 答案、base 图反事实答案之间的碰撞。这样 source 路径作用于 base 内容会给出不同于 source 自己答案的结果。另跑 base 图配 source 起点的自然问题，检查反事实本身可回答。

| 第 2–4 次 call，层 32–47 | Base 答案 | Source 答案 | Base 图反事实 | 其他 |
|---|---:|---:|---:|---:|
| Source pattern → base | 0 | 2 | **52** | 10 |
| Source output → base | 0 | **61** | 2 | 1 |
| Source V → base | 1 | 43 | 5 | 15 |
| 错配 call 的 source pattern | 5 | 0 | 1 | 58 |

每行 n=64。干净 base 正确 63/64，source 正确 61/64，自然反事实正确 62/64。Output patch 的预测首 token 与 source 实际预测 **64/64 一致**，包括 source 的三个错误；这与正确 source 答案率 61/64 是不同指标。

Pattern 相对 output 的反事实匹配高 78.13 个百分点，配对 bootstrap 95% 区间 [67.19,87.50]；output 相对 pattern 的 source 答案匹配高 92.19 个百分点，区间 [84.38,98.44]。这支持两种干预作用不同，不能仅用“复制答案”解释 pattern 的结果。

**位置边界：**早层 0–15 pattern patch 的反事实匹配为 61/64，中层 16–31 为 13/64，局部层 33–34 为 57/64。局部层 33–34 的 output patch 也多产生反事实（50/64），而非 source 答案（6/64）。因此不能泛化成“所有层的 head output 都装着最终答案”，也不能说 routing 只发生在后层。这里的强交叉结果指预先选定的后 16 层组合。

## 2. L34.H5 是有用的局部位置

另取 64 对样本。在 base 图上换提问起点得到另一次运行，将其 L34 的全部 Q 放入原运行，保留接收运行的 K,V；随后恢复原运行一个 head 的 z=αV。操作覆盖 calls 2–4 的 prompt prefill；没有替换 residual stream。

| 条件 | 全部 64 例正确数 | 55 个被打错样本中修回数 |
|---|---:|---:|
| 干净运行 | 63 | — |
| Q 替换后 | 8 | 0 |
| 恢复 H5 | **62** | **54/55** |
| 恢复相邻 H4 | 8 | 0/55 |
| 恢复其余最佳 H0 | 11 | 3/55 |
| 恢复全部 heads | 63 | 55/55 |

H5 恢复率 98.18%，Wilson 95% 区间 [90.39,99.68]。相对其余最佳 head 高 92.73 个百分点；每次 bootstrap 重新选择最佳其他 head 的区间为 [85.45,98.18]。全 head 恢复逐例精确复现干净 logits（最大误差 0）。

H5 在此前研究中已选定，没有用这批确认样本挑选位置。这个结果证明：在此 Q 干预下，H5 是强恢复位置。它不证明 H5 是全模型最优 head，也不证明 H5 单独承担完整的语义区分或 J 效果。恢复调用了干净运行激活，因此也不是模型自主纠错。

## 3. Routing 能传递 J 的行为效果

用第三组 64 对样本的 base 输入，分别运行 raw 和 with-J。将 with-J 的 patterns 移入 raw，接收运行仍使用自身 V；反向将 raw patterns 放入 with-J。均对 calls 2–4、层 32–47 的全部 heads 操作。

| 条件 | 正确数 /64 | 准确率 |
|---|---:|---:|
| Raw | 0 | 0.00% |
| With J | 62 | 96.88% |
| Pattern rescue | **60** | **93.75%** |
| Output rescue | 60 | 93.75% |
| V rescue | 15 | 23.44% |
| Pattern damage | **0** | **0.00%** |
| 无关图的 with-J patterns | 0 | 0.00% |
| 错配 call 的 with-J patterns | 0 | 0.00% |
| 早层 0–15 pattern rescue | 16 | 25.00% |
| 中层 16–31 pattern rescue | 1 | 1.56% |
| 局部层 33–34 pattern rescue | 11 | 17.19% |

Pattern rescue 恢复了 (60−0)/(62−0)=**96.77% 的净准确率增益**；这是聚合准确率比值，不是逐例恢复率。Pattern rescue 的 Wilson 95% 区间 [85.00,97.54]。相对 V rescue 的配对提升为 70.31 个百分点，bootstrap 区间 [57.81,81.29]。所记录的主要条件完整姓名准确率均与首 token 指标一致。

这比仅观察 attention 改变或仅做破坏更强：正向移植带来效果，反向替换消除效果，错图和错 call 都不奏效。但这是跨 16 层、三次 call 的组合干预，不能据此认定唯一中介、自然计算的全部路径，或 J 只改变了 attention。被移植 patterns 来自完整 with-J 运行，也可能承载上游累计影响。

## 4. 保留的负面结果与论文措辞

- 任意独立图对的首次筛查：H5 的 10 层 × 3 call 扫描（8 对），以及层 33、34 的全 16-head 单 call 扫描，未找到所需交叉语义结果。应保留这些负面结果；新的正面证据依赖更清楚的共享路径构造。
- 共享路径上的 6 层全-head 联合 calls 2–4 探索中，单 H5 只在少数样本出现目标变化。不能把群体结果归因于一个 H5。
- 十节点环上的八步任务不能证明内部真的逐步执行八次操作，也不能排除反向两步等策略。四个 recurrent calls 不等于四个任务 hops。
- 三组是同一任务训练 backbone/J 的独立样本确认。没有证明其他 checkpoint、模板、长程长度泛化或未经任务训练的原始 Ouro 也有同样机制。
- 样本图在本次探索/确认集合之间严格不重合；历史训练样本来源无法完整复原，不声明与训练集严格图身份去重。

建议 Section 5 的三段逻辑可以迁移到 Ouro：**先分清路径与内容，再找有效局部位置，最后验证 J 的效果如何传递。** 写作上分别指明“后层组合”和“单个 H5”，避免把三个结论误写成一个 head 包办一切。当前结果已足以支持一个有实质机制内容的 Ouro 扩展；尚未改动论文版本。

## 5. 验证与复现

- `AUDIT.json`：三个 completed manifest；13/13/20 个条件、每条件 64 行，无遗漏重复；代码、数据、协议 SHA256 与运行时一致；backbone/J 相同且参数版本未变。
- 全部 1408 次记录的完整生成均与 forward 首 token 一致，无 16-token 截断。语义实验的一次 Emma 完整姓名与规范首 token 差异单独保留。
- Self-output patch 和 all-head restore 最大 logit 误差 0。Self-pattern 重算因 BF16 SDPA 与显式计算的差别，最大误差 0.125；baseline 准确率保持一致，不能声称位级一致。
- `audit.py` 另验证五个探索/确认数据文件共 416 张图互不重合、均为单环、答案碰撞排除、非姓名 token 一致与七边共享构造。
- `semantic_confirmation64/`、`mediation_confirmation64/`、`restore_confirmation64/` 保存逐例结果、manifest、统计；各固定协议位于根目录。`plot_evidence.py` 直接读取分析 JSON，输出 PDF/PNG 与 `figure_data.csv`。
- 统计单位是配对任务样本；不是 head 次数，也不是独立模型。主 bootstrap 10000 次；补充 Wilson、配对精确检验和保守区间在 analysis.json 中。
- 本轮远程进程 301077、475482 已结束，运行状态已由远程 manifest 和进程查询共同确认。

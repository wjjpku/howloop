# Ouro：将 J 的 pattern 中介范围缩小到最后一次 loop 的少量 heads

## 结果

固定原 backbone 与 J，在新生成且未参与筛选的64对图上，直接比较完整干预、10-head主候选及预先固定的8/16-head敏感性对照。探索使用8对旧样本；确认阶段不再选位置。

| 干预集合 | Shared heads | Calls | Pattern rescue | Pattern damage 后准确率 |
|---|---:|---|---:|---:|
| 原完整后16层 | 256 | 2–4 | 62/64 (96.88%) | 0/64 (0.00%) |
| 预设16-head对照 | 16 | 4 | 61/64 (95.31%) | 0/64 (0.00%) |
| 主候选10heads | 10 | 4 | 50/64 (78.12%) | 5/64 (7.81%) |
| 预设8-head对照 | 8 | 4 | 44/64 (68.75%) | 12/64 (18.75%) |
| 10个相邻对照heads | 10 | 4 | 0/64 (0.00%) | 62/64 (96.88%) |
| 同三层剩余heads | 38 | 4 | 0/64 (0.00%) | 62/64 (96.88%) |
| L47.H4单head | 1 | 4 | 0/64 (0.00%) | 63/64 (98.44%) |

原始无J：0/64 (0.00%)；完整J：62/64 (96.88%)。所有指标为全词表首答案token argmax。Rescue越高越好，damage后准确率越低表示破坏越强。

![定位结果](compact_heads.png)

**统一图注。** a：预设16-head对照的所在位置，只干预call4；蓝色为选中位置，灰色为同层其他heads。b、c：各集合的双向pattern干预。Full使用层32–47全部256heads、calls2–4；其余小集合仅call4。虚线是完整J准确率，误差线为95% Wilson区间。Neighbor与主候选匹配层和head数量但不重合。

## 范围与选择过程

主候选10heads：L41: H10, H12；L43: H6, H11, H13, H15；L47: H0, H3, H4, H13。

预设16heads：L41: H2, H10, H12, H15；L43: H1, H6, H11, H13, H15；L47: H0, H2, H3, H4, H7, H13, H15。层、head编号从0开始；call4是第4次、即最后一次循环。

原完整干预涉及256个共享heads、跨3次call共768个(layer,head,call)位置；10-head候选仅10个位置，16-head对照仅16个位置。共享head数量分别下降96.09%和93.75%；不能把768误写成768个独立参数heads。

筛选路径：后16层 → 最后8层 → 层41/43/47 → 四head分组与单head删减 → 嵌套小集合 → call子集。探索时10heads首次在测试的嵌套序列中达到8/8rescue与0/8damage；8heads为6/8rescue。只用call4也达到8/8与0/8，calls2–3为0/8与8/8。随后固定10-head-call4为主候选，8/16heads为预设敏感性对照，再打开新64样本。详细负面候选均保存在SEARCH_LOG.md及逐例结果中。

曾有两个9-head的全三call子集也通过探索，但其单call组合未在探索验证，未用作主候选。因此不宣称10或16是最小电路，不宣称全模型最优。

## 确认实验与解释

10heads单call rescue的Wilson区间：[66.57%, 86.50%]；damage后准确率区间：[3.38%, 17.02%]。16heads对应：[87.10%, 98.39%]、[0.00%, 5.66%]。

selected 相对原完整干预的净恢复增益保留 80.65%，破坏效应保留 91.94%。未同时达到两方向90%的预期保留水平；不能把探索集8/8当作充分确认。这些是点估计比例，不是统计等效证明。

size16 相对原完整干预的净恢复增益保留 98.39%，破坏效应保留 100.00%。两方向点估计均达到90%。这些是点估计比例，不是统计等效证明。


以下对照针对主候选10heads：

| 条件 | 准确率 |
|---|---:|
| 三个calls均移植有J patterns | 51/64 (79.69%) |
| 三个calls均移植无J patterns | 5/64 (7.81%) |
| 仅calls2–3 rescue | 0/64 (0.00%) |
| 仅calls2–3 damage | 63/64 (98.44%) |
| 另一张图的有J patterns | 0/64 (0.00%) |
| 把source call2 patterns放到接收call4 | 0/64 (0.00%) |

主要结论是：在这个任务训练checkpoint上，J的行为效果可以通过最后一次call里跨三层的少量attention patterns大幅传递或抵消；候选集变小时存在恢复稳定性代价。单个head没有复现集合的双向作用，不能简化成一个router head。

**证据范围。** 这些heads的位置偏晚，可能作用于答案相关信息的读取或整合；本实验不能单独证明逐步图遍历，也不能推出早期calls无关。用with-J运行保存的patterns做移植，可能带来它之前累计计算的信息。此前route/content交叉语义证据来自后16层群体，不自动转移到这个小集合。这里也不重新解释H5：H5的Q-corruption恢复实验与本次J/raw pattern中介实验衡量不同功能。

## 验证与文件

确认代码、配置、数据、协议哈希与运行记录一致；冻结配置和确认协议的服务器时间早于运行开始，见confirmation_freeze.json。新64对的128张图与既有416张图不重合，均为十节点单环，token槽对齐。无正确性筛选。

- native：首token正确 0/64，完整姓名正确 0/64，截断 0。
- J：首token正确 62/64，完整姓名正确 62/64，截断 0。
- full_rescue_pattern：首token正确 62/64，完整姓名正确 62/64，截断 0。
- full_damage_pattern：首token正确 0/64，完整姓名正确 0/64，截断 0。
- selected_rescue_pattern：首token正确 50/64，完整姓名正确 50/64，截断 0。
- selected_damage_pattern：首token正确 5/64，完整姓名正确 5/64，截断 0。

Self-output必须精确复现logits；self-pattern数值误差和预测一致性在AUDIT.json与analysis.json中记录。代码保留逐head运算，smoke的预测及概率与原脚本精确一致。配对bootstrap与精确检验保存在localize_confirmation64/analysis.json；不把大量候选干预次数算作独立样本数。

所有模型权重冻结；只跑推理干预，没有新增训练。原始逐例数据在localize_confirmation64/results.jsonl，固定配置在confirmation.json，完整搜索轨迹在SEARCH_LOG.md；图由plot.py读取统计结果直接绘制。论文版本未修改。

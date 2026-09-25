import json
from pathlib import Path
P=Path(__file__).resolve().parent;D=json.loads((P/'localize_confirmation64/analysis.json').read_text());s=D['conditions'];cfg=json.loads((P/'confirmation.json').read_text());rank=json.loads((P/'compact.json').read_text())['ranking']
rate=lambda k:f"{s[k]['correct']}/64 ({s[k]['accuracy']:.2%})"
ci=lambda k:'['+', '.join(f'{x:.2%}' for x in s[k]['wilson95'])+']'
heads=lambda n:'；'.join('L'+str(l)+': '+', '.join('H'+str(h) for _,ll,h in sorted(rank[:n],key=lambda x:(x[1],x[2])) if ll==l) for l in [41,43,47])
t=['# Ouro：将 J 的 pattern 中介范围缩小到最后一次 loop 的少量 heads\n',
'## 结果\n',
'固定原 backbone 与 J，在新生成且未参与筛选的64对图上，直接比较完整干预、10-head主候选及预先固定的8/16-head敏感性对照。探索使用8对旧样本；确认阶段不再选位置。\n',
'| 干预集合 | Shared heads | Calls | Pattern rescue | Pattern damage 后准确率 |\n|---|---:|---|---:|---:|']
for name,label,n,calls in [('full','原完整后16层',256,'2–4'),('size16','预设16-head对照',16,'4'),('selected','主候选10heads',10,'4'),('size8','预设8-head对照',8,'4'),('neighbor','10个相邻对照heads',10,'4'),('complement','同三层剩余heads',38,'4'),('single','L47.H4单head',1,'4')]:t.append(f"| {label} | {n} | {calls} | {rate(name+'_rescue_pattern')} | {rate(name+'_damage_pattern')} |")
t += ['\n原始无J：'+rate('native')+'；完整J：'+rate('J')+'。所有指标为全词表首答案token argmax。Rescue越高越好，damage后准确率越低表示破坏越强。\n',
'![定位结果](compact_heads.png)\n',
'**统一图注。** a：预设16-head对照的所在位置，只干预call4；蓝色为选中位置，灰色为同层其他heads。b、c：各集合的双向pattern干预。Full使用层32–47全部256heads、calls2–4；其余小集合仅call4。虚线是完整J准确率，误差线为95% Wilson区间。Neighbor与主候选匹配层和head数量但不重合。\n',
'## 范围与选择过程\n',
'主候选10heads：'+heads(10)+'。\n',
'预设16heads：'+heads(16)+'。层、head编号从0开始；call4是第4次、即最后一次循环。\n',
'原完整干预涉及256个共享heads、跨3次call共768个(layer,head,call)位置；10-head候选仅10个位置，16-head对照仅16个位置。共享head数量分别下降96.09%和93.75%；不能把768误写成768个独立参数heads。\n',
'筛选路径：后16层 → 最后8层 → 层41/43/47 → 四head分组与单head删减 → 嵌套小集合 → call子集。探索时10heads首次在测试的嵌套序列中达到8/8rescue与0/8damage；8heads为6/8rescue。只用call4也达到8/8与0/8，calls2–3为0/8与8/8。随后固定10-head-call4为主候选，8/16heads为预设敏感性对照，再打开新64样本。详细负面候选均保存在SEARCH_LOG.md及逐例结果中。\n',
'曾有两个9-head的全三call子集也通过探索，但其单call组合未在探索验证，未用作主候选。因此不宣称10或16是最小电路，不宣称全模型最优。\n',
'## 确认实验与解释\n',
'10heads单call rescue的Wilson区间：'+ci('selected_rescue_pattern')+'；damage后准确率区间：'+ci('selected_damage_pattern')+'。16heads对应：'+ci('size16_rescue_pattern')+'、'+ci('size16_damage_pattern')+'。\n']
for name in ['selected','size16']:
 rescue=(s[name+'_rescue_pattern']['accuracy']-s['native']['accuracy'])/(s['full_rescue_pattern']['accuracy']-s['native']['accuracy']);damage=(s['J']['accuracy']-s[name+'_damage_pattern']['accuracy'])/(s['J']['accuracy']-s['full_damage_pattern']['accuracy'])
 t.append(f"{name} 相对原完整干预的净恢复增益保留 {rescue:.2%}，破坏效应保留 {damage:.2%}。{'两方向点估计均达到90%。' if min(rescue,damage)>=.9 else '未同时达到两方向90%的预期保留水平；不能把探索集8/8当作充分确认。'}这些是点估计比例，不是统计等效证明。\n")
t+=['\n以下对照针对主候选10heads：\n','| 条件 | 准确率 |\n|---|---:|']
for k,lab in [('selected_allcalls_rescue_pattern','三个calls均移植有J patterns'),('selected_allcalls_damage_pattern','三个calls均移植无J patterns'),('selected_earlycalls_rescue_pattern','仅calls2–3 rescue'),('selected_earlycalls_damage_pattern','仅calls2–3 damage'),('selected_unrelated_pattern','另一张图的有J patterns'),('selected_wrong_call_pattern','把source call2 patterns放到接收call4')]:t.append(f'| {lab} | {rate(k)} |')
t += ['\n主要结论是：在这个任务训练checkpoint上，J的行为效果可以通过最后一次call里跨三层的少量attention patterns大幅传递或抵消；候选集变小时存在恢复稳定性代价。单个head没有复现集合的双向作用，不能简化成一个router head。\n',
'**证据范围。** 这些heads的位置偏晚，可能作用于答案相关信息的读取或整合；本实验不能单独证明逐步图遍历，也不能推出早期calls无关。用with-J运行保存的patterns做移植，可能带来它之前累计计算的信息。此前route/content交叉语义证据来自后16层群体，不自动转移到这个小集合。这里也不重新解释H5：H5的Q-corruption恢复实验与本次J/raw pattern中介实验衡量不同功能。\n',
'## 验证与文件\n',
'确认代码、配置、数据、协议哈希与运行记录一致；冻结配置和确认协议的服务器时间早于运行开始，见confirmation_freeze.json。新64对的128张图与既有416张图不重合，均为十节点单环，token槽对齐。无正确性筛选。\n']
for k in ['native','J','full_rescue_pattern','full_damage_pattern','selected_rescue_pattern','selected_damage_pattern']:t.append(f"- {k}：首token正确 {s[k]['correct']}/64，完整姓名正确 {s[k]['generation_correct']}/64，截断 {s[k]['truncations']}。")
t += ['\nSelf-output必须精确复现logits；self-pattern数值误差和预测一致性在AUDIT.json与analysis.json中记录。代码保留逐head运算，smoke的预测及概率与原脚本精确一致。配对bootstrap与精确检验保存在localize_confirmation64/analysis.json；不把大量候选干预次数算作独立样本数。\n',
'所有模型权重冻结；只跑推理干预，没有新增训练。原始逐例数据在localize_confirmation64/results.jsonl，固定配置在confirmation.json，完整搜索轨迹在SEARCH_LOG.md；图由plot.py读取统计结果直接绘制。论文版本未修改。\n']
(P/'REPORT_CN.md').write_text('\n'.join(t))

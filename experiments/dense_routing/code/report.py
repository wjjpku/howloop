from pathlib import Path
import json
P=Path(__file__).resolve().parents[1];s=json.loads((P/'comparison.json').read_text());b=json.loads((P/'battery_comparison.json').read_text())
fmt=lambda x:f'{100*x:.2f}' if x is not None else 'NA'
rows=['# Dense affine 与低秩 J：配对重跑','', '所有数值为百分比，格式为原低秩 → dense。两组控制器使用同一冻结骨干和训练数据流，训练更新数、学习率及选模规则一致。dense 为 hW+b（65,792 参数）；原方案为 D+AB+b、rank48（25,088 参数）。均从恒等函数初始化，优化参数化不同。','', '## 单次控制与路由交换（4,116 个标签互异样本）','','|骨干 seed|一跳 J|两跳 J|Jone→无J 的 L2 pattern|两跳→一跳：两跳答案|一跳→两跳：一跳答案|','|---|---:|---:|---:|---:|---:|']
def pair(v):return fmt(v['lowrank']['mean'])+' → '+fmt(v['dense']['mean'])
for n,seed in [('A',6),('C',4),('B',3),('D',5),('E',7)]:
 if n not in s:continue
 r=s[n];v=[r['single']['one'],r['single']['two']]+[r['matrix'][k] for k in ['one_to_raw/L2/pattern','two_to_one/L2/pattern','one_to_two/L2/pattern']];rows.append('|'+str(seed)+'|'+'|'.join(map(pair,v))+'|')
rows+=['','## 连续组合（3,200 个当前位置及后四跳互异样本）','','|骨干 seed|一跳→一跳|一跳→两跳|两跳→一跳|两跳→两跳|','|---|---:|---:|---:|---:|']
for n,seed in [('A',6),('C',4),('B',3),('D',5),('E',7)]:
 if n in s:rows.append('|'+str(seed)+'|'+'|'.join(pair(v) for v in s[n]['composition'].values())+'|')
rows+=['','## 固定原 discovery head 的机制比较','','以下以双方共同有效样本为主，避免可解样本集合随 J 改变造成偏差。每个 fit 单独计算，再平均；各自原生分母和 bootstrap 区间见 battery_comparison.json。跨 fit / 跨方法的样本数并不保证相同。','','|骨干|固定 head|选定输出恢复|Pattern→接收图目标|Output→来源目标|J→无J|无J→J|','|---|---:|---:|---:|---:|---:|---:|']
for n,seed in [('A',6),('C',4),('B',3),('D',5),('E',7)]:
 if n not in b:continue
 cells=[]
 for key in ['restore_selected','pattern_rerouted','output_source','J_to_raw','raw_to_J']:
  vals=b[n]['metrics'][key];vv=[]
  for family in ['lowrank_common','dense_common']:
   a=[v[family]['mean'] for v in vals.values()];vv.append(sum(a)/len(a) if all(x is not None for x in a) else None)
  cells.append(fmt(vv[0])+' → '+fmt(vv[1]))
 rows.append('|'+str(seed)+'|'+str(b[n]['locked_head'])+'|'+'|'.join(cells)+'|')
rows+=['','NA 表示至少一个 fit 的双方共同有效样本为空，不能计算配对均值。各自分母下的结果仍保存在 JSON 中。','','## 审计与解释边界','','- 图任务骨干完全冻结；Ouro 原本使用 dense affine，本次未重训。','- 完整 matrix 逐例预测、两次组合、所有四个 head 的干预结果均保存。主机制比较不根据 dense 测试结果重新选 head。','- comparison.json 保留按图配对 bootstrap 的置信区间和 dense-minus-lowrank 差值；battery_comparison.json 同时保留各自及共同有效样本统计。','- 这些对比复用现有训练排除的评估图，不是新的独立确认集。','- 更高任务准确率不自动等于更强路由机制；应分别看控制效果、干预效果和连续组合。','- 未修改论文。']
(P/'结果比较.md').write_text('\n'.join(rows)+'\n');print('\n'.join(rows))

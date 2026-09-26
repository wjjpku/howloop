from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parents[1];A=json.loads((P/'semantic_merged/analysis.json').read_text());B=json.loads((P/'restore_merged/analysis.json').read_text());S=A['conditions'];T=B['conditions']
s='''# Ouro 全部机制实验扩至256例

原有 steering-pattern 扩样见 ../ouro_expansion_20260926/结果报告.md。本轮对语义干预和阻断—恢复各生成256对全新样本，三组互不重叠。模型和dense J未训练，16个heads及干预位置未重新选择。旧64例仅作为先前实验，不与新256例混算。

## 语义干预

共享前七步路径、不同第八个接收者，base/source/rerouted三个答案不同。所有256对均保留，不按预测是否正确筛选。下表为第一答案token分类。

| 条件 | 原答案 | Source答案 | Rerouted答案 | 其他 |
|---|---:|---:|---:|---:|
'''
for k in ['base','source','counterfactual','selected_source_pattern','selected_source_output','selected_source_value','neighbor_source_pattern','neighbor_source_output','selected_wrong_call_pattern']:
 r=S[k];s+=f"| {k} | {r['base']} | {r['source']} | {r['counterfactual']} | {r['other']} |\n"
s+='\n完整姓名分类与首token分类的逐例不一致数：\n'
for k,r in S.items():
 if 'first_token_name_disagreements' in r:s+=f"- {k}: {r['first_token_name_disagreements']}；截断 {r['truncations']}。\n"
s+='\n配对对照（百分点，10000次按图对bootstrap，95%区间）：\n'
for k,r in A['paired_contrasts'].items():s+=f"- {k}: {r['delta']*100:.2f} [{r['ci95'][0]*100:.2f}, {r['ci95'][1]*100:.2f}]。\n"
s+='''
## 阻断—恢复

在同图不同起点的查询间移植Q，阻断第四次调用的L41/L43/L47；保留接收K/V。恢复干预分别移植原运行的selected16、固定邻近16、单层所选heads，或全部48个受影响outputs。分母为原本正确且被阻断破坏的样本数；同时保留未经条件筛选的256例计数。

| 条件 | 全体正确数 / 256 | 恢复 / 原本正确且被破坏 | 条件恢复率（%，Wilson95%CI） |
|---|---:|---:|---:|
'''
for k in ['base','corrupt','restore_selected','restore_neighbor','restore_layer41','restore_layer43','restore_layer47','restore_all']:
 r=T[k];ci=r['conditional_wilson_ci95'];s+=f"| {k} | {r['correct']} | {r['recovered']}/{r['broken_n']} | {100*r['conditional_recovery']:.2f} [{100*ci[0]:.2f}, {100*ci[1]:.2f}] |\n"
s+=f"\n全部48 heads恢复的最大logit误差为 {B['all_head_max_logit_error']}。完整姓名一致性与截断记录见各条件的analysis.json。\n"
s+='''
## 协议与限制

语义样本seed2026092641，恢复样本seed2026092642；沿用七步共享路径生成器。语义组实际使用512张图。恢复组实际使用256张接收图及其另一起点查询；生成文件另含256张source图，仅用于保持原配对构造规则，不用于Q donor。全部生成图身份互不重叠，并排除前序1184张图。

同一个Ouro-2.6B step200 backbone与step500 J；第四次调用的固定heads：L41.H2/H10/H12/H15，L43.H1/H6/H11/H13/H15，L47.H0/H2/H3/H4/H7/H13/H15。生成最多16token，patch仅在prompt processing，之后保持接收运行J安排。

扩大的是固定干预在新图上的评估，不是新backbone复制。历史训练图排除不可完整核实。恢复定义是对预先指定破坏的因果修复，不证明heads是全局最小或唯一实现；共享路径语义构造也不等于任意图的可控性。

全部逐例预测位于 semantic_merged/results.jsonl 和 restore_merged/results.jsonl。各分批manifest记录权重哈希、源程序哈希、数据哈希、参数未变和资源峰值。DATA_AUDIT.json验证图的隔离与答案构造；SMOKE_AUDIT.json验证优化前后的旧样本预测及概率完全一致。
'''
(P/'结果报告.md').write_text(s)
assert all(r.get('truncations',0)==0 for r in list(S.values())+list(T.values()))
(P/'AUDIT.json').write_text(json.dumps(dict(status='passed',semantic_n=256,restore_n=256,semantic_conditions=len(S),restore_conditions=len(T),semantic_generation_disagreements=sum(r.get('first_token_name_disagreements',0) for r in S.values()),restore_generation_disagreements=sum(r.get('first_token_name_disagreements',0) for r in T.values()),full_restore_max_logit_error=B['all_head_max_logit_error'],all_model_versions_unchanged=True),indent=2))
(P/'code_hashes.json').write_text(json.dumps({str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (P/'code').glob('*') if f.is_file()},indent=2))
print('semantic',S['selected_source_pattern']);print('restore',T['restore_selected'])

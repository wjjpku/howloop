from pathlib import Path
import json,shutil,datetime,subprocess,re
R=Path(__file__).resolve().parent;P=Path('/Users/jiaju/Documents/looped_transformer_paper_repro_20260920/paper_spotlight_v1');F=R.parent/'fig4_redesign';s=json.loads((R/'analysis/summary.json').read_text())['results'];old=json.loads((R/'screening/analysis/summary.json').read_text())['results'];p=P/'main_v64_subfigures.tex'
b=P/'revisions'/('before_selected_n10_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));b.mkdir();shutil.copy2(p,b/p.name);shutil.copy2(F/'interval_values.json',b/'interval_values.json');shutil.copy2(P/'figures/approved_three/fig4_shared_axis.pdf',b/'fig4.pdf')
def rows_for(name):
 head={'A':2,'C':3}[name];out=[]
 for typ,key in [('cross','pattern_cf'),('cross','context_cf'),('cross','pattern_donor'),('cross','context_donor'),('restoration',f'clean_context_H{head}'),('restoration',f'clean_context_H{(head+1)%4}'),('restoration','clean_context_H0123'),('pattern_swap','native'),('pattern_swap','full_J'),('pattern_swap','rescue'),('pattern_swap','damage')]:
  out.append([(s[name][f]['cross_graph']['own_eligible'] if typ=='cross' else s[name][f][typ])[key] for f in ['1','2']])
 return out
rows=json.loads((F/'interval_values.json').read_text());aa=rows_for('A');cc=rows_for('C')
for row,vals in zip(rows,aa):row['graph']=vals
(F/'interval_values.json').write_text(json.dumps(rows,indent=2));subprocess.run(['python3',str(F/'draw_shared_axis.py')],check=True);shutil.copy2(F/'fig4_shared_axis.pdf',P/'figures/approved_three/fig4_shared_axis.pdf')
# Matching appendix view makes model heterogeneity visible without averaging selected backbones.
(R/'interval_values.json').write_text(json.dumps([{'graph':a,'ouro':c} for a,c in zip(aa,cc)],indent=2))
plot=(F/'draw_shared_axis.py').read_text().replace("es=row[key] if j==0 else [row[key]]","es=row[key]").replace("if j==0 else 0","if len(es)>1 else 0").replace("['Graph A · N10','Ouro']","['N10 A · seed 6','N10 C · seed 4']").replace('fig4_shared_axis.{ext}','selected_backbones.{ext}')
(R/'draw_selected.py').write_text(plot);subprocess.run(['python3',str(R/'draw_selected.py')],check=True);shutil.copy2(R/'selected_backbones.pdf',P/'figures/approved_three/n10_selected_backbones.pdf')
mean=lambda row:sum(v['mean'] for v in row)/2*100
v=list(map(mean,aa));r,full,rescue,damage=v[7:];gain=(rescue-r)/(full-r)*100
text=p.read_text()
text=re.sub(r'The selected N10 graph head H2 repairs .*?Other individual heads recover at most .*?\\%.',lambda _: f'The selected N10 graph head H2 repairs {v[4]:.1f}'+r'\% of eligible broken cases on average, compared with '+f'{v[5]:.1f}'+r'\% for the prespecified H3 control.',text)
text=text.replace('(82.5\\% in Graph A; 71.9\\% in Ouro)',f'({v[0]:.1f}'+r'\% in Graph A; 71.9\% in Ouro)').replace('(85.0\\%; 95.3\\%)',f'({v[3]:.1f}'+r'\%; 95.3\%)')
start=text.index('In N10 Graph A, steered patterns raise');end=text.index(' For Ouro,',start)
text=text[:start]+f'In N10 Graph A, steered patterns raise raw accuracy from {r:.1f}'+r'\% to '+f'{rescue:.1f}'+r'\%, compared with '+f'{full:.1f}'+r'\% for full steering; reverse replacement reduces it to '+f'{damage:.1f}'+r'\% (Fig.~\ref{fig:parallel-mechanism}c). Pattern transfer recovers '+f'{gain:.1f}'+r'\% of the mean control gain, indicating substantial but incomplete mediation.'+text[end:]
ns=[s['A'][f]['cross_graph']['own_eligible']['pattern_cf']['n'] for f in ['1','2']];nr=[s['A'][f]['restoration']['clean_context_H2']['n'] for f in ['1','2']];nt=s['A']['1']['pattern_swap']['native']['n']
for a,z in [('4,493/4,491',f'{nr[0]:,}/{nr[1]:,}'),('3,977/3,973',f'{ns[0]:,}/{ns[1]:,}'),('3,977 and 3,973',f'{ns[0]:,} and {ns[1]:,}'),('4,112',f'{nt:,}'),('2026092403','2026092404'),('18,543 unused','17,519 unused')]:text=text.replace(a,z)
text=text.replace('including their generated graph variants.','including their generated graph variants and the preceding 1,024-graph A-only evaluation.')
text=text.replace('maximum checked logit discrepancies are below $1.6\\times10^{-5}$','maximum checked logit discrepancies are recorded per backbone in the evidence archive')
text=text.replace('A CPU replay of four raw/corrupted pairs and all ten currents matches all 254 event rows and their eligibility masks.','For each selected backbone, a CPU replay of four raw/corrupted pairs and all ten currents matches all 254 event rows and their eligibility masks.')
text=text.replace('We evaluate new graph identities and pairings without retraining or head reselection;',r'A is selected for its strongest complete mechanism chain in the earlier five-backbone evaluation; C is retained as a strong-control, weaker-mediation contrast (Appendix~\ref{app:n10-selection}). We evaluate a separate fresh confirmation batch without retraining or head reselection;')
# Refresh fit-level table.
a=text.index('Condition & Fit 1 & Fit 2');z=text.index('\\bottomrule',a)
labels=['Pattern: raw-graph answer','Output: raw-graph answer','Pattern: corrupted answer','Output: corrupted answer','Restore H2','Restore H3','Restore all heads','No J','Full J','J patterns into raw','Raw patterns into J']
table='Condition & Fit 1 & Fit 2'+r'\\\midrule'+'\n'
for i in [4,5,6,0,1,2,3,7,8,9,10]:table+=labels[i]+''.join(f" & {x['mean']*100:.2f}" for x in aa[i])+r' \\'+'\n'
text=text[:a]+table+text[z:]
add=r'''\subsection{Selection and fresh confirmation across N10 backbones}
\label{app:n10-selection}
The N10 A--E suite comprises seeds 6, 3, 4, 5 and 7 under the same D8L6 training protocol. All five previously completed both controller fits and the mechanism battery. We screened those earlier results before evaluating the fresh batch above. A was retained as the strongest complete-chain example; C was retained as a contrast because its one-hop control succeeds almost perfectly while routing transfer is weaker. This is an explicitly selected case study, not an estimate of typical behavior across random seeds. Head choices remain fixed from the original discovery split. No new-confirmation result is used to change the selected models or heads.

\begin{table}[!htbp]\centering\small
\caption{Earlier N10 screening results used to select cases, averaged over two controller fits (percent). Cross-graph eligibility counts are reported per fit out of 5,120 candidates. These data are separate from the fresh confirmation.}
\begin{tabular}{@{}lrrrrr@{}}\toprule
Model & Full J & Pattern transfer & Pattern: raw & Restore head & Eligible pairs\\\midrule
'''
for n,fs in old.items():
 h={'A':2,'B':0,'C':3,'D':0,'E':0}[n];avg=lambda typ,key:sum(fs[f][typ][key]['mean'] for f in ['1','2'])/2*100
 cross=sum(fs[f]['cross_graph']['own_eligible']['pattern_cf']['mean'] for f in ['1','2'])/2*100
 counts='/'.join(str(fs[f]['cross_graph']['own_eligible']['pattern_cf']['n']) for f in ['1','2'])
 add+=f"{n} & {avg('pattern_swap','full_J'):.1f} & {avg('pattern_swap','rescue'):.1f} & {cross:.1f} & {avg('restoration',f'clean_context_H{h}'):.1f} & {counts}"+r' \\'+'\n'
add+=r'''\bottomrule\end{tabular}\end{table}
\begin{figure}[!htbp]\centering
\includegraphics[width=\linewidth]{figures/approved_three/n10_selected_backbones.pdf}
\caption{Fresh confirmation of the selected N10 cases on the same 512 paired graphs. Bars average two fits, with separate fit markers and 95\% graph-bootstrap intervals. A uses H2 and C uses H3 in (a,b), with the next head modulo four as control; (c) transfers all four second-layer patterns. Selection precedes this confirmation, and models are not pooled.}
\label{fig:n10-selected}\end{figure}
'''
cv=list(map(mean,cc));add+=f"On fresh confirmation, C attains {cv[8]:.1f}"+r'\% with full J, but only '+f'{cv[9]:.1f}'+r'\% after transferring J patterns into the raw run (raw accuracy '+f'{cv[7]:.1f}'+r'\%). Its selected-head restoration is '+f'{cv[4]:.1f}'+r'\%, and pattern transfer favors the raw-graph answer in '+f'{cv[0]:.1f}'+r'\% of eligible pairs. Thus strong behavioral control does not guarantee the same concentration of mediation in these tested heads. This contrasts with A without identifying a unique alternative mechanism for C.'+'\n\n'
text=text.replace('\\subsection{Historical N8 configurations}',add+'\\subsection{Historical N8 configurations}',1);p.write_text(text)
e=P/'evidence/n10_selected_mechanism';e.mkdir(exist_ok=True)
for f in ['SELECTION_LOCK.json','CPU_COMPARISON.json','datasets.json','prepare_run.py','analyze.py']:shutil.copy2(R/f,e/f)
shutil.copy2(R/'analysis/summary.json',e/'confirmation_summary.json');shutil.copy2(R/'screening/analysis/summary.json',e/'screening_summary.json')
print('A',v,'gain',gain,'C',cv,'denominators',nr,ns,nt)

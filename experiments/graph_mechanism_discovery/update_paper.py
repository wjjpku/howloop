from pathlib import Path
import json,shutil,datetime
R=Path(__file__).resolve().parent;P=Path('/Users/jiaju/Documents/looped_transformer_paper_repro_20260920/paper_spotlight_v1');p=P/'main_v64_subfigures.tex';b=P/'revisions'/('before_n10_mechanism_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));b.mkdir();shutil.copy2(p,b/p.name);shutil.copy2(P/'figures/approved_three/fig4_shared_axis.pdf',b/'fig4.pdf')
s=json.loads((R/'analysis/summary.json').read_text())['results']['A'];mean=lambda typ,key:sum(s[f][typ][key]['mean'] for f in ['1','2'])/2*100
cf=lambda key:sum(s[f]['cross_graph']['own_eligible'][key]['mean'] for f in ['1','2'])/2*100
restore=mean('restoration','clean_context_H2');control=mean('restoration','clean_context_H3');raw=mean('pattern_swap','native');full=mean('pattern_swap','full_J');rescue=mean('pattern_swap','rescue');damage=mean('pattern_swap','damage');gain=(rescue-raw)/(full-raw)*100
text=p.read_text()
text=text.replace('These N10 models are distinct from the historical N8 mechanism cohort used in Section~\\ref{sec:hypothesis}.','The N10 D8L6 backbone A and its two one-hop maps are reused for the mechanism tests in Section~\\ref{sec:hypothesis}.')
text=text.replace('Here Backbone A denotes the N10 model. The historical N8 maps used below are a separate cohort; their protocols are listed in Appendix~\\ref{app:main-config}.','Backbone A denotes the N10 model. The same two one-hop maps are reused in the mechanism tests on newly paired held-out graphs; provenance and cohort definitions are in Appendix~\\ref{app:n10-mechanism}.')
text=text.replace('Graph A here denotes the historical eight-node backbone and its two rank-48 maps, distinct from the new N10 target-control example. Thus its mechanism rates and the N10 control rates are not estimates on a shared cohort.','Graph A is the same N10 checkpoint and the same two rank-48 one-hop maps used in Figure~\\ref{fig:target}. We evaluate new graph identities and pairings without retraining or head reselection; the behavioral and mechanism cohorts remain distinct.')
text=text.replace('Graph suite A--E & Historical eight-node D8L6 mechanism cohort; distinct from the new N10 trajectory and target-control models. Pure-CE unification is complete only for historical A.','Graph A / historical A--E & Main figures use one N10 D8L6 checkpoint and fixed maps; appendix A--E comparisons retain their separately identified N8 cohort.')
text=text.replace('Graph A uses H0 in (a,b)','Graph A uses H2 in (a,b)')
text=text.replace('Graph uses 3,574 broken cases per fit in (a), 4,096 pairs in (b), and 3,056 cases in (c).','Graph uses 4,493/4,491 eligible broken cases in (a), 3,977/3,973 eligible pairs out of 5,120 in (b), and 4,112 label-distinct cases in (c).')
text=text.replace('The selected graph head repairs 70--78\\% of broken cases across the two fits; other individual heads repair none.',f'The selected N10 graph head H2 repairs {restore:.1f}\\% of eligible broken cases on average (60.1\\% and 59.7\\% across fits), compared with {control:.1f}\\% for the prespecified H3 control. Other individual heads recover at most 4.1\\%.')
text=text.replace('(80.1\\% in Graph A; 71.9\\% in Ouro)',f'({cf("pattern_cf"):.1f}\\% in Graph A; 71.9\\% in Ouro)').replace('(86.5\\%; 95.3\\%)',f'({cf("context_donor"):.1f}\\%; 95.3\\%)')
text=text.replace('Steered patterns bring raw-run accuracy close to full steering in both models; raw patterns substantially impair steered runs (Fig.~\\ref{fig:parallel-mechanism}c).',f'In N10 Graph A, steered patterns raise raw accuracy from {raw:.1f}\\% to {rescue:.1f}\\%, compared with {full:.1f}\\% for full steering; reverse replacement reduces it to {damage:.1f}\\% (Fig.~\\ref{{fig:parallel-mechanism}}c). Pattern transfer recovers {gain:.1f}\\% of the mean control gain, indicating substantial but incomplete mediation.')
text=text.replace('Other graph backbones show heterogeneous results, reported in Appendix~\\ref{app:causal}.','Historical N8 graph backbones show heterogeneous results in Appendix~\\ref{app:causal}; those results are separate from the N10 mechanism test above.')
text=text.replace('In Backbone A, one pair of pure-CE maps connects behavior, semantic patching, and routing transfer.','In N10 Backbone A, the same pair of pure-CE maps connects behavior, restoration, semantic patching, and routing transfer on separate held-out cohorts.')
# Add full reproducibility scope ahead of the historical appendix material.
anchor='\\paragraph{Shared graph setup.}'
protocol=r'''\subsection{N10 main-figure mechanism protocol}
\label{app:n10-mechanism}
Figures~\ref{fig:target} and \ref{fig:parallel-mechanism} use the same N10 D8L6 seed-6 backbone (checkpoint step 16,000) and the same two rank-48 one-hop maps. Their saved hashes were checked before evaluation, and parameter version counters remain unchanged. The second-layer H2 was already selected on the original 32-graph discovery split by mean eligible pattern-transfer accuracy across the two fits, with ties resolved by head index; it is not selected on the new results. H3, the next head, is the prespecified control; all four individual restoration outcomes are retained.

Sampling seed 2026092403 selects 512 raw and 512 corrupted permutation graphs, without replacement, from 18,543 unused graphs in the pre-training exclusion reserve. All 1,024 graph identities are excluded from backbone and controller fitting and from graph identities used by the preceding discovery, smoke and confirmation mechanism evaluations, including their generated graph variants. The reserve itself was generated as one-swap relatives of earlier held-out graphs, so this is fresh graph-identity confirmation, not a claim of an unrelated graph-generating distribution. Each raw graph is paired with one independently drawn reserved corrupted graph; unlike the earlier local-edit protocol, pairs need not share most edges. Node slots remain aligned. All ten current nodes are evaluated, and the corrupted current is shifted by three modulo ten. The input start is chosen by inverse traversal so the prescribed current is reached after eight graph steps. The fixed backbone runs six calls before the same all-token $J$ and seventh frozen call.

The 5,120 candidate pairs are fixed before inspecting outcomes. For cross-graph patches, the three answer labels must differ and both runs must have correct current-node and next-node readouts; 3,977 and 3,973 pairs qualify. Pattern and output comparisons use identical masks within each fit. Restoration additionally requires the same-graph alternative-current run to have correct current/next readouts and a distinct target; its denominator is the 4,493/4,491 eligible clean transitions made incorrect by query corruption. Pattern transfer uses 4,112 cases with current, one-hop and two-hop labels pairwise distinct, without success filtering. These are different cohorts; Figure~\ref{fig:target} retains its original 4,110-example behavioral cohort.

Intervals use 5,000 graph-cluster bootstrap draws separately for each fit, resampling the 512 raw graphs (and their paired corrupted graphs). The two fits are not backbone replications. Direct-forward, instrumented-pattern, independent output-hook and self-patch checks preserve predictions; maximum checked logit discrepancies are below $1.6\times10^{-5}$. A CPU replay of four raw/corrupted pairs and all ten currents matches all 254 event rows and their eligibility masks. Every event and all head controls are archived.

\begin{table}[!htbp]\centering\small
\caption{Fresh N10 mechanism results, percentages by controller fit. Historical N8 results below are not pooled with this experiment.}
\begin{tabular}{@{}lrr@{}}\toprule
Condition & Fit 1 & Fit 2\\\midrule
'''
for label,typ,key in [('Restore H2','restoration','clean_context_H2'),('Restore H3','restoration','clean_context_H3'),('Restore all heads','restoration','clean_context_H0123'),('Pattern: raw-graph answer','cross','pattern_cf'),('Output: raw-graph answer','cross','context_cf'),('Pattern: corrupted answer','cross','pattern_donor'),('Output: corrupted answer','cross','context_donor'),('No J','pattern_swap','native'),('Full J','pattern_swap','full_J'),('J patterns into raw','pattern_swap','rescue'),('Raw patterns into J','pattern_swap','damage')]:
 vals=[(s[f]['cross_graph']['own_eligible'] if typ=='cross' else s[f][typ])[key]['mean']*100 for f in ['1','2']];protocol+=label+f' & {vals[0]:.2f} & {vals[1]:.2f}'+r' \\'+'\n'
protocol+=r'''\bottomrule\end{tabular}\end{table}

\subsection{Historical N8 configurations}
The remaining A--E configurations and historical tables in this appendix describe the earlier N8 campaign; they do not specify the new N10 main-figure cohort.

'''
text=text.replace(anchor,protocol+anchor,1)
p.write_text(text)
shutil.copy2(R.parent/'fig4_redesign/fig4_shared_axis.pdf',P/'figures/approved_three/fig4_shared_axis.pdf')
e=P/'evidence/n10_fresh_mechanism';e.mkdir(exist_ok=True)
for name in ['MANIFEST.json','datasets.json','analyze.py','prepare.py']:shutil.copy2(R/name,e/name)
shutil.copy2(R/'analysis/summary.json',e/'summary.json');shutil.copy2(R/'evaluation/confirmation/manifest.json',e/'evaluation_manifest.json');shutil.copy2(R/'cpu_check/comparison.json',e/'cpu_comparison.json')
print('mean gain recovered',gain)

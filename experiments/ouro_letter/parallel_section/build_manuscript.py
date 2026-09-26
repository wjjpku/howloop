from pathlib import Path
import json,re
P=Path(__file__).resolve().parent;ROOT=Path('/data/paperexperiment/Documents/looped_transformer_paper_repro_20260920/paper_spotlight_v1');G=json.loads((P/'graph_summary.json').read_text())['results'];S=json.loads((P/'parallel_confirmation64/analysis.json').read_text())['conditions'];R=json.loads((P/'parallel_restore64/analysis.json').read_text())['conditions'];M=json.loads((P.parent/'localization/localize_confirmation64/analysis.json').read_text())['conditions']
s=(ROOT/'main_v60_narrative.tex').read_text();s=s.replace('% Version 58: merged narrative preview, including three graph mechanism experiments and revised Ouro section.','% Version 61: parallel graph/Ouro mechanism tests; inherits v60 appendix move; updated uniform graph fits.')
macros={'OuroPatternRoute':S['selected_source_pattern']['counterfactual'],'OuroOutputRoute':S['selected_source_output']['counterfactual'],'OuroOutputSource':S['selected_source_output']['source'],'OuroPatternSource':S['selected_source_pattern']['source'],'OuroBroken':R['corrupt']['broken_n'],'OuroRestored':R['restore_selected']['recovered'],'OuroNeighborRestored':R['restore_neighbor']['recovered']}
s=s.replace('\\begin{document}', '\n'.join('\\newcommand{\\'+k+'}{'+str(v)+'}' for k,v in macros.items())+'\n\\begin{document}')
oldouro=s[s.index('\\section{Routing Dependence in a Larger Recurrent Model}'):s.index('\\section{Generalization Across Lengths and Limits of Continued Control}')]
oldouro=oldouro.replace('\\section{Routing Dependence in a Larger Recurrent Model}','\\subsection{Earlier single-head Ouro interventions}').replace('\\label{sec:large}','\\label{app:ouro-earlier}')
oldouro='The following archived 256-prompt study predates the compact-set experiments and uses a different intervention scope. It is retained as a separate local-head diagnostic.\n\n'+oldouro
start=s.index('\\section{Tracing How Steering Changes the Computation}');end=s.index('\\section{Generalization Across Lengths and Limits of Continued Control}')
s=s[:start]+(P/'section.tex').read_text()+'\n'+s[end:]
abstract=r'''Looped Transformers reuse the same weights across recurrent calls, yet the hidden state can direct those weights toward different task targets. We use activation steering to control frozen computation and trace its effect through attention in a graph-walk model and task-adapted Ouro-2.6B. Parallel interventions distinguish the weights that select where to read from the head outputs that carry the result: pattern patches favor answers obtained by following the source route on the receiving graph, whereas output patches favor the source answer. Restoring clean outputs at selected positions repairs disrupted reads. Finally, swapping patterns between steered and raw runs transfers most of the steering gain. In Ouro, 16 heads at the final call raise raw accuracy from 0/64 to 61/64, compared with 62/64 for full steering; the reverse swap reduces it to 0/64. These experiments connect state control to a compact attention pathway. Additional graph backbones expose model-dependent limits, while parity illustrates how steering can extend useful computation without guaranteeing sustained correction.'''
s=re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}',lambda m:'\\begin{abstract}\n'+abstract+'\n\\end{abstract}',s,flags=re.S)
s=s.replace('We next test a related routing dependence in Ouro-2.6B, before examining the reach and limits of control:', 'We apply the same three tests to graph-walk computation and Ouro-2.6B, then examine the reach and limits of control:')
s=s.replace('We test these roles inside steered one-hop runs, where the intended next graph target is explicit.', 'We test these roles in steered graph transitions and in a task-adapted recurrent language model.')
s=s.replace('The paper moves from observation to control, then follows the routing evidence through a larger model before testing generalization and continued execution.', 'The paper moves from observation to control, compares the same causal tests in graph models and Ouro, then examines generalization and continued execution.')
s=s.replace('Graph mechanism (Sec.~\\ref{sec:hypothesis}) & Pattern and output patches have distinct effects; corrupt--restore validates a local site; pattern transfer carries most one-hop steering in A.', 'Parallel mechanism (Sec.~\\ref{sec:hypothesis}) & Pattern/output swaps, output restoration, and bidirectional routing transfer connect steering to attention in Graph A and Ouro.')
s=re.sub(r'Ouro \(Sec\.~\\ref\{sec:large\}\).*?\\\\ Length',r'Length',s)
s=s.replace('figures/v58/preview/target_control.pdf','figures/v61/target_control.pdf')
s=s.replace('Two-hop accuracy rises to 84.20\\% and 59.78\\%, but the first fit also changes the pre-$F$ current-node readout.', 'Two-hop accuracy rises to 61.88\\% and 71.24\\%, while pre-$F$ current-node readability falls to 88.02\\% and 81.09\\%.')
s=s.replace('Fit-level values and an earlier shuffled-state control appear in Appendix~\\ref{app:target}.','Fit-level values appear in Appendix~\\ref{app:target}.')
s=s.replace("Ouro's selective ablations and limited rescue give a related example at larger scale.", "Ouro extends this analysis to a compact 16-head set at the final call, while a smaller ten-head set loses recovery accuracy.")
s=s.replace('Across five circuit models, pattern patches favor answers obtained by following a transferred route through base content, while head-output patches favor the source answer. Corrupt--restore validates a useful local head in two backbones, and a matched pattern swap transfers most of the one-hop steering gain in the main checkpoint.', 'Parallel tests in Graph A and Ouro distinguish reads over receiving-graph content from source outputs, validate useful restoration sites, and transfer most of the steering gain through patterns. The same fixed 16-head set supports all three Ouro tests; uniform graph-backbone comparisons reveal substantial variation beyond A.')
s=s.replace('Different pattern/output patching effects replicate across five checkpoints.', 'Graph A supports the three-test mechanism chain; uniform A--E reruns expose heterogeneous behavior and mediation.')
s=s.replace('Ouro L34.H5 panel & causal component intervention & Pattern dominance and distributed necessity in one frozen 2.6B checkpoint/task.', 'Ouro compact patterns & causal component intervention & The fixed 16-head set supports semantic patching, restoration, and routing transfer in one task-trained checkpoint.')
s=s.replace('Joint Ouro patterns & component intervention & Specific damage is strong; rescue is limited.', 'Earlier Ouro head sets & component intervention & Strong damage and limited rescue at earlier, smaller head selections; separate from the final 16-head set.')
a=s.index('\\paragraph{Unified Backbone-A protocol.}');b=s.index('\\paragraph{Other tasks.}',a)
s=s[:a]+r'''\paragraph{Unified graph protocol.}
All A--E maps are refitted under the same rule: rank-48 $D+AB+b$, post-$F$ target CE, AdamW with learning rate $10^{-4}$, batch 128, gradient clipping 1, and 8,000 updates. Two initialization seeds are used per target. Initialization and training sampling are seeded separately before use; this corrects the earlier initialization order, so the rerun is not a bitwise replay of the historical fits. Validation every 400 updates selects the earliest best checkpoint. A's one-hop updates are 800 and 400; both two-hop fits select update 8000. The same two one-hop fits are used for behavior and all three graph mechanism tests.

Evaluation uses the existing 512 permutation graphs and all eight starts. A's cross-graph test contains 4,096 distinct-answer pairs; matched-input pattern transfer uses 3,056 label-distinct cases from 505 graphs, without success selection. Head B2.H0 is fixed from earlier discovery. The evaluation graphs have been inspected before and are not excluded from the finite-domain training generator; these are matched in-domain comparisons, not new graph-disjoint confirmation or OOD tests. A--E retain their original backbone histories and fixed selected heads; only map fitting and evaluation are standardized. The separate 12-backbone G4 continuation experiment retains its own protocol.

'''+s[b:]
s=s.replace('The head panel and four-arm test reuse the same 256 depth-eight prompts; head interventions cover prompt processing, while branch schedules also apply during generation.', 'The main parallel tests use the fixed 16-head set at call 4; their sample sets and metrics are detailed in Appendix~\\ref{app:parallel}. The earlier H5 panel and continuation test reuse a separate 256-prompt cohort. Pattern/output patches act during prompt processing, while the receiving J schedule remains active during generation.')
# Replace outdated mixed-protocol graph appendix and target/mediation tables with the standardized rerun.
a=s.index('\\section{Detailed Causal Outcomes}');b=s.index('\\section{Restricted Controller Controls}',a)
app=r'''\section{Detailed Graph Mechanism Outcomes}
\label{app:causal}
The standardized rerun preserves every backbone and both controller fits. Cross-graph eligibility requires distinct answers and correct clean current/next readouts in both runs. All 4,096 pairs qualify for A; the smaller eligible populations for other backbones must not be generalized to all inputs. Source pairs are constructed before evaluation and are not resampled to improve results. Table~\ref{tab:outcome-ledger} gives both predicted answer categories; the remaining outcomes include the base answer and other nodes.
\begin{table}[!htbp]\centering\small
\caption{Uniform pure-CE rerun. Rates are percentages within the eligible cross-graph cohort. The two column groups ask which answer a patch produces.}
\label{tab:outcome-ledger}\label{tab:cross-summary}
\begin{tabular}{@{}lrrrrr@{}}\toprule
 & & \multicolumn{2}{c}{Source route + base graph} & \multicolumn{2}{c}{Source answer}\\
Fit & Eligible / 4096 & Pattern & Output & Pattern & Output\\\midrule
'''
for model,fits in G.items():
 for seed,v in fits.items():
  q=v['cross_graph']['own_eligible'];app+=f"{model}/{seed} & {q['pattern_cf']['n']} & {100*q['pattern_cf']['mean']:.2f} & {100*q['context_cf']['mean']:.2f} & {100*q['pattern_donor']['mean']:.2f} & {100*q['context_donor']['mean']:.2f} \\\\\n"
app+=r'''\bottomrule\end{tabular}\end{table}

The causal signature and transfer strength vary across backbones. B has high one-hop post-$F$ accuracy but often changes the answer readout before $F$; C has a strong source-route/base-graph contrast yet little raw-state pattern rescue; D's one-hop behavior and eligible population are weak; E gives partial rescue. These are boundaries on extending A's full account, not failed models removed from the comparison.
\begin{table}[!htbp]\centering\small
\caption{Same-input pattern transfer under uniform map training (\%). Each fit uses the same 3,056 label-distinct instances.}
\begin{tabular}{@{}lrrrr@{}}\toprule
Fit & Raw & With $J$ & Pattern rescue & Pattern damage\\\midrule
'''
for model,fits in G.items():
 for seed,v in fits.items():
  q=v['pattern_swap'];app+=f"{model}/{seed} & "+' & '.join(f"{100*q[k]['mean']:.2f}" for k in ['native','full_J','rescue','damage'])+r' \\'+'\n'
app+=r'''\bottomrule\end{tabular}\end{table}

Graph corrupt--restore replaces all second-layer answer queries using another current node in the same graph, keeping base keys and values. For A, 3,574 transitions per fit are originally correct and become wrong. Restoring H0 recovers 77.62\% and 70.23\%; every other individual head recovers zero, and all four heads recover all broken cases. This conditional metric is not unconditional task accuracy. Fit-level graph-cluster bootstrap intervals use 5,000 draws; controller fits are not counted as independent backbones. Behavioral predictions were independently replayed on CPU, and identity/instrumentation checks were performed on representative batches.

\section{Target-Selection Results}
\label{app:target}
The main target figure and same-input pattern test use the same 3,056 label-distinct instances from 505 graphs. Values come from the standardized rerun; historical map results are not pooled with these fits.
\begin{table}[!htbp]\centering\small
\caption{Graph A target accuracy before and after $F$, and current-node readability after $J$ (\%).}
\label{tab:target}
\begin{tabular}{@{}lrrr@{}}\toprule
Target / fit & Pre-$F$ target & Post-$F$ target & Pre-$F$ current\\\midrule
'''
for hop,label in [('one','One hop'),('two','Two hops')]:
 for seed,v in G['A'].items():
  q=v['behavior']['J_'+hop]['distinct_labels'];app+=f'{label} / {seed} & '+' & '.join(f"{100*q[k]['mean']:.2f}" for k in ['pre_F_'+hop,'post_F_'+hop,'pre_F_current'])+r' \\'+'\n'
app+=r'''\bottomrule\end{tabular}\end{table}

\section{Backbone-A Pattern Transfer Details}
\label{app:fourcell}
The two pure-CE fits connect behavior, cross-graph patching, restoration, and same-input pattern transfer. Raw accuracy is 37.17\% and full steering is 100\% in both fits. Pattern rescue is 99.90\% and 99.80\%; reverse-swap accuracy is 50.16\% and 50.46\%. These correspond to 99.84\% and 99.69\% of the net steering gain, respectively. The main bars average fit-level rates, while dots preserve variation between fits. Remaining reverse-swap accuracy is compatible with other useful changes induced by $J$; it does not isolate their location. The two-hop results above are separate target-specific fits and do not establish the same mechanism for two-hop control.

'''
s=s[:a]+app+s[b:]
# Keep old Ouro figure and its label available, explicitly historical and outside the main narrative.
a=s.index('\\section{Supplementary Ouro Interventions}')
s=s[:a]+(P/'ouro_appendix.tex').read_text()+'\n'+s[a:]
a=s.index('\\paragraph{Head-intervention configuration.}')
s=s[:a]+oldouro+'\n'+s[a:]
# Use ordinary descriptions throughout, retaining internal cross-reference labels.
s=s.replace('base-graph counterfactual','answer from the source route on the base graph').replace('Base-graph counterfactual','Answer from the source route on the base graph')
s=re.sub(r'\bdonors\b','sources',s);s=re.sub(r'\bdonor\b','source',s);s=s.replace('semantic counterfactual answers','distinct predicted answers').replace('counterfactual','alternative-answer')
(ROOT/'main_v61_parallel_mechanism.tex').write_text(s)
# Standalone section shares the exact source and macros, figures and references to external sections have simple descriptive text.
sec=(P/'section.tex').read_text();sec=re.sub(r'Section~\\ref\{sec:target\}', 'the steering experiment',sec);sec=re.sub(r'Appendix~\\ref\{[^}]+\}', 'the supplementary methods',sec)
sec=sec.replace('\\subsection{Restoring', '\\clearpage\n\\subsection{Restoring').replace('\\subsection{Pattern swaps', '\\clearpage\n\\subsection{Pattern swaps')
preamble=s[:s.index('\\begin{document}')];preamble=preamble.replace('\\usepackage{iclr2027_conference,times}',r'\usepackage[margin=1in]{geometry}'+'\n'+r'\usepackage{times}');preamble=re.sub(r'\\title\{.*?\}\n\\author',lambda m:r'\title{Parallel Mechanism Analysis: Graph Models and Ouro}'+'\n'+r'\author',preamble,flags=re.S)
(ROOT/'section_v61_parallel.tex').write_text(preamble+'\n\\begin{document}\n\\setcounter{section}{4}\n'+sec+'\n\\end{document}\n')
(P/'section_final.tex').write_text(sec)

from pathlib import Path
import json
P=Path(__file__).resolve().parent;S=json.loads((P/'parallel_confirmation64/analysis.json').read_text())['conditions'];R=json.loads((P/'parallel_restore64/analysis.json').read_text())['conditions'];L=json.loads((P.parent/'localization/localize_confirmation64/analysis.json').read_text())['conditions']
t=r'''\section{Parallel Ouro Mechanism Protocols and Outcomes}
\label{app:parallel}

\paragraph{Fixed model and intervention positions.}
All panels freeze the same task-adapted Ouro-2.6B step-200 checkpoint and the same step-500 dense affine map, inserted before calls 2--4 on all token states. Attention outputs are patched before the output projection, during prompt processing only; the receiving run retains its normal J schedule during greedy generation. Indices are zero-based: the selected heads are L41.H2/H10/H12/H15, L43.H1/H6/H11/H13/H15, and L47.H0/H2/H3/H4/H7/H13/H15, all at the fourth recurrent call. Their selection used eight discovery pairs and a separate 64-pair pattern-transfer confirmation. The semantic and restoration extensions do not reselect heads.

\paragraph{Selection and compression.}
The search tested late layers, contiguous layer groups, head groups, individual removals, and call subsets. All failed candidates are retained. Sixteen heads at call 4 were a prespecified sensitivity set alongside a ten-head primary candidate; the primary failed to retain 90\% of full-scope recovery on confirmation, while the sixteen-head set retained most of it. This is not an exhaustive or minimal-circuit search. On the same 64-pair cohort, raw and full-J accuracies are 0/64 and 62/64. The complete 256-head, three-call patch gives rescue/damage accuracies 62/64 and 0/64; sixteen heads give 61/64 and 0/64; ten give 50/64 and 5/64; eight give 44/64 and 12/64. These are first-answer-token counts. Complete-name generation on the raw/J/full-scope/ten-head arms gives the same counts; it was not collected for the sixteen-head mediation arm. Sixteen-head semantic and restoration arms below do record complete names.

The ten-head localization controls restore no accuracy when importing patterns from an unrelated graph or from the wrong call, nor when restricting the selected heads to calls 2--3. Matched ten-head neighbors also give 0/64 rescue and 62/64 after reverse replacement. These controls concern the ten-head search candidate and are not relabeled as sixteen-head controls. For the fixed sixteen-head semantic and restoration tests, new layer-matched disjoint sixteen-head controls are evaluated directly.

\paragraph{Semantic pair construction and data separation.}
The semantic extension samples 64 new source/base graph pairs (seed 2026092901), excluding all 544 graph identities used in preceding exploration, semantic/restoration panels, and localization confirmation. Each graph is a ten-person single cycle. From the source start, the first seven edges match across graphs but the eighth recipient differs. The base question starts elsewhere. Base answer, source answer, and the answer obtained by following the source path on the base graph are distinct by construction; no pairs are filtered by model success. Names occupy aligned token slots. A natural reference run uses the base graph with the source start. This identifies the predicted answer without establishing that the model literally executes eight successive transfers: cycle structure also permits alternative algorithms.

\begin{table}[!htbp]\centering\small
\caption{Ouro semantic extension, all 64 pairs retained. Counts use first-answer-token predictions. ``Route + base'' means the answer obtained by following the source path on the base graph.}
\begin{tabular}{@{}lrrrr@{}}\toprule
Condition & Base answer & Source answer & Route + base & Other\\\midrule
'''
labels={'base':'Base run','source':'Source run','counterfactual':'Base graph, source start','selected_source_pattern':'16-head pattern','selected_source_output':'16-head output','selected_source_value':'16-head values','neighbor_source_pattern':'Matched-neighbor pattern','neighbor_source_output':'Matched-neighbor output','selected_wrong_call_pattern':'16-head wrong-call pattern','allcalls_source_pattern':'16-head pattern, calls 2--4','allcalls_source_output':'16-head output, calls 2--4','full_source_pattern':'256-head pattern, calls 2--4','full_source_output':'256-head output, calls 2--4'}
for k,label in labels.items():
 a=S[k];t+=label+' & '+' & '.join(str(a[x]) for x in ['base','source','counterfactual','other'])+r' \\'+'\n'
t+=r'''\bottomrule\end{tabular}\end{table}
'''
for k in ['selected_source_pattern','selected_source_output']:
 a=S[k];t+=f"Complete-name matches for {labels[k].lower()} are {a['generation_counts'][2]}/64 for the source path on the base graph and {a['generation_counts'][1]}/64 for the source answer. "
t+=r'''

\paragraph{Restoration at the same positions.}
A separate 64-pair set (seed 2026092701), disjoint from localization data, supplies same-graph alternative-start queries. This set was previously used for a different L34.H5 restoration experiment, so it is not presented as a newly sampled cohort. Replace all Q vectors in layers 41, 43, and 47 at call 4, retaining receiving K and V; then restore clean outputs at the selected sixteen heads, at sixteen disjoint neighbors matched per layer, or at all 48 heads. Conditional recovery uses only originally correct cases broken by this corruption, without selecting on restoration success.

\begin{table}[!htbp]\centering\small
\caption{Same-site Ouro restoration. The last column conditions on cases broken by query replacement.}
\begin{tabular}{@{}lrr@{}}\toprule
Condition & Correct /64 & Recovered / broken\\\midrule
'''
for k,label in [('base','Clean'),('corrupt','Replaced queries'),('restore_selected','Selected 16 heads'),('restore_neighbor','Matched 16 neighbors'),('restore_layer41','Selected heads in L41'),('restore_layer43','Selected heads in L43'),('restore_layer47','Selected heads in L47'),('restore_all','All 48 affected heads')]:
 a=R[k];rec='---' if k=='base' else f"{a['recovered']}/{a['broken_n']}";t+=f"{label} & {a['correct']} & {rec} \\\\\n"
t+=r'''\bottomrule\end{tabular}\end{table}

An earlier L34.H5 experiment corrupts Q at that single layer across calls 2--4. H5 restores 54/55 broken cases, versus at most 3/55 for another individual head. It is a distinct local repair result, not evidence that H5 mediates the complete J effect or belongs to the compact sixteen-head set.

\paragraph{Numerical and statistical checks.}
The semantic and restoration extensions preserve exact model/map hashes and unchanged parameter versions. Self-output patches and full-output restoration have zero maximum logit error. Explicit pattern reconstruction differs slightly from BF16 SDPA (self-pattern maximum error 0.125 in the earlier tests), so it is checked for prediction preservation rather than asserted bitwise identical. Primary comparisons use all 64 cases; figures report Wilson intervals for Ouro and separate fit dots for Graph A. Paired bootstrap summaries and exact paired tests are archived; task samples are not independent model replicas. All recorded full-name generations must agree with the corresponding forward first token, and no truncated generations are silently counted as correct. Training-set graph identity separation is not established for the historical Ouro checkpoint.

'''
(P/'ouro_appendix.tex').write_text(t)

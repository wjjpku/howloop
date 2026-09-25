from pathlib import Path
import shutil
R=Path('/Users/jiaju/Documents/looped_transformer_paper_repro_20260920/paper_spotlight_v1');P=Path(__file__).resolve().parent;Q=R/'revisions/v64_target_pca';Q.mkdir(exist_ok=True)
shutil.copy2(P/'pca_B_transparent.pdf',R/'figures/v64/target_pca.pdf')
(Q/'panels.tex').write_text(r'''\setcounter{subfigure}{0}%
\begin{subfigure}[t]{.60\linewidth}
\textbf{a}\par\vspace{2pt}
\includegraphics[width=\linewidth]{figures/approved_three/target_control.pdf}
\phantomcaption\label{fig:target-behavior}
\end{subfigure}\hfill
\begin{subfigure}[t]{.38\linewidth}
\textbf{b}\par\vspace{2pt}
\includegraphics[width=\linewidth]{figures/v64/target_pca.pdf}
\phantomcaption\label{fig:target-pca}
\end{subfigure}
''')
p=R/'main_v64_subfigures.tex';s=p.read_text();s=s.replace('Figure~\\ref{fig:target} groups the resulting answers','Figure~\\ref{fig:target}a groups the resulting answers')
s=s.replace('\\includegraphics[width=0.82\\linewidth]{figures/approved_three/target_control.pdf}','\\input{revisions/v64_target_pca/panels.tex}')
a=s.index('\\caption{\\textbf{Changing the state selects the next target.}');b=s.index('\\label{fig:target}',a)
s=s[:a]+r'''\caption{\textbf{Different steering maps separate states and select different next targets.} (a) Answers after one frozen loop from the same N10 Graph A state $h_6$: $F(h_6)$, $F(J_{\mathrm{one}}(h_6))$, or $F(J_{\mathrm{two}}(h_6))$. Categories are relative to the requested endpoint $u=f_G^8(s)$. All rows use the same 4,110 examples with distinct target labels, without success filtering; steered rows average two map fits. (b) PCA of $J_{\mathrm{one}}(h_6)$ and $J_{\mathrm{two}}(h_6)$ before $F$, averaged over all token positions. Fit 1 is shown. Axes are fitted on 256 graphs and display all ten starts of another 256 graphs, with no outcome or target-collision filtering. The two colors use the same graph/start pairs. This is a separate evaluation cohort from (a); details and replication are in Appendix~\ref{app:target-pca}.}
'''+s[b:]
a=s.index('This establishes the behavior that the mechanism experiments must explain.')
s=s[:a]+r'''The two maps also produce visibly different states before the next loop runs (Fig.~\ref{fig:target}b). Averaging hidden states over token positions reveals two separated groups, replicated by the second pair of map fits. The separation is dominated by a shared shift between the maps, while the answer token's first two principal components overlap. Thus the geometry identifies a consistent difference between the controlled states; it does not by itself identify a causal direction for choosing one versus two hops. We turn to activation patching to explain how the state change affects the next computation.

'''+s[a:]
a=s.index('\\label{app:main-config}')+len('\\label{app:main-config}')
s=s[:a]+r'''

\subsection{Geometry of one-hop and two-hop steered states}
\label{app:target-pca}
We reuse the frozen N10 Graph A checkpoint and both pairs of rank-48 maps. The PCA cohort contains 512 reserved, training-excluded graphs and all ten starting positions per graph (5,120 paired examples), without filtering by prediction or target collisions. We split by graph into 256 fitting and 256 display/test graphs. For each state, we average the 35 token vectors before final normalization; PCA subtracts the training mean without feature standardization. Figure~\ref{fig:target}b fits one common projection to both map conditions after $J$ and before $F$. The first two components explain 98.6\% of variance for fit 1 and 98.8\% for fit 2. A linear classifier on these components distinguishes the two conditions on every test example for both fits. This measures steering-condition classification, not answer accuracy.

The mean paired shift accounts for 99.7\% of the squared difference between token-averaged one-hop and two-hop states. In contrast, answer-token PCA yields only 54.6--55.9\% condition-classification accuracy in its first two components, despite 99.92--99.98\% accuracy using all 256 dimensions. Across map fits, the full-state classifier transfers at 98.75--99.39\% after $J$. The separation therefore reflects a reproducible difference between maps, but its prominence depends on the representation being projected. No intervention on a principal component is tested here.

'''+s[a:]
p.write_text(s)
for n in ['summary.json','manifest.json','REPORT.md','analyze.py','draw_fig3_preview.py','extract.py']:shutil.copy2(P/n,Q/n)

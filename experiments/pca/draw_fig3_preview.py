from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;z=np.load(P/'states.npz');ids=np.random.default_rng(20260924).permutation(512);tr=np.isin(z['graph_id'],ids[:256]);te=~tr
a=z['fit1_hop1_J_mean'];b=z['fit1_hop2_J_mean'];pc=PCA(n_components=2,svd_solver='full').fit(np.concatenate([a[tr],b[tr]]));xy=[pc.transform(a[te]),pc.transform(b[te])]
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':14,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#A5AEB7','text.color':'#344350','axes.labelcolor':'#344350','pdf.fonttype':42})
f,ax=plt.subplots(figsize=(4.5,3.0));f.subplots_adjust(left=.18,right=.98,top=.94,bottom=.25)
for v,c,lab in zip(xy,['#7E99F4','#CC7C71'],['One-hop J','Two-hop J']):
 ax.scatter(v[:,0],v[:,1],s=6,alpha=.20,c=c,edgecolors='none',rasterized=True)
 ax.text(np.median(v[:,0]),1.47,lab,ha='center',va='bottom',color=c,fontsize=15,fontweight='medium')
ax.set(xlim=(-12,12),ylim=(-2.25,1.9),xticks=[-10,0,10],yticks=[-2,-1,0,1],xlabel=f'PC1 ({pc.explained_variance_ratio_[0]:.1%})',ylabel=f'PC2 ({pc.explained_variance_ratio_[1]:.1%})');ax.tick_params(labelsize=12)
for ext in ['pdf','png']:f.savefig(P/f'pca_B_transparent.{ext}',transparent=True,dpi=220)
plt.close(f)
(P/'fig3_pca_preview.tex').write_text(r'''\documentclass{article}
\usepackage[paperwidth=12in,paperheight=3.7in,margin=.15in]{geometry}
\usepackage{graphicx,subcaption}
\pagestyle{empty}
\begin{document}
\noindent
\begin{minipage}[t]{.60\linewidth}
\textbf{a}\par\vspace{2pt}
\includegraphics[width=\linewidth]{/Users/jiaju/Documents/looped_transformer_paper_repro_20260920/paper_spotlight_v1/figures/approved_three/target_control.pdf}
\end{minipage}\hfill
\begin{minipage}[t]{.38\linewidth}
\textbf{b}\par\vspace{2pt}
\includegraphics[width=\linewidth]{pca_B_transparent.pdf}
\end{minipage}
\end{document}
''')

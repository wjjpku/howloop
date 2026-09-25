from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("PAPEREXPERIMENT_OUTPUT", ROOT / "outputs")) / "figures"
OUT.mkdir(parents=True, exist_ok=True)
from pathlib import Path
import json,numpy as np
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=ROOT;out=OUT
z=np.load(ROOT/'data/plot/pca_means.npz');ids=np.random.default_rng(20260924).permutation(512);tr=np.isin(z['graph_id'],ids[:256]);te=~tr
report={}
for fit in [1,2]:
 arrays=[z[f'fit{fit}_hop1_J_mean'],z[f'fit{fit}_hop2_J_mean']]
 pc=PCA(n_components=2,svd_solver='full').fit(np.concatenate([a[tr] for a in arrays]));xy=[pc.transform(a[te]) for a in arrays]
 report[str(fit)]=pc.explained_variance_ratio_.tolist()
 if fit!=1:continue
 plt.rcParams.update({'font.family':'Arial','font.size':14,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#A5AEB7','pdf.fonttype':42})
 fig,ax=plt.subplots(figsize=(4.5,3.0));fig.subplots_adjust(left=.19,right=.98,top=.80,bottom=.24)
 for v,c,lab in zip(xy,['#287EAD','#D46B46'],['$J_{\\mathrm{one}}$','$J_{\\mathrm{two}}$']):
  ax.scatter(v[:,0],v[:,1],s=7,alpha=.28,c=c,edgecolors='none',rasterized=True,label=lab)
 ax.margins(x=.25,y=.30)
 ax.set_xlabel(f'PC1 ({pc.explained_variance_ratio_[0]:.1%})');ax.set_ylabel(f'PC2 ({pc.explained_variance_ratio_[1]:.1%})');ax.tick_params(labelsize=12)
 leg=ax.legend(loc='lower center',bbox_to_anchor=(.5,1.04),ncol=2,frameon=False,handlelength=.6,handletextpad=.3,columnspacing=.9,fontsize=13,markerscale=2)
 for h in leg.legend_handles:h.set_alpha(1)
 dest=OUT/'fig3b_pca.pdf';fig.savefig(dest,transparent=True,dpi=240);fig.savefig(out/'pca_preview.png',dpi=200);plt.close(fig)
(out/'pca_variance.json').write_text(json.dumps(report,indent=2))

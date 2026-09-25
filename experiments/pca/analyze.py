from pathlib import Path
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,silhouette_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;z=np.load(P/'states.npz');rng=np.random.default_rng(20260924);ids=rng.permutation(512);train=np.isin(z['graph_id'],ids[:256]);test=~train;blue='#7E99F4';red='#CC7C71'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#A5AEB7','text.color':'#344350','axes.labelcolor':'#344350'})
report={'protocol':{'graphs':512,'examples':5120,'train_graphs':256,'test_graphs':256,'split_seed':20260924,'PCA':'uncentered feature scaling disabled; training mean subtracted; fitted on training graphs only','probe':'logistic regression; graph-disjoint test; no success filtering'},'results':[],'behavior':{}}
for fit in [1,2]:
 for hop in [1,2]:
  for stage in ['J','JF']:
   report['behavior'][f'fit{fit}_hop{hop}_{stage}']=float(np.mean(z[f'fit{fit}_hop{hop}_{stage}_pred']==z[f'target{hop}']))
 for rep in ['answer','mean']:
  f,axs=plt.subplots(1,3,figsize=(12,3.7));f.subplots_adjust(left=.055,right=.985,bottom=.24,top=.90,wspace=.32)
  for ix,stage in enumerate(['raw','J','JF']):
   ax=axs[ix];ax.text(-.15,1.07,'abc'[ix],transform=ax.transAxes,fontweight='bold',fontsize=14)
   if stage=='raw':
    a=z['raw_'+rep];pc=PCA(n_components=2,svd_solver='full').fit(a[train]);u=pc.transform(a[test]);ax.scatter(u[:,0],u[:,1],s=5,alpha=.25,c='#A5AEB7',rasterized=True)
   else:
    a=z[f'fit{fit}_hop1_{stage}_{rep}'];b=z[f'fit{fit}_hop2_{stage}_{rep}'];xtr=np.concatenate([a[train],b[train]]);xte=np.concatenate([a[test],b[test]]);ytr=np.repeat([0,1],train.sum());yte=np.repeat([0,1],test.sum());pc=PCA(n_components=2,svd_solver='full').fit(xtr);u=pc.transform(xte)
    probes={}
    for name,tr,te in [('PC1_PC2',pc.transform(xtr),u),('full_hidden',xtr,xte)]:
     model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=3000));model.fit(tr,ytr);probes[name]=float(model.score(te,yte))
     if name=='full_hidden':
      other=3-fit;cross=np.concatenate([z[f'fit{other}_hop1_{stage}_{rep}'][test],z[f'fit{other}_hop2_{stage}_{rep}'][test]]);probes['other_fit_full_hidden']=float(model.score(cross,yte))
    for k,c in [(0,blue),(1,red)]:ax.scatter(u[yte==k,0],u[yte==k,1],s=5,alpha=.20,c=c,rasterized=True)
    # Separability after removing class means within each target label: labels themselves balanced by permutation graphs/all starts.
    row={'fit':fit,'representation':rep,'stage':stage,'explained_variance_2PC':float(pc.explained_variance_ratio_.sum()),'probe_accuracy':probes,'silhouette_2PC':float(silhouette_score(u,yte,sample_size=2000,random_state=42))};report['results'].append(row)
   ax.set_xlabel(f'PC1 ({pc.explained_variance_ratio_[0]*100:.1f}%)');ax.set_ylabel(f'PC2 ({pc.explained_variance_ratio_[1]*100:.1f}%)');ax.tick_params(labelsize=8)
  handles=[plt.Line2D([],[],marker='o',linestyle='',color=c,label=l) for c,l in [('#A5AEB7','Same initial state'),(blue,'One-hop J'),(red,'Two-hop J')]]
  f.legend(handles=handles,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.005));f.savefig(P/f'pca_fit{fit}_{rep}.png',dpi=200);plt.close(f)
(P/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

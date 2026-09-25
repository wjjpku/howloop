from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("PAPEREXPERIMENT_OUTPUT", ROOT / "outputs")) / "figures"
OUT.mkdir(parents=True, exist_ok=True)
from pathlib import Path
import json,csv,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
R=ROOT/'experiments/n10/figure_work';O=OUT
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':220})
colors=['#9BA5B1','#287EAD','#D46B46','#DDCDAA']; categories=['Stay','One hop','Two hops','Others']; keys=['endpoint','one','two','other']
rows=[];distributions={}; ns={}
for name in 'ABCDE':
 raw=[]; fits={1:[],2:[]}
 for hop in [1,2]:
  for fit in [1,2]:
   p=R/'source/local'/name/f'hop{hop}_seed{fit}/evaluation/aggregate.csv'
   data=list(csv.DictReader(p.open()))
   for mode in ['raw','full']:
    x=next(x for x in data if x['mode']==mode and x['readout']=='post_executor');n=int(x['distinct_examples']);counts=np.array([int(x[k]) for k in keys]);assert counts.sum()==n
    values=counts/n
    if mode=='raw':raw.append(values)
    else:fits[hop].append(values)
    rows.append(dict(backbone=name,hop=hop,fit=fit,mode=mode,n=n,**dict(zip(keys,counts.tolist()))))
   ns[name]=n
 assert all(np.array_equal(raw[0],x) for x in raw)
 distributions[name]=np.stack([raw[0],np.mean(fits[1],0),np.mean(fits[2],0)])*100
with (OUT/'control_counts.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def bars(ax,values,title=None,large=False):
 for i,row in enumerate(values):
  left=0
  for k,v in enumerate(row):
   ax.barh(i,v,left=left,height=.58,color=colors[k],edgecolor='white',linewidth=.6)
   if v>=(7 if large else 20):ax.text(left+v/2,i,f'{v:.1f}%',ha='center',va='center',fontsize=12 if large else 9,color='white' if k in [1,2] else '#253345',fontweight='medium')
   left+=v
 ax.set_yticks([0,1,2],['No $J$','$J_{\\mathrm{one}}$','$J_{\\mathrm{two}}$']);ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xticks([0,25,50,75,100],['0','25','50','75','100%']);ax.set_xlabel('Post-$F$ output distribution');ax.tick_params(axis='y',length=0,pad=10);ax.spines['left'].set_visible(False);ax.spines['bottom'].set_color('#C7CDD4');ax.tick_params(axis='x',color='#C7CDD4')
 if title:ax.set_title(title,loc='left',fontweight='bold',pad=12)
def save(fig,name):
 for ext in ['pdf','png','svg']:fig.savefig(O/f'{name}.{ext}',bbox_inches='tight',facecolor='white')
 plt.close(fig)
fig,ax=plt.subplots(figsize=(8.0,2.7));bars(ax,distributions['A'],large=True)
fig.legend(handles=[Patch(facecolor=c,label=l) for c,l in zip(colors,categories)],loc='upper center',bbox_to_anchor=(.55,1.08),ncol=4,frameon=False,handlelength=1.4,columnspacing=1.8)
fig.subplots_adjust(left=.15,right=.98,top=.88,bottom=.21);save(fig,'fig3a_target_accuracy')

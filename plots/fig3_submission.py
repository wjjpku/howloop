"""Submitted Fig. 3: per-example target and two-continuation distributions."""
from pathlib import Path
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get('PAPEREXPERIMENT_OUTPUT', ROOT/'outputs'))/'figures'
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT/'experiments/composition'
ds = [np.load(DATA/f'A_fit{fit}.npz') for fit in (1, 2)]
labels = ds[0]['labels']
assert np.array_equal(labels, ds[1]['labels'])
mask = np.all(np.diff(np.sort(labels, axis=1), axis=1) != 0, axis=1)
assert mask.sum() == 3200
colors = ['#9BA5B1', '#287EAD', '#D46B46', '#DDCDAA']
names = ['Stay', 'One hop', 'Two hops', 'Others']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'mathtext.fontset':'cm','pdf.fonttype':42})

def draw(name, rows):
    fig,ax=plt.subplots(figsize=(2.8,1.75));fig.subplots_adjust(left=.18,right=.96,bottom=.27,top=.97)
    left=np.zeros(3)
    for j,col in enumerate(colors):
        ax.barh(range(3),rows[:,j],left=left,color=col,height=.61,edgecolor='white',lw=.35)
        for i,v in enumerate(rows[:,j]):
            if v>=12:ax.text(left[i]+v/2,i,f'{v:.1f}',ha='center',va='center',fontsize=9,color='white' if j in (1,2) else '#344350')
        left+=rows[:,j]
    ax.set(xlim=(0,100),xticks=[0,50,100],yticks=range(3),yticklabels=['No $J$',r'$J_{\mathrm{one}}$',r'$J_{\mathrm{two}}$'],xlabel='Answer distribution (%)')
    ax.invert_yaxis();ax.tick_params(length=0,pad=4,labelsize=9);ax.spines[['top','right','left']].set_visible(False)
    fig.savefig(OUT/f'{name}.pdf');plt.close(fig)

# Panel a uses its own 4,110-example cohort; do not replace it with the
# stricter 3,200-example composition cohort. fig3a.py generates the source.
import runpy
runpy.run_path(str(ROOT/'plots/fig3a.py'),run_name='__main__')
(OUT/'fig03a_target_accuracy.pdf').write_bytes((OUT/'fig3a_target_accuracy.pdf').read_bytes())

ledger={}
for first,hop,suffix in [('one',1,'b_composition_one'),('two',2,'c_composition_two')]:
    rows=[]
    for second in ('raw','one','two'):
        idx=list(ds[0]['sequences']).index(first+'_'+second)
        values=[np.mean([(d['second'][mask,idx]==labels[mask,hop+k]).mean() for d in ds])*100 for k in range(3)]
        values.append(100-sum(values));rows.append(values)
    rows=np.array(rows);ledger[first]=rows.tolist();draw('fig03'+suffix,rows)

fig=plt.figure(figsize=(7.4,.32))
fig.legend(handles=[Patch(facecolor=c,label=l) for c,l in zip(colors,names)],loc='center',ncol=4,frameon=False,fontsize=12,handlelength=1.4,columnspacing=2)
fig.savefig(OUT/'fig03_legend.pdf');plt.close(fig)
(OUT/'fig03_values.json').write_text(json.dumps({'n_composition':int(mask.sum()),'distributions_percent':ledger},indent=2)+'\n')

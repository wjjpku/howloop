from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("PAPEREXPERIMENT_OUTPUT", ROOT / "outputs")) / "figures"
OUT.mkdir(parents=True, exist_ok=True)
from pathlib import Path
import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=ROOT;O=OUT
v=json.loads((ROOT/'data/plot/ouro_readouts.json').read_text());audit=json.loads((ROOT/'data/plot/ouro_fixed4.json').read_text())
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'font.size':12,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
f=plt.figure(figsize=(8.2,2.85))
axs=[f.add_axes(r) for r in [[.065,.25,.23,.60],[.335,.25,.23,.60],[.70,.25,.285,.60]]]
for ax,z,title in zip(axs[:2],[v['stepwise'],v['final_only']],['(a) Stepwise','(b) Final-only']):
 z=np.array(z);ax.imshow(z,origin='lower',vmin=0,vmax=100,cmap='viridis',aspect='auto',extent=(.5,4.5,.5,4.5))
 ax.set(xticks=[1,2,3,4],yticks=[1,2,3,4],xlabel='Loop');ax.set_title(title,pad=9)
 for k in range(4):
  for t in range(4):ax.text(t+1,k+1,f'{z[k,t]:.0f}',ha='center',va='center',fontsize=11,color='black' if z[k,t]>60 else 'white')
axs[0].set_ylabel('Deletion depth')
a=axs[2];xx=np.arange(4);w=.32
for key,col,off,lab in [('final_only','#6E8FD0',-w/2,'Final-only + $J$'),('stepwise','#C97968',w/2,'Stepwise + $J$')]:
 y=[100*audit[key]['counts'][f't0_s500_k{k}']['correct']/64 for k in range(5,9)];a.bar(xx+off,y,w,color=col,label=lab,zorder=3)
 for x,val in zip(xx+off,y):a.text(x,val+2,f'{val:.1f}',ha='center',fontsize=9.5)
a.set(xticks=xx,xticklabels=['5','6','7','8'],ylim=(0,117),yticks=[0,50,100],xlabel='Requested deletion depth',ylabel='Accuracy (%)');a.set_title('(c) Four-loop steering',pad=9);a.grid(axis='y',alpha=.2);a.set_axisbelow(True)
a.legend(loc='upper center',bbox_to_anchor=(.5,-.25),ncol=2,frameon=False,fontsize=10,handlelength=1,columnspacing=1)
for ext in ['pdf','png']:f.savefig(O/f'fig07_supervision_capacity.{ext}',dpi=200,bbox_inches='tight',pad_inches=.04)

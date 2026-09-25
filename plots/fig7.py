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
v=json.loads((ROOT/'data/plot/ouro_readouts.json').read_text());data=json.loads((ROOT/'data/plot/capacity.json').read_text());audit=json.loads((ROOT/'data/plot/ouro_fixed4.json').read_text())
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'font.size':12,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
f=plt.figure(figsize=(10.8,3.25))
axs=[f.add_axes(r) for r in [[.045,.23,.15,.65],[.225,.23,.15,.65],[.45,.23,.22,.65],[.795,.23,.165,.65]]]
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
b=axs[3];z=np.array(list(data['KG_128x16'].values()));im=b.imshow(z,vmin=0,vmax=100,cmap='Blues',aspect='auto');b.set(xticks=range(4),xticklabels=data['lengths'],yticks=range(4),yticklabels=['Rank-48 affine','Dense affine','MLP','Attention'],xlabel='Composition length');b.set_title('(d) Map capacity',pad=9);b.tick_params(length=0)
for i in range(4):
 for j in range(4):b.text(j,i,f'{z[i,j]:.1f}',ha='center',va='center',fontsize=10.5,color='white' if z[i,j]>60 else '#293946')
cb=f.colorbar(im,cax=f.add_axes([.972,.23,.008,.65]),ticks=[0,50,100]);cb.ax.set_title('%',fontsize=11)
for ext in ['pdf','png']:f.savefig(O/f'fig7_supervision_capacity.{ext}',dpi=180,bbox_inches='tight',pad_inches=.04)

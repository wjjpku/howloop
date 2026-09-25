from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
R=Path(__file__).resolve().parent
rows=json.loads((R/'interval_values.json').read_text())
order=[4,5,6,0,1,2,3,7,8,9,10]
rows=[rows[i] for i in order]
plt.rcParams.update({'font.family':'Arial','font.size':10,'pdf.fonttype':42})
f,ax=plt.subplots(figsize=(10.5,4.1));f.subplots_adjust(left=.065,right=.99,bottom=.30,top=.85)
xs=np.array([0,.8,1.6,2.7,3.5,4.3,5.1,6.2,7,7.8,8.6]);colors=['#5C92B7','#CF8F94']
for i,(x,row) in enumerate(zip(xs,rows)):
 for j,key in enumerate(['graph','ouro']):
  es=row[key];mean=np.mean([e['mean'] for e in es])*100;xx=x+[-.13,.13][j]
  ax.bar(xx,mean,width=.22,color=colors[j],alpha=.33,edgecolor=colors[j],linewidth=.7,zorder=2)
  for k,e in enumerate(es):
   xp=xx+([-.037,.037][k] if len(es)>1 else 0);v=e['mean']*100;lo,hi=np.array(e['ci95'])*100
   ax.errorbar(xp,v,yerr=[[max(0,v-lo)],[max(0,hi-v)]],fmt='o' if k==0 else 'D',markersize=3.4,markerfacecolor='white',markeredgecolor=colors[j],markeredgewidth=.85,ecolor=colors[j],elinewidth=.85,capsize=2,zorder=4)
ax.set_ylim(-1.5,104);ax.set_xlim(-.4,9.0);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel('Accuracy (%)')
labels=['Pattern → raw-graph answer','Output → raw-graph answer','Pattern → corrupted answer','Output → corrupted answer','Restore selected heads','Restore control heads','Restore all heads','Raw execution','Full J steering','J patterns → raw','Raw patterns → J']
labels=[labels[i] for i in order]
ax.set_xticks(xs,labels,rotation=35,ha='right',rotation_mode='anchor',fontsize=9)
ax.spines[['top','right']].set_visible(False);ax.spines[['left','bottom']].set_color('#6E7880');ax.tick_params(length=3,color='#6E7880');ax.grid(axis='y',color='#E8EBEE',lw=.65);ax.set_axisbelow(True)
for x in [2.15,5.65]:ax.axvline(x,color='#CDD4DB',lw=.9,ls='--')
for left,right,title,subtitle in [(-.3,1.9,'a  Restore disrupted reads','Broken cases repaired'),(2.4,5.4,'b  Route vs. retrieved content','Answer frequency'),(5.9,8.9,'c  Transfer the steering effect','Answer accuracy')]:
 ax.text((left+right)/2,1.08,title,transform=ax.get_xaxis_transform(),ha='center',fontsize=11,fontweight='bold',color='#273846')
f.legend(handles=[Patch(facecolor=c,alpha=.5,label=n) for c,n in zip(colors,['N10 A · seed 6','N10 C · seed 4'])],loc='upper center',bbox_to_anchor=(.52,1.06),ncol=2,frameon=False,fontsize=10)
for ext in ['png','pdf','svg']:f.savefig(R/f'selected_backbones.{ext}',dpi=240,bbox_inches='tight',pad_inches=.10,facecolor='white')

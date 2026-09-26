import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1];d=json.loads((P/'summary.json').read_text())['conditions']
keys=['native','J','selected_rescue_pattern','selected_damage_pattern','selected_rescue_value','selected_unrelated_pattern','selected_wrong_call_pattern','neighbor_rescue_pattern']
labels=['Native','Full J','J pattern → native','Native pattern → J','J value → native','Unrelated pattern → native','Wrong-call pattern → native','Neighbor-head pattern → native']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(7.3,4.3));vals=np.array([d[k]['accuracy'] for k in keys])*100;ci=np.array([d[k]['wilson95'] for k in keys])*100
for i,k in enumerate(keys):
 c='#276984' if i in [1,2] else '#b77b55' if i==3 else '#8a959c';ax.errorbar(vals[i],i,xerr=[[vals[i]-ci[i,0]],[ci[i,1]-vals[i]]],fmt='o',color=c,capsize=3,markersize=6);ax.text(104,i,f"{d[k]['correct']}/256",va='center',fontsize=9,color=c)
ax.set_xlim(-3,127);ax.set_xticks([0,25,50,75,100]);ax.set_yticks(range(len(keys)),labels);ax.invert_yaxis();ax.set_xlabel('Accuracy (%) · Wilson 95% CI');ax.grid(axis='x',alpha=.13);ax.set_title('Fixed Ouro heads on 256 new graph pairs',loc='left',fontweight='bold',pad=16)
fig.subplots_adjust(left=.36,right=.98,top=.88,bottom=.21);fig.text(.36,.045,'16 fixed heads · fourth recurrent call · no training or reselection\nComplete-name and first-token counts are reported separately in the report.',fontsize=8,color='#555')
for ext in ['pdf','png']:fig.savefig(P/f'ouro_confirmation256.{ext}',dpi=200,bbox_inches='tight')

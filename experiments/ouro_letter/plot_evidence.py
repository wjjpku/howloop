import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
s=json.loads((P/'semantic_confirmation64/analysis.json').read_text())['conditions'];r=json.loads((P/'restore_confirmation64/analysis.json').read_text())['conditions'];m=json.loads((P/'mediation_confirmation64/analysis.json').read_text())['conditions']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
blue='#1689D4';yellow='#D8ED16';gray='#AAB4BF'
f,ax=plt.subplots(1,3,figsize=(15,4.3),gridspec_kw={'width_ratios':[1,1,1.3]})
rows=[]
def bars(a,x,vals,ns,colors,labels,intervals):
 b=a.bar(x,np.array(vals)/np.array(ns)*100,color=colors,edgecolor='#4E5965',linewidth=.5,width=.66)
 for z,k,n,ci,l in zip(b,vals,ns,intervals,labels):
  y=k/n*100;lo,hi=np.array(ci)*100;a.errorbar(z.get_x()+z.get_width()/2,y,yerr=[[y-lo],[hi-y]],fmt='none',color='#354052',capsize=3,lw=1)
  a.text(z.get_x()+z.get_width()/2,max(y,hi)+3,f'{k}/{n}',ha='center',fontsize=9)
  rows.append([l,k,n,k/n,ci[0],ci[1]])
 a.set_ylim(0,117);a.set_yticks([0,25,50,75,100]);a.grid(axis='y',alpha=.16);a.set_axisbelow(True)
vals=[s['late_source_pattern']['counterfactual'],s['late_source_output']['counterfactual'],s['late_source_pattern']['source'],s['late_source_output']['source']]
cis=[s['late_source_pattern']['wilson_ci95'][2],s['late_source_output']['wilson_ci95'][2],s['late_source_pattern']['wilson_ci95'][1],s['late_source_output']['wilson_ci95'][1]]
bars(ax[0],[0,1,2.5,3.5],vals,[64]*4,[blue,yellow]*2,['pattern to CF','output to CF','pattern to source','output to source'],cis)
ax[0].set_xticks([.5,3],['Base-graph\ncounterfactual','Source\nanswer']);ax[0].set_ylabel('Answer match (%)');ax[0].legend([plt.Rectangle((0,0),1,1,color=blue),plt.Rectangle((0,0),1,1,color=yellow)],['Pattern patch','Output patch'],loc='upper center',bbox_to_anchor=(.5,-.21),ncol=2,frameon=False)
keys=['restore_5','restore_0','restore_all'];bars(ax[1],np.arange(3),[r[k]['recovered'] for k in keys],[55]*3,[blue,gray,yellow],keys,[r[k]['conditional_wilson_ci95'] for k in keys]);ax[1].set_xticks(range(3),['H5','Best other\nhead (H0)','All heads']);ax[1].set_ylabel('Recovery of 55 corrupted cases (%)')
keys=['native','J','late_rescue_pattern','late_rescue_value','late_damage_pattern'];bars(ax[2],np.arange(5),[m[k]['correct'] for k in keys],[64]*5,[gray,yellow,blue,gray,blue],keys,[m[k]['wilson_ci95'] for k in keys]);ax[2].set_xticks(range(5),['Raw','With J','Pattern\nrescue','Value\nrescue','Pattern\ndamage']);ax[2].set_ylabel('Accuracy (%)')
for a,label in zip(ax,['a','b','c']):a.text(-.13,1.02,label,transform=a.transAxes,fontsize=16,fontweight='bold')
f.subplots_adjust(left=.055,right=.99,bottom=.27,top=.91,wspace=.36)
f.savefig(P/'ouro_mechanism_evidence.pdf');f.savefig(P/'ouro_mechanism_evidence.png',dpi=170)
with (P/'figure_data.csv').open('w') as out:w=csv.writer(out);w.writerow(['condition','count','n','rate','wilson_low','wilson_high']);w.writerows(rows)

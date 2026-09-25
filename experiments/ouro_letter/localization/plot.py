import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;D=json.loads((P/'localize_confirmation64/analysis.json').read_text())['conditions'];cfg=json.loads((P/'confirmation.json').read_text());rank=json.loads((P/'compact.json').read_text())['ranking'];sites16=[[l,h] for _,l,h in rank[:16]];blue='#1689D4';yellow='#D8ED16'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
f,axs=plt.subplots(1,3,figsize=(15,4.0),gridspec_kw={'width_ratios':[1.05,1.1,1.1]});a=axs[0]
for y,l in enumerate([41,43,47]):
 for h in range(16):a.scatter(h,y,s=100,color=blue if [l,h] in sites16 else '#E3E8EF',marker='s',edgecolor='white',linewidth=.5)
a.set_yticks(range(3),['L41','L43','L47']);a.set_xticks([0,4,8,12,15]);a.set_xlabel('Head index');a.set_xlim(-.8,15.8);a.set_ylim(2.8,-.8);a.text(.5,-.24,'16 heads · call 4 only',transform=a.transAxes,ha='center',fontweight='bold');a.spines[['left','bottom']].set_visible(False);a.tick_params(length=0)
labels=['Full\n256','Compact\n16','Selected\n10','Compact\n8','Neighbor\n10'];names=['full','size16','selected','size8','neighbor'];rows=[]
for a,direction,color in zip(axs[1:],['rescue','damage'],[blue,yellow]):
 vals=[D[n+'_'+direction+'_pattern'] for n in names];x=np.arange(len(names));rates=np.array([v['accuracy']*100 for v in vals]);bars=a.bar(x,rates,color=color,edgecolor='#667281',linewidth=.5,width=.67)
 for i,v in enumerate(vals):
  lo,hi=np.array(v['wilson95'])*100;y=rates[i];a.errorbar(i,y,yerr=[[y-lo],[hi-y]],fmt='none',color='#344253',capsize=3,lw=1);a.text(i,hi+3,str(v['correct'])+'/64',ha='center',fontsize=9);rows.append([names[i],direction,v['correct'],64,*v['wilson95']])
 a.axhline(D['J']['accuracy']*100,c='#7D8895',ls='--',lw=1);a.set_ylim(0,118);a.set_yticks([0,25,50,75,100]);a.set_xticks(x,labels);a.grid(axis='y',alpha=.15);a.set_axisbelow(True);a.set_ylabel('Accuracy after '+direction+' (%)')
for a,label in zip(axs,'abc'):a.text(-.13,1.02,label,transform=a.transAxes,fontweight='bold',fontsize=16)
f.subplots_adjust(left=.05,right=.99,wspace=.35,bottom=.27,top=.9);f.savefig(P/'compact_heads.pdf');f.savefig(P/'compact_heads.png',dpi=180)
with (P/'figure_data.csv').open('w') as out:w=csv.writer(out);w.writerow(['scope','direction','correct','n','wilson_low','wilson_high']);w.writerows(rows)

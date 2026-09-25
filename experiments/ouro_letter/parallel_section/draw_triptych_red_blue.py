from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
P=Path(__file__).resolve().parent
O=Path('/Users/jiaju/Documents/looped_transformer_paper_repro_20260920/paper_spotlight_v1/figures/v62');O=O/'red_blue';O.mkdir(parents=True,exist_ok=True)
G=json.loads((P/'graph_summary.json').read_text())['results']['A']
S=json.loads((P/'parallel_confirmation64/analysis.json').read_text())['conditions']
R=json.loads((P/'parallel_restore64/analysis.json').read_text())['conditions']
M=json.loads((P.parent/'localization/localize_confirmation64/analysis.json').read_text())['conditions']
BLUE='#7E99F4'; YELLOW='#CC7C71'; INK='#243443'; GRAY='#788693'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':10,'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':GRAY,'pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.6})
f,axs=plt.subplots(1,3,figsize=(13.6,4.15),gridspec_kw={'width_ratios':[1.2,.83,1.12]})
f.subplots_adjust(left=.044,right=.994,bottom=.235,top=.76,wspace=.29)
ledger=[]
for ax,letter,title,ylabel in zip(axs,'abc',['Separate route from content','Restore selected outputs','Transfer the effect of J'],['Answer match (%)','Broken cases recovered (%)','Accuracy (%)']):
 ax.set_ylim(-1,112);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel(ylabel,labelpad=5,fontsize=10)
 ax.spines[['top','right','left']].set_visible(False);ax.spines['bottom'].set_color('#A9B4BE');ax.tick_params(axis='both',length=0,pad=6)
 ax.grid(axis='y',color='#E8EDF0',linewidth=.6);ax.set_axisbelow(True)
 ax.text(0,1.24,letter,transform=ax.transAxes,fontsize=15,fontweight='bold')
 ax.text(.065,1.25,title,transform=ax.transAxes,fontsize=12,fontweight='bold',va='baseline')
def pair(ax,x,g,o,ci,tag,annotate=True):
 for model,xx,val,color in [('Graph A',x-.18,np.mean(g)*100,BLUE),('Ouro',x+.18,o*100,YELLOW)]:
  ax.bar(xx,val,width=.31,color=color,edgecolor=INK if model=='Ouro' else BLUE,linewidth=.45,zorder=3)
  if model=='Graph A':
   ax.scatter([xx-.045,xx+.045],np.array(g)*100,s=12,facecolors='white',edgecolors=INK,linewidths=.65,zorder=5);hi=max(g)*100
  else:
   lo,hi=np.array(ci)*100;ax.errorbar(xx,val,yerr=[[val-lo],[hi-val]],fmt='none',ecolor=INK,elinewidth=.75,capsize=2,zorder=4)
  if annotate:ax.text(xx,max(hi,val)+2.3,f'{val:.1f}',ha='center',va='bottom',fontsize=8.2,color=INK)
  ledger.append({'panel':tag,'model':model,'percent':val})
a=axs[0]
xs=[0,1,2.6,3.6]
for x,(metric,idx,typ,okey) in zip(xs,[('cf',2,'pattern','pattern'),('cf',2,'context','output'),('donor',1,'pattern','pattern'),('donor',1,'context','output')]):
 z=S['selected_source_'+okey];k='counterfactual' if idx==2 else 'source'
 pair(a,x,[G[t]['cross_graph']['own_eligible'][typ+'_'+metric]['mean'] for t in ['1','2']],z[k]/64,z['wilson_ci95'][idx],'a')
a.set_xticks(xs,['Pattern','Output','Pattern','Output']);a.set_xlim(-.65,4.25)
a.text(.5,108,'Source route + base graph',ha='center',fontsize=9.2,fontweight='bold');a.text(3.1,108,'Source answer',ha='center',fontsize=9.2,fontweight='bold')
a.axvline(1.8,color='#CFD7DE',lw=.8,ymax=.91)
a.text(.5,-.22,'Patched activation',transform=a.transAxes,ha='center',fontsize=9,color=GRAY)
a=axs[1]
for x,gk,ok in [(0,'clean_context_H0','restore_selected'),(1,'clean_context_H1','restore_neighbor'),(2,'clean_context_H0123','restore_all')]:
 z=R[ok];pair(a,x,[G[t]['restoration'][gk]['mean'] for t in ['1','2']],z['conditional_recovery'],z['conditional_wilson_ci95'],'b')
a.set_xticks([0,1,2],['Selected','Control','All']);a.set_xlim(-.65,2.65)
a.text(.5,-.22,'Restored head outputs',transform=a.transAxes,ha='center',fontsize=9,color=GRAY)
a=axs[2]
for x,gk,ok in [(0,'native','native'),(1,'full_J','J'),(2.4,'rescue','size16_rescue_pattern'),(3.4,'damage','size16_damage_pattern')]:
 z=M[ok];pair(a,x,[G[t]['pattern_swap'][gk]['mean'] for t in ['1','2']],z['accuracy'],z['wilson95'],'c')
a.set_xticks([0,1,2.4,3.4],['Raw','With J','J → Raw','Raw → J']);a.set_xlim(-.65,4.05)
a.axvline(1.7,color='#CFD7DE',lw=.8,ymax=.91)
a.text(.5,108,'Unpatched',ha='center',fontsize=9.2,fontweight='bold');a.text(2.9,108,'Pattern swaps',ha='center',fontsize=9.2,fontweight='bold')
a.text(.5,-.22,'Arrows: pattern source → receiving run',transform=a.transAxes,ha='center',fontsize=9,color=GRAY)
f.legend(handles=[Patch(facecolor=BLUE,label='Graph A'),Patch(facecolor=YELLOW,edgecolor=INK,linewidth=.5,label='Ouro · fixed 16 heads')],loc='lower center',bbox_to_anchor=(.5,.018),ncol=2,frameon=False,fontsize=10.5,handlelength=1.5,columnspacing=2.3)
for ext in ['pdf','svg','png']:f.savefig(O/f'parallel_mechanism_triptych.{ext}',dpi=230,facecolor='white')
(O/'parallel_mechanism_values.json').write_text(json.dumps(ledger,indent=2))
(O/'caption.tex').write_text(r'''\textbf{Parallel interventions connect attention routing to steering in Graph A and Ouro.} (a) Pattern patches favor the answer obtained by following the source route on the base graph; head-output patches favor the source answer. (b) Restoring selected clean outputs recovers answers disrupted by query replacement, whereas control heads recover few or none. (c) Importing steered patterns into raw runs reproduces most of the steering gain; the reverse swap reduces accuracy. Arrows indicate the source and receiving run of the pattern patch; values remain in the receiving run. Graph A uses H0 in (a,b) and all four heads of the same layer in (c); Ouro uses the same fixed 16 heads at call 4 throughout. Graph bars average two controller fits, shown as white dots. Ouro error bars are 95\% Wilson intervals over task instances. Graph populations are 4,096 pairs per fit in (a), 3,574 broken transitions per fit in (b), and 3,056 label-distinct instances in (c). Ouro uses 64 pairs in (a,c), with separate cohorts, and 63 broken cases in (b). Graph controls are other individual heads; Ouro uses a layer-matched disjoint 16-head set. These interventions support shared functional roles, not identical or minimal circuits.''')
print(O)

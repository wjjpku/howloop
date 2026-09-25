from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
P=Path(__file__).resolve().parent
O=Path('/Users/jiaju/Documents/looped_transformer_paper_repro_20260920/paper_spotlight_v1/figures/v62/options');O.mkdir(exist_ok=True)
G=json.loads((P/'graph_summary.json').read_text())['results']['A'];S=json.loads((P/'parallel_confirmation64/analysis.json').read_text())['conditions'];R=json.loads((P/'parallel_restore64/analysis.json').read_text())['conditions'];M=json.loads((P.parent/'localization/localize_confirmation64/analysis.json').read_text())['conditions']
D=[[],[],[]]
for metric,idx,typ,ok in [('cf',2,'pattern','pattern'),('cf',2,'context','output'),('donor',1,'pattern','pattern'),('donor',1,'context','output')]:
 z=S['selected_source_'+ok];D[0].append(([G[t]['cross_graph']['own_eligible'][typ+'_'+metric]['mean']*100 for t in ['1','2']],100*z['counterfactual' if idx==2 else 'source']/64,np.array(z['wilson_ci95'][idx])*100))
for gk,ok in [('clean_context_H0','restore_selected'),('clean_context_H1','restore_neighbor'),('clean_context_H0123','restore_all')]:
 z=R[ok];D[1].append(([G[t]['restoration'][gk]['mean']*100 for t in ['1','2']],100*z['conditional_recovery'],np.array(z['conditional_wilson_ci95'])*100))
for gk,ok in [('native','native'),('full_J','J'),('rescue','size16_rescue_pattern'),('damage','size16_damage_pattern')]:
 z=M[ok];D[2].append(([G[t]['pattern_swap'][gk]['mean']*100 for t in ['1','2']],100*z['accuracy'],np.array(z['wilson95'])*100))
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':10,'text.color':'#35414A','axes.labelcolor':'#35414A','xtick.color':'#35414A','ytick.color':'#7D878F','pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.6})
for name,colors,style in [('01_teal_terracotta',['#317B83','#C78367'],'bar'),('02_indigo_copper',['#6677A5','#C4935E'],'point'),('03_plum_sage',['#886583','#83A397'],'horizontal')]:
 f,axs=plt.subplots(1,3,figsize=(11.8,3.3),gridspec_kw={'width_ratios':[1.2,.86,1.06]})
 f.subplots_adjust(left=.053,right=.99,bottom=.24,top=.83,wspace=.3)
 for i,ax in enumerate(axs):
  ax.spines[['top','right','left']].set_visible(False);ax.spines['bottom'].set_color('#CAD0D4');ax.tick_params(length=0,pad=6)
  ax.text(-.08,1.11,'abc'[i],transform=ax.transAxes,fontweight='bold',fontsize=15)
  ax.set_axisbelow(True)
  if style=='horizontal':
   labels=[['Pattern · base route','Output · base route','Pattern · source','Output · source'],['Selected','Control','All'],['Raw','With J','J → Raw','Raw → J']][i]
   for k,(g,o,ci) in enumerate(D[i]):
    for j,(v,col) in enumerate(zip([np.mean(g),o],colors)):
     y=k+(-.13 if j==0 else .13);ax.scatter(v,y,s=32,color=col,zorder=4)
     if j==0:ax.plot([min(g),max(g)],[y,y],color=col,lw=2)
     else:ax.plot(ci,[y,y],color=col,lw=1.2)
   ax.set_yticks(range(len(labels)),labels,fontsize=9);ax.invert_yaxis();ax.set_ylim(len(labels)-.5,-.5);ax.set_xlim(-3,105);ax.set_xticks([0,50,100]);ax.grid(axis='x',color='#E9ECEE',lw=.65);ax.set_xlabel(['Answer match (%)','Recovery (%)','Accuracy (%)'][i]);ax.spines['bottom'].set_visible(False)
  else:
   xs=[[0,1,2.7,3.7],[0,1,2],[0,1,2.4,3.4]][i]
   for k,(g,o,ci) in enumerate(D[i]):
    for j,(v,col) in enumerate(zip([np.mean(g),o],colors)):
     x=xs[k]+(-.18 if j==0 else .18)
     if style=='bar':ax.bar(x,v,width=.31,color=col,zorder=3)
     else:ax.plot([x,x],[0,v],color=col,alpha=.22,lw=2);ax.scatter(x,v,s=39,color=col,zorder=4,edgecolor='white',linewidth=.6)
     if j==0:ax.scatter([x-.045,x+.045],g,s=9,facecolor='white' if style=='bar' else col,edgecolor='#36434D',lw=.45,zorder=5)
     else:ax.errorbar(x,v,yerr=[[v-ci[0]],[ci[1]-v]],fmt='none',ecolor='#48535D' if style=='bar' else col,elinewidth=.8,capsize=2,zorder=5)
   ax.set_xticks(xs,[['Pattern','Output','Pattern','Output'],['Selected','Control','All'],['Raw','With J','J → Raw','Raw → J']][i],fontsize=9)
   ax.set_ylim(-2,112);ax.set_yticks([0,50,100]);ax.set_ylabel(['Answer match (%)','Recovery (%)','Accuracy (%)'][i],fontsize=10);ax.grid(axis='y',color='#E9ECEE',lw=.65)
   if i==0:
    ax.text(.5,108,'Source route + base graph',ha='center',fontsize=9);ax.text(3.2,108,'Source answer',ha='center',fontsize=9)
    ax.axvline(1.85,color='#D5DBDF',lw=.65,ymax=.92)
   if i==2:ax.axvline(1.7,color='#D5DBDF',lw=.65,ymax=.92)
 handles=[Patch(facecolor=c,label=l) if style=='bar' else Line2D([0],[0],marker='o',color=c,lw=0,label=l,markersize=6) for c,l in zip(colors,['Graph A','Ouro'])]
 f.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.015),frameon=False,ncol=2,handlelength=1.1,columnspacing=2.2,fontsize=10)
 if style=='horizontal':f.subplots_adjust(left=.135,wspace=.7)
 for ext in ['pdf','svg','png']:f.savefig(O/(name+'.'+ext),dpi=220,facecolor='white')
 plt.close(f)
print(O)

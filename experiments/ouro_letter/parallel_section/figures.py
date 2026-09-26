import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;ROOT=Path('/data/paperexperiment/Documents/looped_transformer_paper_repro_20260920/paper_spotlight_v1');OUT=ROOT/'figures/v61'
G=json.loads((P/'graph_summary.json').read_text())['results']['A'];S=json.loads((P/'parallel_confirmation64/analysis.json').read_text())['conditions'];R=json.loads((P/'parallel_restore64/analysis.json').read_text())['conditions'];M=json.loads((P.parent/'localization/localize_confirmation64/analysis.json').read_text())['conditions']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42});BLUE='#1689D4';YELLOW='#D8ED16';GRAY='#BBC4CE'
def frame(n=2):
 f,a=plt.subplots(1,n,figsize=(10.6,3.15));f.subplots_adjust(left=.065,right=.99,bottom=.23,top=.88,wspace=.3)
 for ax,k in zip(a,'abc'):
  ax.text(-.13,1.05,k,transform=ax.transAxes,fontweight='bold',fontsize=14);ax.set_ylim(0,115);ax.set_yticks([0,25,50,75,100]);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
 return f,a
def bar(ax,x,vals,color,ci=None,label=True):
 val=float(np.mean(vals))*100;ax.bar(x,val,color=color,edgecolor='#5E6978',lw=.4,width=.65)
 if len(vals)>1:ax.scatter(np.array([-.07,.07])+x,np.array(vals)*100,s=18,color='#374556',zorder=4)
 if ci is not None:
  lo,hi=np.array(ci)*100;ax.errorbar(x,val,yerr=[[val-lo],[hi-val]],fmt='none',color='#374556',capsize=3,lw=1)
 else:hi=max(vals)*100
 if label:ax.text(x,hi+3,f'{val:.1f}',ha='center',fontsize=10)
def save(f,n):f.savefig(OUT/(n+'.pdf'));f.savefig(OUT/(n+'.png'),dpi=160);plt.close(f)
f,a=frame()
for ax,metric,column in zip(a,['cf','donor'],[2,1]):
 for x,typ,col in [(0,'pattern',BLUE),(1,'context',YELLOW)]:bar(ax,x,[G[k]['cross_graph']['own_eligible'][typ+'_'+metric]['mean'] for k in ['1','2']],col)
 for x,typ,col in [(3,'pattern',BLUE),(4,'output',YELLOW)]:
  z=S['selected_source_'+typ];target='counterfactual' if column==2 else 'source';bar(ax,x,[z[target]/64],col,z['wilson_ci95'][column])
 ax.set_xticks([.5,3.5],['Graph A','Ouro · 16 heads']);ax.set_ylabel('Answer match (%)')
a[0].set_title('Source route + base graph',fontsize=12);a[1].set_title('Source answer',fontsize=12)
f.legend([plt.Rectangle((0,0),1,1,color=BLUE),plt.Rectangle((0,0),1,1,color=YELLOW)],['Pattern patch','Head-output patch'],ncol=2,loc='lower center',frameon=False,bbox_to_anchor=(.5,-.025));save(f,'parallel_semantics')
f,a=frame()
for i,key in enumerate(['clean_context_H0','clean_context_H1','clean_context_H0123']):bar(a[0],i,[G[k]['restoration'][key]['mean'] for k in ['1','2']],[BLUE,GRAY,YELLOW][i])
a[0].set_xticks(range(3),['Selected H0','Best other head','All 4 heads']);a[0].set_title('Graph A',fontsize=12)
for i,key in enumerate(['restore_selected','restore_neighbor','restore_all']):bar(a[1],i,[R[key]['conditional_recovery']],[BLUE,GRAY,YELLOW][i],R[key]['conditional_wilson_ci95'])
a[1].set_xticks(range(3),['Selected 16','Matched 16','All 48 heads']);a[1].set_title('Ouro',fontsize=12)
for ax in a:ax.set_ylabel('Broken cases recovered (%)')
save(f,'parallel_restore')
f,a=frame();labels=['Raw','With J','Pattern\nrescue','Pattern\ndamage']
for i,key in enumerate(['native','full_J','rescue','damage']):bar(a[0],i,[G[k]['pattern_swap'][key]['mean'] for k in ['1','2']],[GRAY,YELLOW,BLUE,BLUE][i])
for i,key in enumerate(['native','J','size16_rescue_pattern','size16_damage_pattern']):bar(a[1],i,[M[key]['accuracy']],[GRAY,YELLOW,BLUE,BLUE][i],M[key]['wilson95'])
for ax,title in zip(a,['Graph A · 4 heads','Ouro · 16 heads, call 4']):ax.set_xticks(range(4),labels);ax.set_title(title,fontsize=12);ax.set_ylabel('Accuracy (%)')
save(f,'parallel_mediation')
f,a=frame()
for ax,hop in zip(a,['one','two']):
 for i,seed in enumerate(['1','2']):
  z=G[seed]['behavior']['J_'+hop]['distinct_labels'];ys=[z['pre_F_'+hop]['mean']*100,z['post_F_'+hop]['mean']*100];xs=np.array([0,1])+i*2.5;ax.plot(xs,ys,c=BLUE if i==0 else '#677480',lw=1.5);ax.scatter(xs[0],ys[0],marker='s',s=45,facecolor='white',edgecolor=BLUE if i==0 else '#677480');ax.scatter(xs[1],ys[1],s=45,c=BLUE if i==0 else '#677480');
  for x,y in zip(xs,ys):ax.text(x,y+4,f'{y:.1f}',ha='center',fontsize=10)
 ax.set_xticks([0,1,2.5,3.5],['After J','After F','After J','After F']);ax.set_ylabel('Target accuracy (%)');ax.set_title('One-hop target' if hop=='one' else 'Two-hop target',fontsize=12);ax.text(.18,-.21,'Fit 1',transform=ax.transAxes);ax.text(.75,-.21,'Fit 2',transform=ax.transAxes)
save(f,'target_control')

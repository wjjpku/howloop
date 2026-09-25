from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("PAPEREXPERIMENT_OUTPUT", ROOT / "outputs")) / "figures"
OUT.mkdir(parents=True, exist_ok=True)
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch,FancyBboxPatch
R=OUT;P=ROOT
rows=json.loads((ROOT/'data/plot/mechanism_intervals.json').read_text());order=[4,5,6,0,1,2,3,7,8,9,10];rows=[rows[i] for i in order]
plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'cm','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
ink='#273846';route='#597E9D';output='#A46C52'
f=plt.figure(figsize=(8,3.6));d=f.add_axes([.012,.12,.265,.82]);d.set(xlim=(0,1),ylim=(0,1));d.axis('off')
def text(x,y,s,**kw):d.text(x,y,s,ha='center',va='center',color=ink,**kw)
def box(x,y,w,h,label,color=ink,fill='#F4F6F8'):
 d.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.01,rounding_size=.018',lw=.8,edgecolor=color,facecolor=fill));text(x,y,label,fontsize=9)
def arrow(a,b,color=ink,ls='-',rad=0):d.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='-|>',color=color,lw=.85,linestyle=ls,connectionstyle=f'arc3,rad={rad}',shrinkA=1,shrinkB=1,mutation_scale=8))
text(.5,.98,'Attention and intervention sites',fontsize=10)
box(.5,.87,.78,.09,r'State $h$ or $J(h)$')
box(.3,.70,.33,.09,r'$Q, K$');box(.79,.70,.20,.09,r'$V$')
arrow((.37,.815),(.30,.757));arrow((.69,.815),(.79,.757))
box(.30,.51,.43,.14,'Attention pattern\n'+r'$\alpha$',route,'#EDF3F8')
# Compact formula sits below the pattern name, sized independently.
d.texts[-1].set_fontsize(9)
arrow((.30,.644),(.30,.591))
box(.50,.28,.50,.12,'Head output\n'+r'$z=\alpha V$',output,'#FAF1EC')
arrow((.30,.431),(.43,.352));arrow((.79,.644),(.62,.352),rad=-.12)
arrow((.50,.208),(.50,.143));text(.50,.10,'Output projection + remaining network',fontsize=7.8)
# Coloured boundary markers identify the objects that are replaced.
text(.08,.39,'(b, c)',fontsize=8.5);arrow((.13,.395),(.21,.444),route,'--')
text(.89,.27,'(a, b)',fontsize=8.5);arrow((.81,.28),(.76,.28),output,'--')
# Experimental keys below the schematic.
f.text(.024,.082,r'(a) Restore $z$   (b) Patch $\alpha$ or $z$',fontsize=8,color=ink)
f.text(.024,.035,r'(c) Exchange $\alpha$: steered $\leftrightarrow$ unsteered',fontsize=8,color=ink)
f.add_artist(plt.Line2D([.293,.293],[.10,.94],transform=f.transFigure,color='#D7DDE2',lw=.65))
ax=f.add_axes([.35,.31,.638,.48]);xs=np.array([0,.8,1.6,2.7,3.5,4.3,5.1,6.2,7,7.8,8.6]);colors=['#5C92B7','#CF8F94']
ledger=[]
for x,row in zip(xs,rows):
 for j,key in enumerate(['graph','ouro']):
  es=row[key] if j==0 else [row[key]];mean=np.mean([e['mean'] for e in es])*100;xx=x+[-.13,.13][j]
  ax.bar(xx,mean,width=.22,color=colors[j],alpha=.33,edgecolor=colors[j],linewidth=.7,zorder=2)
  for k,e in enumerate(es):
   xp=xx+([-.037,.037][k] if j==0 else 0);v=e['mean']*100;lo,hi=np.array(e['ci95'])*100
   ax.errorbar(xp,v,yerr=[[max(0,v-lo)],[max(0,hi-v)]],fmt='o' if k==0 else 'D',markersize=2.6,markerfacecolor='white',markeredgecolor=colors[j],markeredgewidth=.7,ecolor=colors[j],elinewidth=.7,capsize=1.4,zorder=4)
  ledger.append({'condition':len(ledger)//2,'model':key,'mean':float(mean),'entries':es})
ax.set_ylim(-1.5,104);ax.set_xlim(-.4,9);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel('Recovery / answer accuracy (%)',fontsize=8.5,labelpad=3)
labels=['Selected','Control','All','Pattern','Output','Pattern','Output','No J','Full J','J → no J','no J → J']
ax.set_xticks(xs,labels,rotation=50,ha='right',rotation_mode='anchor',fontsize=8);ax.tick_params(axis='y',labelsize=8)
ax.spines[['top','right']].set_visible(False);ax.spines[['left','bottom']].set_color('#6E7880');ax.tick_params(length=2,color='#6E7880',pad=2);ax.grid(axis='y',color='#E8EBEE',lw=.5);ax.set_axisbelow(True)
for x in [2.15,5.65]:ax.axvline(x,color='#CDD4DB',lw=.8,ls='--')
for x,title in [(.8,'(b) Restore outputs'),(3.9,'(c) Route vs. content'),(7.4,'(d) Transfer steering')]:ax.text(x,1.15,title,transform=ax.get_xaxis_transform(),ha='center',fontsize=9,color=ink)
for x,title in [(3.1,'Raw-graph\nanswer'),(4.7,'Corrupted\nanswer')]:ax.text(x,1.005,title,transform=ax.get_xaxis_transform(),ha='center',va='bottom',fontsize=7,color=ink,linespacing=1)
for x,title in [(.8,'Restored heads'),(3.9,'Patched activation'),(7.4,'Pattern source → receiver')]:ax.text(x,-.47,title,transform=ax.get_xaxis_transform(),ha='center',fontsize=8,color=ink)
f.legend(handles=[Patch(facecolor=c,alpha=.5,label=n) for c,n in zip(colors,['D8L6','Ouro'])],loc='upper center',bbox_to_anchor=(.69,.995),ncol=2,frameon=False,fontsize=9,handlelength=1.3)
for ext in ['pdf','svg','png']:f.savefig(R/f'fig4_uncropped.{ext}',dpi=250,facecolor='white')
(R/'values.json').write_text(json.dumps(ledger,indent=2));plt.close(f)

# Final v66 layout: deterministic vector crops of newly generated panels.
import runpy
runpy.run_path(str(ROOT/'plots/attention_components.py'),run_name='__main__')
import pymupdf as fitz
bars=fitz.open(OUT/'fig4_uncropped.pdf');diagram=fitz.open(OUT/'attention_components.pdf')
doc=fitz.open();page=doc.new_page(width=576,height=209.6134490966797)
page.show_pdf_page(fitz.Rect(204.48,0,576,209.6134490966797),bars,0,clip=fitz.Rect(175.104,0,576,226.18752))
page.show_pdf_page(fitz.Rect(0,26.5141,178.559,183.10022),diagram,0,clip=fitz.Rect(5.992,3.038,225.94911,195.92792))
page.insert_text((7,33.8),'(a)',fontname='tiro',fontsize=9,color=(.15,.22,.28))
doc.save(OUT/'fig4_attention_mechanism.pdf');doc.close()

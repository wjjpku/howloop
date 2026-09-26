from pathlib import Path
import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pymupdf as fitz
R=Path(__file__).resolve().parents[2];O=Path(__file__).parent
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'font.size':10,'pdf.fonttype':42,'mathtext.fontset':'cm','axes.spines.top':False,'axes.spines.right':False})
blue='#527FA3';red='#B56D57';ink='#293746'
v=json.loads((R/'edits_v65_fig4_panel_labels/values.json').read_text());old={(x['condition'],x['model']):x for x in v};S=json.loads((R/'paper_strengthening_20260925/analysis/graph_summary.json').read_text())['A']
f=plt.figure(figsize=(8,4.55))
# Original vector attention schematic is inserted into the empty upper-left region below.
f.text(.35,.953,'(b) Routing vs. content',fontsize=11)
f.text(.735,.953,'(c) Steering effect',fontsize=11)
for pos,cs,labels in [([.385,.59,.245,.29],[3,4,5,6],['Pattern','Output','Pattern','Output']),([.745,.59,.24,.29],[7,8,9,10],['No $J$','Full $J$',r'$J\to0$',r'$0\to J$'])]:
 ax=f.add_axes(pos)
 for j,model in enumerate(['graph','ouro']):
  xs=np.arange(4)+(j-.5)*.27;y=[old[c,model]['mean'] for c in cs];col=[blue,red][j];ax.bar(xs,y,.24,color=col,alpha=.35,zorder=2)
  for x,c in zip(xs,cs):
   for k,e in enumerate(old[c,model]['entries']):
    val=e['mean']*100;lo,hi=np.array(e['ci95'])*100;xx=x+(k-.5)*.055 if model=='graph' else x
    ax.errorbar(xx,val,yerr=[[val-lo],[hi-val]],fmt='o' if k==0 else 'D',ms=2.6,mfc='white',mec=col,ecolor=col,lw=.7,capsize=1.5,zorder=4)
 ax.set(ylim=(0,105),yticks=[0,50,100],xticks=range(4),xticklabels=labels);ax.tick_params(labelsize=9,length=2);ax.grid(axis='y',color='#e6e9eb',lw=.5);ax.set_axisbelow(True)
 if cs[0]==3:
  ax.set_ylabel('Answer rate (%)',fontsize=9,labelpad=3);ax.text(.5,1.02,'Rerouted',transform=ax.get_xaxis_transform(),ha='center',fontsize=8.5);ax.text(2.5,1.02,'Source',transform=ax.get_xaxis_transform(),ha='center',fontsize=8.5);ax.axvline(1.5,color='#c9cfd4',lw=.7,ls='--')
 else:ax.set_ylabel('Accuracy (%)',fontsize=9,labelpad=3)
f.legend(handles=[plt.Line2D([],[],color=blue,lw=5,alpha=.6,label='D8L6'),plt.Line2D([],[],color=red,lw=5,alpha=.6,label='Ouro')],loc='center',bbox_to_anchor=(.66,.49),frameon=False,ncol=2,fontsize=9)
f.text(.035,.422,'(d) Attention patterns redirect the next target',fontsize=11)
for receiver,direction,pos,title in [('one','two_to_one',[.17,.115,.31,.235],r'$J_{\mathrm{two}}\to J_{\mathrm{one}}$'),('two','one_to_two',[.665,.115,.31,.235],r'$J_{\mathrm{one}}\to J_{\mathrm{two}}$')]:
 ax=f.add_axes(pos);ax.axhspan(1.6,2.4,color='#f0f3f5',zorder=0)
 rows=[S['baseline'][receiver]]+[S['interventions'][f'{direction}/{l}/pattern'] for l in ['L1','L2','L12']]
 for i,row in enumerate(rows):
  a,b=[row[k]['mean']*100 for k in ['one','two']];ax.plot([a,b],[i,i],color='#c4ccd2',lw=1,zorder=1)
  for k,col in [('one',blue),('two',red)]:
   e=row[k];val=e['mean']*100;lo,hi=np.array(e['ci95'])*100
   ax.errorbar(val,i,xerr=[[val-lo],[hi-val]],fmt='o' if k=='one' else 'D',ms=4.5,color=col,mec='white',mew=.5,lw=1.1,capsize=2,zorder=3)
   ax.annotate(f'{val:.1f}',(val,i),xytext=(7 if (val<5 or (5<=val<=95 and val==max(a,b))) else -7,0),textcoords='offset points',ha='left' if (val<5 or (5<=val<=95 and val==max(a,b))) else 'right',va='center',bbox=dict(facecolor='white',edgecolor='none',pad=.4),fontsize=9,color=col,fontweight='bold' if i==2 else 'normal')
 ax.set(xlim=(-2,102),ylim=(3.55,-.6),yticks=range(4),yticklabels=['Own pattern','Swap L1','Swap L2','Swap both'],xticks=[0,25,50,75,100]);ax.set_title(title,fontsize=12,pad=10);ax.tick_params(length=2,labelsize=9);ax.spines['left'].set_visible(False);ax.spines['bottom'].set_color('#9ca7b0');ax.grid(axis='x',color='#e7eaed',lw=.5);ax.set_axisbelow(True)
f.legend(handles=[plt.Line2D([],[],color=blue,marker='o',lw=0,label='One-hop answer'),plt.Line2D([],[],color=red,marker='D',lw=0,label='Two-hop answer')],loc='center',bbox_to_anchor=(.55,.022),frameon=False,ncol=2,fontsize=9)
f.text(.57,.057,'Answer rate (%)',ha='center',fontsize=9)
f.savefig(O/'mechanism_base.pdf');plt.close(f)
doc=fitz.open(O/'mechanism_base.pdf');oldpdf=fitz.open(R/'v66/target_switch_revision/fig04_before.pdf');doc[0].show_pdf_page(fitz.Rect(0,10,190,173),oldpdf,0,clip=fitz.Rect(0,20,195,190));doc.save(R/'v66/overleaf/figures/fig04_attention_mechanism.pdf',garbage=4,deflate=True);doc[0].get_pixmap(matrix=fitz.Matrix(2,2)).save(O/'fig04.png')
# Figure 7: broaden supervision and steering; remove the capacity panel.
src=(R/'edits_v65_fig7_taller/draw.py').read_text();src=src[:src.index("b=axs[3]")];src=src.replace("O=Path(__file__).parent",f"O=Path({str(O)!r})").replace("O.parent/'edits_v65_supervision/data.json'",f"Path({str(R/'edits_v65_supervision/data.json')!r})")
src=src.replace('figsize=(10.8,3.25)','figsize=(8.2,2.85)').replace('[[.045,.23,.15,.65],[.225,.23,.15,.65],[.45,.23,.22,.65],[.795,.23,.165,.65]]','[[.065,.25,.23,.60],[.335,.25,.23,.60],[.70,.25,.285,.60]]').replace("'font.size':12","'font.size':10").replace('fontsize=11','fontsize=10').replace('fontsize=9.5','fontsize=9').replace('fontsize=10,handlelength','fontsize=9,handlelength')
src+="\nfor ext in ['pdf','png']:f.savefig(O/f'fig07.{ext}',dpi=200,bbox_inches='tight',pad_inches=.04)\n";exec(compile(src,'fig7','exec'),{});shutil=None
import shutil
shutil.copy2(O/'fig07.pdf',R/'v66/overleaf/figures/fig07_supervision_capacity.pdf')

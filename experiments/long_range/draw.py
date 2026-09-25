from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
P=Path('/Users/jiaju/Documents/ChatGPT/final paper/fig6_retest_20260924');D=P/'results';R=Path('/Users/jiaju/Documents/looped_transformer_paper_repro_20260920/paper_spotlight_v1');O=R/'figures/v64';O.mkdir(exist_ok=True)
BLUE='#7E99F4';RED='#CC7C71'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#A4AFB8','text.color':'#344350','axes.labelcolor':'#344350','pdf.fonttype':42})
def panel(letter):
 f,ax=plt.subplots(figsize=(4.5,2.8));f.subplots_adjust(left=.15,right=.94,bottom=.20,top=.90);ax.set(ylim=(-1,104),yticks=[0,25,50,75,100],xlabel='Loop index',ylabel='Accuracy (%)');ax.grid(axis='y',alpha=.18);ax.text(-.13,1.045,letter,transform=ax.transAxes,fontweight='bold',fontsize=12);return f,ax

def save(f,name):
 for ext in ['pdf','png']:f.savefig(O/f'{name}.{ext}',dpi=200)
 plt.close(f)
old=list(csv.DictReader((R/'revisions/v59_parity/data/parity_long500.csv').open()));new=list(csv.DictReader((D/'parity_dense_extension.csv').open()));f,ax=panel('a')
for v,col in [('raw',BLUE),('J',RED)]:
 ds=sorted([r for r in old if r['variant']==v and int(r['seed'])==2],key=lambda r:int(r['length']));x=np.array([int(r['length']) for r in ds]);y=np.array([float(r['exact_match'])*100 for r in ds]);assert len(ds)==500
 old_x=np.convolve(x,np.ones(10)/10,'valid');old_y=np.convolve(y,np.ones(10)/10,'valid')
 ds=sorted([r for r in new if r['variant']==v],key=lambda r:int(r['length']));x=np.array([int(r['length']) for r in ds]);y=np.array([float(r['exact_match'])*100 for r in ds]);assert len(ds)==101
 ax.plot(x,y,color=col,alpha=.23,marker='.',ms=2,lw=.55)
 new_x=np.convolve(x,np.ones(2)/2,'valid');new_y=np.convolve(y,np.ones(2)/2,'valid')
 ax.plot(np.concatenate([old_x,new_x]),np.concatenate([old_y,new_y]),color=col,lw=1.6)
ax.axvspan(20,40,color='#A5B7C5',alpha=.22);ax.set(xlim=(0,1000),xticks=[0,250,500,750,1000]);save(f,'fig6a_parity')
z=np.load(D/'graph_curve_summary.npz');f,ax=panel('b');t=np.arange(9,41);ax.axvspan(8.5,16.5,color='#A5B7C5',alpha=.22);ax.plot(t,z['raw'][8:40]*100,color=BLUE,lw=1.6);ax.plot(t,z['full'][8:40]*100,color=RED,lw=1.6);ax.fill_between(t,z['low'][8:40]*100,z['high'][8:40]*100,color=RED,alpha=.17,linewidth=0);ax.set(xlim=(9,40),xticks=[9,16,24,32,40]);save(f,'fig6b_graph')
f=plt.figure(figsize=(9,.35));f.legend(handles=[Line2D([0],[0],color=BLUE,lw=2,label='Without J'),Line2D([0],[0],color=RED,lw=2,label='With J')],loc='center',frameon=False,ncol=2);save(f,'fig6_legend')
layout=r'''\setcounter{subfigure}{0}%
\begin{subfigure}[t]{.495\linewidth}
\includegraphics[width=\linewidth]{figures/v64/fig6a_parity.pdf}
\phantomcaption\label{fig:v64-parity}
\end{subfigure}\hfill
\begin{subfigure}[t]{.495\linewidth}
\includegraphics[width=\linewidth]{figures/v64/fig6b_graph.pdf}
\phantomcaption\label{fig:v64-graph}
\end{subfigure}
\par\nointerlineskip
\includegraphics[width=\linewidth]{figures/v64/fig6_legend.pdf}
'''
(R/'revisions/v64_retest').mkdir(exist_ok=True);(R/'revisions/v64_retest/fig6.tex').write_text(layout)

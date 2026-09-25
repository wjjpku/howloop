from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("PAPEREXPERIMENT_OUTPUT", ROOT / "outputs")) / "figures"
OUT.mkdir(parents=True, exist_ok=True)
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=ROOT/'experiments/n10/figure_work'
P=ROOT
O=OUT
sources=['source/trajectories/L8_seed10/summary.json','source/trajectories/L8_seed5/summary.json','source/local/A/native/summary.json','source/local/E/native/summary.json']
plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'cm','font.size':10,'pdf.fonttype':42,'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9})
fig=plt.figure(figsize=(7,2.5))
for i,(label,source) in enumerate(zip('abcd',sources)):
 d=json.loads((R/source).read_text());m=np.array(d['splits']['rings']['category_match'])[:,:17]
 assert m.shape==(10,17) and np.allclose(m.sum(0),1,atol=1e-6)
 ax=fig.add_axes([.055+i*.2325,.20,.21,.76])
 im=ax.imshow(m.T,origin='upper',aspect='auto',cmap='viridis',vmin=0,vmax=1,interpolation='nearest',extent=(-.5,9.5,16.5,-.5))
 for loop in range(17):
  col=m[:,loop];matches=np.flatnonzero(np.isclose(col,col.max(),atol=1e-7))
  if len(matches)==1:ax.scatter(matches,[loop],s=12,c='white',edgecolors='#343434',linewidths=.35,zorder=4)
 ax.axhline(d['loops']+.5,color='#FF5252',linestyle='--',linewidth=1.15)
 ax.set_xticks(range(0,10,2),[f'$f^{{{k}}}$' for k in range(0,10,2)])
 ax.set_yticks(range(0,17,4));ax.set_xlabel('Node along path',labelpad=2);ax.set_ylabel('Loop' if label=='a' else '',labelpad=1)
 if label!='a':ax.tick_params(axis='y',labelleft=False)
 ax.tick_params(length=3,width=.7,pad=2)
 if label=='d':
  cb=fig.colorbar(im,cax=fig.add_axes([.975,.20,.006,.76]));cb.set_ticks([0,1]);cb.ax.tick_params(labelsize=8,length=2,pad=1)
fig.savefig(O/'combined.pdf',facecolor='white');plt.close(fig)
import pymupdf as fitz
doc=fitz.open(O/'combined.pdf');page=doc[0];cuts=[0,.27625,.50875,.74125,1]
for i,label in enumerate('abcd'):
 clip=fitz.Rect(cuts[i]*page.rect.width,0,cuts[i+1]*page.rect.width,page.rect.height)
 dst=fitz.open();p=dst.new_page(width=clip.width,height=clip.height);p.show_pdf_page(p.rect,doc,0,clip=clip);dst.save(O/f'fig2{label}_readout.pdf');dst.close()

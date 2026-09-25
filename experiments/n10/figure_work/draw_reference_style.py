from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
R=Path(__file__).resolve().parent
panels=[('source/trajectories/L8_seed10/summary.json','(a) D8L8'),('source/trajectories/L8_seed5/summary.json','(b) D8L8'),('source/local/A/native/summary.json','(c) D8L6'),('source/local/E/native/summary.json','(d) D8L6')]
for font in ['Arial','Helvetica']:
 with plt.rc_context({'font.family':font,'font.size':10,'pdf.fonttype':42,'savefig.dpi':240}):
  fig=plt.figure(figsize=(13,4.4))
  width=.197
  height=.65
  axes=[]
  for i,(source,title) in enumerate(panels):
   ax=fig.add_axes([.058+.218*i,.17,width,height]);axes.append(ax)
   d=json.loads((R/source).read_text());m=np.array(d['splits']['rings']['category_match'])[:,:17]
   assert m.shape==(10,17) and np.allclose(m.sum(0),1,atol=1e-6)
   im=ax.imshow(m.T,origin='upper',aspect='auto',cmap='viridis',vmin=0,vmax=1,interpolation='nearest',extent=(-.5,9.5,16.5,-.5))
   # Equal maxima are not arbitrarily presented as a uniquely preferred node.
   for loop in range(17):
    col=m[:,loop];matches=np.flatnonzero(np.isclose(col,col.max(),atol=1e-7))
    if len(matches)==1:ax.scatter(matches,[loop],s=9,c='white',edgecolors='#343434',linewidths=.35,zorder=4)
   ax.axhline(d['loops']+.5,color='#FF5252',linestyle='--',linewidth=1.15)
   ax.set_xticks(range(10),[f'f$^{{{k}}}$' for k in range(10)])
   ax.set_yticks(range(0,17,2));ax.set_xlabel('node iteration',labelpad=5)
   ax.set_title(title,fontsize=12,pad=9)
   ax.tick_params(length=4,width=.9,labelsize=9)
   for spine in ax.spines.values():spine.set_linewidth(.9)
  axes[0].set_ylabel('loop index',labelpad=8)
  cb=fig.colorbar(im,cax=fig.add_axes([.933,.19,.012,height-.04]))
  cb.set_ticks(np.linspace(0,1,6));cb.set_label('top-1 output fraction');cb.ax.tick_params(labelsize=9)
  fig.text(.058,.962,'Intermediate readout on held-out random 10-cycle graphs',ha='left',va='top',fontsize=14,fontweight='bold')
  fig.legend(handles=[Line2D([],[],linestyle='none',marker='o',markersize=4,markerfacecolor='white',markeredgecolor='#343434',markeredgewidth=.6,label=r'$\arg\max_k\,p_\ell(k)$'),Line2D([],[],color='#FF5252',linestyle='--',linewidth=1.15,label=r'$\ell=L$')],loc='lower center',bbox_to_anchor=(.5,-.025),ncol=2,frameon=False,fontsize=10,handlelength=2.2,columnspacing=3)
  fig.canvas.draw()
  for ext in ['png','pdf','svg']:fig.savefig(R/'figures'/f'section3_reference_compact_{font.lower()}.{ext}',bbox_inches='tight',pad_inches=.12,facecolor='white')
  plt.close(fig)

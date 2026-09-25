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
 with plt.rc_context({'font.family':font,'font.size':10,'pdf.fonttype':42,'savefig.dpi':260,'axes.labelcolor':'#262626','text.color':'#262626','xtick.color':'#444444','ytick.color':'#444444'}):
  fig=plt.figure(figsize=(12.8,5.8))
  panel_width=.215
  panel_height=panel_width*12.8*(17/10)/5.8
  axes=[]
  for i,(source,title) in enumerate(panels):
   ax=fig.add_axes([.06+i*.23,.14,panel_width,panel_height]);axes.append(ax)
   d=json.loads((R/source).read_text());m=np.array(d['splits']['rings']['category_match'])
   assert m.shape==(10,17) and np.allclose(m.sum(0),1,atol=1e-6)
   im=ax.imshow(m.T,origin='upper',aspect='equal',cmap='viridis',vmin=0,vmax=1,interpolation='nearest',extent=(-.5,9.5,16.5,-.5))
   ax.axhline(d['loops'],color='white',linestyle='--',linewidth=2.1,alpha=.8)
   ax.axhline(d['loops'],color='#E34B4B',linestyle='--',linewidth=1.2)
   ax.set_xticks([0,2,4,6,8,9]);ax.set_yticks([0,4,8,12,16])
   ax.set_yticklabels([str(k) for k in [0,4,8,12,16]] if i==0 else [])
   ax.set_title(title,fontsize=11,fontweight='normal',pad=12)
   ax.tick_params(axis='both',length=0,pad=6,labelsize=9)
   for spine in ax.spines.values():spine.set_visible(False)
  axes[0].set_ylabel('loop index',labelpad=10,fontsize=11)
  cb=fig.colorbar(im,cax=fig.add_axes([.16,.045,.19,.017]),orientation='horizontal')
  cb.set_ticks([0,.5,1]);cb.ax.tick_params(length=0,pad=4,labelsize=9);cb.outline.set_visible(False)
  fig.text(.148,.0535,'top-1 fraction',ha='right',va='center',fontsize=10)
  fig.text(.5125,.0535,'node iteration',ha='center',va='center',fontsize=11)
  fig.legend(handles=[Line2D([0],[0],color='#E34B4B',linestyle='--',linewidth=1.4,label='Supervised loop (D8L8: 8; D8L6: 6)')],loc='center left',bbox_to_anchor=(.64,.0535),frameon=False,fontsize=9,handlelength=2.2)
  fig.canvas.draw()
  for ax in axes:
   bbox=ax.get_window_extent()
   assert abs((bbox.width/10)/(bbox.height/17)-1)<1e-6, 'Cells must be square'
  for ext in ['png','pdf','svg']:
   fig.savefig(R/'figures'/f'section3_trajectories_transposed_{font.lower()}.{ext}',facecolor='white',bbox_inches='tight',pad_inches=.09)
  plt.close(fig)

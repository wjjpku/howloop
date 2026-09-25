from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
R=Path(__file__).resolve().parent
panels=[('source/trajectories/L8_seed10/summary.json','(a) D8L8'),('source/trajectories/L8_seed5/summary.json','(b) D8L8'),('source/local/A/native/summary.json','(c) D8L6'),('source/local/E/native/summary.json','(d) D8L6')]
for font in ['Arial','Helvetica']:
 font_manager.findfont(font,fallback_to_default=False)
 with plt.rc_context({'font.family':font,'font.size':10,'mathtext.fontset':'custom','mathtext.rm':font,'mathtext.it':font+':italic','mathtext.bf':font+':bold','pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':240}):
  fig,axes=plt.subplots(1,4,figsize=(13,3.45),sharey=True)
  for ax,(source,title) in zip(axes,panels):
   d=json.loads((R/source).read_text());m=np.array(d['splits']['rings']['category_match'])
   assert m.shape==(10,17) and np.allclose(m.sum(0),1,atol=1e-6)
   im=ax.imshow(m,origin='lower',aspect='auto',cmap='viridis',vmin=0,vmax=1,interpolation='nearest',extent=(-.5,16.5,-.5,9.5))
   ax.axvline(d['loops'],color='#E64B4B',linestyle='--',linewidth=1.3)
   ax.set_xticks([0,4,8,12,16]);ax.set_yticks(range(10),[f'$f^{k}(s)$' for k in range(10)])
   ax.set_xlabel('loop index');ax.set_title(title,loc='left',fontweight='bold',fontsize=11,pad=10)
   ax.tick_params(length=0)
   for spine in ax.spines.values():spine.set_visible(False)
  axes[0].set_ylabel('node')
  fig.subplots_adjust(left=.065,right=.91,bottom=.27,top=.90,wspace=.16)
  cb=fig.colorbar(im,cax=fig.add_axes([.93,.27,.012,.63]));cb.set_label('top-1 fraction');cb.set_ticks([0,.5,1]);cb.outline.set_visible(False)
  fig.legend(handles=[Line2D([0],[0],color='#E64B4B',linestyle='--',linewidth=1.3,label='Supervision position during training (D8L8: loop 8; D8L6: loop 6)')],loc='lower center',bbox_to_anchor=(.5,.02),frameon=False,fontsize=10,handlelength=2.5)
  for ext in ['pdf','png','svg']:
   fig.savefig(R/'figures'/f'section3_trajectories_{font.lower()}.{ext}',bbox_inches='tight',facecolor='white')
  plt.close(fig)

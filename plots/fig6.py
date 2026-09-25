from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("PAPEREXPERIMENT_OUTPUT", ROOT / "outputs")) / "figures"
OUT.mkdir(parents=True, exist_ok=True)
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch
R=ROOT;D=ROOT/'data/plot/parity';O=OUT;out=OUT
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#87929C','text.color':'#344350','axes.labelcolor':'#344350','pdf.fonttype':42})
f=plt.figure(figsize=(10.8,3.15))
# Equal square heatmaps; a narrower phase panel with matching plot height.
a=f.add_axes([.055,.20,.22,.754]);b=f.add_axes([.335,.20,.22,.754]);c=f.add_axes([.715,.20,.275,.754]);cbax=f.add_axes([.575,.20,.012,.754])
rows=list(csv.DictReader((D/'parity_seed2.csv').open()))
for ax,v,title in [(a,'raw','(a) Without J'),(b,'J','(b) With J')]:
 z=np.full((60,60),np.nan)
 for r in rows:
  n,t=int(r['length']),int(r['loop'])
  if r['variant']==v and n<=60 and t<=60:z[n-1,t-1]=float(r['exact_match'])
 assert np.isfinite(z).all()
 im=ax.imshow(z,origin='lower',extent=(.5,60.5,.5,60.5),cmap='viridis',vmin=0,vmax=1,aspect='equal',interpolation='nearest')
 ax.plot([1,60],[1,60],'--',color='white',lw=.8);ax.axhline(40.5,color='white',ls=':',lw=.9)
 ax.set(xlabel='Loop index t',xticks=[20,40,60],yticks=[20,40,60]);ax.set_title(title,loc='left',fontsize=11,pad=8)
 # Identical cell-preserving zooms; local maxima are restricted to t=n-1,n,n+1.
 box=Rectangle((53.5,53.5),7,7,fill=False,edgecolor='#FFAD71',lw=1.05,zorder=6)
 ax.add_patch(box)
 ins=ax.inset_axes([.105,.525,.405,.405],zorder=10)
 ins.imshow(z,origin='lower',extent=(.5,60.5,.5,60.5),cmap='viridis',vmin=0,vmax=1,aspect='equal',interpolation='nearest')
 ins.set(xlim=(53.5,60.5),ylim=(53.5,60.5),xticks=[54,57,60],yticks=[54,57,60])
 ins.set_xticks(np.arange(53.5,61,1),minor=True);ins.set_yticks(np.arange(53.5,61,1),minor=True)
 ins.grid(which='minor',color='white',alpha=.25,lw=.35)
 ins.tick_params(which='major',labelsize=6.5,length=2,pad=1,colors='white');ins.tick_params(which='minor',length=0)
 ins.plot([53.5,60.5],[53.5,60.5],'--',color='white',lw=.8,zorder=3)
 for nn in range(54,61):
  ts=np.arange(nn-1,min(nn+1,60)+1);vals=z[nn-1,ts-1];best=ts[vals==vals.max()]
  if len(best)==1:ins.plot(best[0],nn,'o',ms=3.2,mfc='white',mec='#243440',mew=.65,zorder=4)
 for sp in ins.spines.values():sp.set_visible(True);sp.set_color('#FFAD71');sp.set_linewidth(1)
 ax.add_artist(ConnectionPatch(xyA=(60.5,60.5),coordsA=ax.transData,xyB=(60.5,60.5),coordsB=ins.transData,color='#FFAD71',lw=.7,zorder=7))

a.set_ylabel('Sequence length n');b.tick_params(labelleft=False)
cb=f.colorbar(im,cax=cbax,ticks=[0,.5,1]);cb.ax.set_title('Acc.',fontsize=9,pad=8)
phase=list(csv.DictReader((D/'phase_seed2.csv').open()))
for v,col,lab in [('raw','#168EC1','Without J'),('J','#A9BD00','With J')]:
 ds=[r for r in phase if r['variant']==v and r['metric']=='exact_match' and 41<=int(r['length_n'])<=490]
 n=np.array([int(r['length_n']) for r in ds]);y=np.array([float(r['unwrapped_offset_t_minus_n']) for r in ds]);assert len(n)>400
 c.plot(n,y,color=col,lw=1,label=lab);coef=np.polyfit(n,y,1);c.plot(n,np.polyval(coef,n),color=col,ls='--',lw=1.6)
c.set(xlabel='Sequence length n',ylabel='Readout offset t − n (loops)',xticks=[100,300,500]);c.set_title('(c) Readout schedule',loc='left',fontsize=11,pad=8);c.legend(frameon=False,fontsize=9,loc='lower left');c.grid(alpha=.15)
for ext in ['pdf','png']:f.savefig(O/f'fig6_parity_timing.{ext}',dpi=220,bbox_inches='tight',pad_inches=.04)

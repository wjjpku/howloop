"""All three saved uniform controllers, whole-answer accuracy at t=n."""
from pathlib import Path
import os,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pymupdf as fitz
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(os.environ.get('PAPEREXPERIMENT_OUTPUT',ROOT/'outputs'))/'figures'
OUT.mkdir(parents=True,exist_ok=True)
rows=list(csv.DictReader((ROOT/'data/plot/parity_long500/registered_accuracy.csv').open()))
fig,axes=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
for seed,ax in enumerate(axes):
 for variant,color in [('raw','#707070'),('J','#26749a')]:
  rr=sorted([r for r in rows if int(r['seed'])==seed and r['variant']==variant],key=lambda r:int(r['length']))
  x=np.array([int(r['length']) for r in rr]);y=np.array([float(r['exact_match']) for r in rr])
  assert len(x)==500 and np.array_equal(x,np.arange(1,501))
  ax.plot(x,y,color=color,alpha=.3,lw=.7)
  ax.plot(np.arange(10.5,490.6),np.convolve(y,np.ones(20)/20,'valid'),color=color,lw=1.5,label=variant)
 ax.axvspan(20,40,color='#aaa',alpha=.2)
 ax.set(title=f'Seed {seed}',xlabel='Input length n; read at t=n',ylim=(0,1.02));ax.legend()
axes[0].set_ylabel('Exact match')
fig.suptitle('Same uniform 20-40 controllers; faint raw points, bold 20-length moving mean')
fig.savefig(OUT/'parity_uniform_long_combined.pdf');plt.close(fig)
doc=fitz.open(OUT/'parity_uniform_long_combined.pdf');w,h=doc[0].rect.width,doc[0].rect.height
# Identical whitespace cuts used for the submitted v66 independent panels.
doc[0].add_redact_annot(fitz.Rect(0,0,w,10),fill=False);doc[0].apply_redactions(images=0,graphics=0)
cuts=[0,296,579,w]
for seed,letter in enumerate('abc'):
 clip=fitz.Rect(cuts[seed],0,cuts[seed+1],h);dst=fitz.open();page=dst.new_page(width=clip.width,height=clip.height);page.show_pdf_page(page.rect,doc,0,clip=clip);dst.save(OUT/f'figS2{letter}_parity_seed{seed}.pdf');dst.close()

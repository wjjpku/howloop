from pathlib import Path
import os
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(os.environ.get('PAPEREXPERIMENT_OUTPUT',ROOT/'outputs'))/'figures'
OUT.mkdir(parents=True,exist_ok=True)
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyArrowPatch
R=OUT
plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'cm','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none'})
f,ax=plt.subplots(figsize=(3.2,3.2));f.subplots_adjust(0,0,1,1);ax.set(xlim=(0,100),ylim=(0,100));ax.axis('off')
ink='#293746';blue='#527FA3';pink='#A97785';warm='#AF7056'
def text(x,y,s,size=9,ha='center'):ax.text(x,y,s,fontsize=size,color=ink,ha=ha,va='center',zorder=30)
def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=7,lw=.65,color=ink,shrinkA=.7,shrinkB=.7,zorder=25))
def path(points):
 ax.plot(*zip(*points[:-1]),lw=.65,color=ink);arrow(points[-2],points[-1])
def stack(x,y,color,vertical=True):
 for i in (range(2,-1,-1) if vertical else range(3)):
  xx=x+3*i;yy=y+2*i;w,h=(5,17) if vertical else (18,4.5)
  for k in range(3):
   cx,cy,cw,ch=(xx,yy+k*h/3,w,h/3) if vertical else (xx+k*w/3,yy,w/3,h)
   ax.add_patch(Rectangle((cx,cy),cw,ch,facecolor='#E1ECF5' if color==blue else '#F2E0D6' if color==warm else '#F1F3F5',edgecolor=color,lw=.5,zorder=10-i if vertical else 10+i))
def grid(x,y,pattern=False):
 for r in range(3):
  for c in range(3):
   hit=r==0 and c==0
   if pattern:
    from matplotlib.colors import to_rgb
    vals=[[.2,.6,.2],[.55,.15,.3],[.15,.25,.6]]
    a=.07+.85*vals[r][c];rgb=to_rgb(blue);fill=tuple(1-a+a*v for v in rgb)
   else:fill='#D5A0AD' if hit else '#F4E2E7'
   ax.add_patch(Rectangle((x+c*8,y+(2-r)*8),8,8,facecolor=fill,edgecolor=blue if pattern else pink,lw=.5,hatch='///' if hit and not pattern else None))

# Component-only diagram: Q/K -> pattern, pattern and V -> output.
stack(41,66,blue);text(46,95,'queries '+r'$Q$',11);text(38,73,r'$q_1$',9,ha='right')
stack(3,39,ink,False);text(15,58,'keys '+r'$K$',11);text(27,50,r'$k_1$',9)
grid(40,30,True)
arrow((43.5,65),(44,54.5));path([(27.4,45.25),(33,45.25),(33,50),(39.4,50)])
text(51,21,'Attention pattern\n'+r'$\alpha$',11)
stack(80,66,ink);text(85,95,'values '+r'$V$',11)
path([(85,65),(85,58),(73,58),(73,45)])
arrow((64.6,42),(70.7,42));text(73,42,r'$\times$',11)
arrow((76,42),(81,42));stack(83,32,warm)
text(85,21,'Head output\n'+r'$z=\alpha V$',11)
for ext in ['svg','pdf','png']:f.savefig(R/f'attention_components.{ext}',dpi=300,facecolor='white',bbox_inches='tight',pad_inches=.015)
plt.close(f)

"""Select head groups on 16 discovery sequences only; summarize on held-out 48."""
import json
from pathlib import Path
import numpy as np
from diagnose_ouro_heads import ROOT

def main():
    arrays = {a: dict(np.load(ROOT/a/'collect/activations.npz')) for a in ['j4','plain4','j8']}
    def top(x, n=8):
        return [list(map(int,np.unravel_index(i,x.shape))) for i in np.argsort(x.reshape(-1))[-n:][::-1]]
    def difference(a,b,sl):
        x,y=arrays[a]['vectors'][sl],arrays[b]['vectors'][sl]
        num=np.linalg.norm(x-y,axis=-1).mean(0)
        den=(np.linalg.norm(x,axis=-1)+np.linalg.norm(y,axis=-1)).mean(0)/2
        return num/np.maximum(den,1e-6)
    groups={}
    rng=np.random.default_rng(20260915)
    for name,a,b in [('j_change','j4','plain4'),('k_change','j8','j4')]:
        groups[name]=top(difference(a,b,slice(0,16)))
        control=[]
        for l,d,h in groups[name]:
            choices=[i for i in range(16) if [l,d,i] not in groups[name]+control]
            control.append([l,d,int(rng.choice(choices))])
        groups[name+'_layer_matched_random']=control
    shared=sum(arrays[a]['rms'][:16].mean(0) for a in arrays)/3
    groups['high_activity']=top(shared)
    (ROOT/'groups.json').write_text(json.dumps(groups,indent=2))
    report={}
    for a,b in [('j4','plain4'),('j8','j4')]:
        x,y=arrays[a]['rms'][16:].mean(0),arrays[b]['rms'][16:].mean(0)
        stats=[]
        for l in range(4):
            xx,yy=x[l].ravel(),y[l].ravel();n=77
            overlap=len(set(np.argsort(xx)[-n:])&set(np.argsort(yy)[-n:]))/n
            vx=np.linalg.norm(arrays[a]['vectors'][16:,l],axis=-1).mean(0).ravel()
            vy=np.linalg.norm(arrays[b]['vectors'][16:,l],axis=-1).mean(0).ravel()
            stats.append(dict(loop=l+1,rms_profile_cosine=float(xx@yy/(np.linalg.norm(xx)*np.linalg.norm(yy))),
                top10pct_overlap=overlap,
                last_prompt_top10pct_overlap=len(set(np.argsort(vx)[-n:])&set(np.argsort(vy)[-n:]))/n,
                last_prompt_norm_profile_cosine=float(vx@vy/(np.linalg.norm(vx)*np.linalg.norm(vy))),
                mean_relative_vector_change=float(difference(a,b,slice(16,64))[l].mean())))
        report[a+'_vs_'+b]=stats
    report['loop1_j4_plain4_max_absolute_difference']=float(np.abs(arrays['j4']['vectors'][:,0]-arrays['plain4']['vectors'][:,0]).max())
    report['scope']='Activation summary is descriptive, not causal; head selection first16, summary last48; layer/head indices in groups zero-based.'
    (ROOT/'activity_summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()

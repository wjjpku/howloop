"""Automatic causal-only maps and confirmed key-head overlap; no activation ranks."""
import argparse,json,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/data/paperexperiment/ouro26_causal_head_scan_20260915')
ARMS=['plain4','j4','j8'];LABELS=['No J, k=4','J, k=4','J, k=8']
def rows(path):return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
def main():
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');a=p.parse_args()
    deadline=time.time()+4*3600
    while not all((ROOT/f'worker{w}'/'complete.json').exists() for w in [0,1]):
        if not a.wait or time.time()>deadline:raise RuntimeError('Causal confirmation incomplete')
        time.sleep(15)
    screen=sum([rows(ROOT/f'worker{w}'/'screen.jsonl') for w in [0,1]],[])
    confirm=sum([rows(ROOT/f'worker{w}'/'confirmation.jsonl') for w in [0,1]],[])
    base={r['arm']:r for r in rows(ROOT/'worker0/confirmation_baseline.jsonl')}
    common=[i for i in range(16,64) if all(next(r for r in base[arm]['rows'] if r['i']==i)['correct'] for arm in ARMS)]
    rng=np.random.default_rng(20260915)
    for r in confirm:
        old={x['i']:x for x in base[r['arm']]['rows']}
        delta=np.array([int(old[x['i']]['correct'])-int(x['correct']) for x in r['rows']])
        boot=rng.choice(delta,(10000,len(delta)),replace=True).mean(1)*100
        r['paired_bootstrap95_pp']=np.quantile(boot,[.025,.975]).tolist()
        r['common_correct_n']=len(common)
        r['common_correct_breaks']=sum(not x['correct'] for x in r['rows'] if x['i'] in common)
    overlap={}
    for threshold in [5,10,20]:
        sets={arm:{r['head_id'] for r in confirm if r['arm']==arm and r['accuracy_drop_pp']>=threshold} for arm in ARMS}
        comparisons=[]
        for x,y in [('j4','plain4'),('j8','j4')]:
            union=sets[x]|sets[y];intersection=sets[x]&sets[y]
            comparisons.append(dict(a=x,b=y,intersection=sorted(intersection),union=sorted(union),
                jaccard=len(intersection)/len(union) if union else None,
                n_a=len(sets[x]),n_b=len(sets[y])))
        overlap[str(threshold)]=dict(sets={k:sorted(v) for k,v in sets.items()},comparisons=comparisons)
    report=dict(status='complete',screen_tests=len(screen),confirmation_tests=len(confirm),
        discovery_n=16,confirmation_n=48,common_correct_indices=common,
        overlap=overlap,confirmation=[{k:v for k,v in r.items() if k!='rows'} for r in confirm],
        caveat='Overlap restricted to union of top8 positive-damage discovery heads per arm. Empty union undefined. Single-head ablation misses redundancy. Bootstrap intervals are empirical, not guarantees.')
    (ROOT/'summary.json').write_text(json.dumps(report,indent=2))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    maps=[]
    for arm in ARMS:
        rr=[r for r in screen if r['arm']==arm];assert len(rr)==768
        z=np.zeros((48,16))
        for r in rr:z[r['layer'],r['head']]=r['accuracy_drop_pp']
        maps.append(z)
    vmax=max(1,max(float(z.max()) for z in maps));vmin=min(0,min(float(z.min()) for z in maps))
    fig,axs=plt.subplots(1,3,figsize=(11,9),layout='constrained')
    for ax,z,label in zip(axs,maps,LABELS):
        im=ax.imshow(z,vmin=vmin,vmax=vmax,cmap='RdYlBu_r',aspect='auto')
        ax.set(title=label,xlabel='Head index (1-based)',ylabel='Layer (1-based)')
        ax.set_xticks([0,3,7,11,15],[1,4,8,12,16]);ax.set_yticks([0,7,15,23,31,39,47],[1,8,16,24,32,40,48])
    fig.colorbar(im,ax=axs,label='Accuracy drop after single-head ablation (pp)',fraction=.025)
    fig.suptitle('Causal head scan: all 768 heads, each removed in all four loops\nGreedy answer accuracy; 16 paired discovery sequences',fontsize=14)
    fig.savefig(ROOT/'causal-head-scan.png',dpi=180);fig.savefig(ROOT/'causal-head-scan.pdf');plt.close(fig)
    ids=sorted({r['head_id'] for r in confirm})
    if ids:
        lookup={(r['arm'],r['head_id']):r for r in confirm}
        ids.sort(key=lambda h:max(lookup[arm,h]['accuracy_drop_pp'] for arm in ARMS),reverse=True)
        z=np.array([[lookup[arm,h]['accuracy_drop_pp'] for arm in ARMS] for h in ids])
        fig,ax=plt.subplots(figsize=(8,max(4,len(ids)*.34)),layout='constrained')
        im=ax.imshow(z,cmap='RdYlBu_r',vmin=min(-1,float(z.min())),vmax=max(1,float(z.max())),aspect='auto')
        ax.set_xticks(range(3),LABELS);ax.set_yticks(range(len(ids)),[f'Layer {h//16+1}, head {h%16+1}' for h in ids])
        for i in range(len(ids)):
            for j in range(3):ax.text(j,i,f'{z[i,j]:+.1f} pp',ha='center',va='center',color='black')
        ax.set_title('Independent confirmation: which heads actually matter?\nUnion of each arm\'s top8 damage candidates; 48 new sequences',pad=12)
        fig.colorbar(im,ax=ax,label='Accuracy drop (pp)',fraction=.04)
        fig.savefig(ROOT/'confirmed-key-heads.png',dpi=180);fig.savefig(ROOT/'confirmed-key-heads.pdf');plt.close(fig)
    fig,axs=plt.subplots(2,1,figsize=(9,4.8),layout='constrained')
    for ax,c in zip(axs,overlap['10']['comparisons']):
        shared=len(c['intersection']);only_a=c['n_a']-shared;only_b=c['n_b']-shared
        left=0
        for value,label,color in [(only_a,c['a']+' only','#2066a8'),(shared,'Shared','#5f9b62'),(only_b,c['b']+' only','#bd581b')]:
            ax.barh([0],[value],left=left,color=color,label=label)
            if value:ax.text(left+value/2,0,str(value),ha='center',va='center',color='white',fontsize=13)
            left+=value
        jac='undefined (empty sets)' if c['jaccard'] is None else f'{100*c["jaccard"]:.1f}%'
        ax.set_title(f'{c["a"]} vs {c["b"]}: Jaccard overlap = {jac}')
        ax.set_xlim(0,max(1,len(c['union'])));ax.set_yticks([]);ax.set_xlabel('Number of confirmed large-effect heads')
        ax.legend(frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(.5,-.45))
    fig.suptitle('Critical-head overlap within the confirmed candidate pool\nKey criterion: >=10 pp accuracy loss on 48 independent sequences',fontsize=13)
    fig.savefig(ROOT/'critical-head-overlap.png',dpi=180);fig.savefig(ROOT/'critical-head-overlap.pdf');plt.close(fig)
    print(json.dumps(dict(status='complete',screen_tests=len(screen),confirmation_tests=len(confirm),overlap=overlap)),flush=True)

if __name__=='__main__':main()

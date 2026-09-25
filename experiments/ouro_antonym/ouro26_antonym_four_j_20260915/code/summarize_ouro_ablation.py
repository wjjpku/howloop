"""Paired held-out results; all conditions and every selected control reported."""
import json
import numpy as np
from diagnose_ouro_heads import ROOT, ARMS

def rows(path):return [json.loads(x) for x in path.read_text().splitlines()]
def main():
    base={a:{r['i']:r for r in rows(ROOT/a/'collect/results.jsonl')} for a in ARMS}
    common=[i for i in range(16,64) if all(base[a][i]['correct'] for a in ARMS)]
    groups=json.loads((ROOT/'groups.json').read_text())
    report={'common_correct_indices':common,'groups':groups,'arms':{}}
    rng=np.random.default_rng(20260915)
    for a in ARMS:
        assert (ROOT/a/'ablate/complete.json').exists()
        res=rows(ROOT/a/'ablate/results.jsonl')
        arm={'baseline_all64':sum(x['correct'] for x in base[a].values()),
             'baseline_confirmation_correct':sum(base[a][i]['correct'] for i in range(16,64)),
             'n':48,'groups':{}}
        for g in groups:
            rr={r['i']:r for r in res if r['group']==g};assert len(rr)==48
            delta=np.array([int(base[a][i]['correct'])-int(rr[i]['correct']) for i in range(16,64)])
            boot=rng.choice(delta,(10000,48),replace=True).mean(1)*100
            arm['groups'][g]=dict(correct=sum(r['correct'] for r in rr.values()),
                accuracy_drop_pp=float(delta.mean()*100),paired_bootstrap95_pp=np.quantile(boot,[.025,.975]).tolist(),
                breaks=int((delta==1).sum()),repairs=int((delta==-1).sum()),
                common_correct_retained=sum(rr[i]['correct'] for i in common),common_correct_n=len(common))
        report['arms'][a]=arm
    (ROOT/'causal_summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()

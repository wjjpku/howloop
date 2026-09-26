"""Recompute new-submission composition and target-exchange table entries."""
from pathlib import Path
import json, numpy as np
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/composition'
summary=json.loads((BASE/'summary.json').read_text())
reported={'A':(93.64,65.28,74.50,68.12),'C':(83.78,40.98,55.02,28.42),'B':(22.14,10.53,46.22,97.44),'D':(18.39,6.83,28.64,83.58),'E':(20.00,9.45,28.14,91.47)}
rows={}
for model,expected in reported.items():
    ds=[np.load(BASE/f'{model}_fit{fit}.npz') for fit in (1,2)]
    labels=ds[0]['labels'];assert np.array_equal(labels,ds[1]['labels'])
    mask=np.all(np.diff(np.sort(labels,axis=1),axis=1)!=0,axis=1)
    assert mask.sum()==summary[model]['n_distinct_0to4']
    vals=[]
    for seq,target in [('one_one',2),('one_two',3),('two_one',3),('two_two',4)]:
        i=list(ds[0]['sequences']).index(seq)
        v=np.mean([(d['second'][mask,i]==labels[mask,target]).mean() for d in ds])*100
        assert np.isclose(v,summary[model]['sequences'][seq]['mean']*100,atol=1e-9)
        vals.append(float(v))
    assert np.allclose(vals,expected,atol=.006),(model,vals,expected)
    rows[model]={'n':int(mask.sum()),'accuracies_percent':dict(zip(('one_one','one_two','two_one','two_two'),vals))}
switch=json.loads((ROOT/'experiments/target_exchange/graph_summary.json').read_text())
a=switch['A'];target_rates={
    'two_to_one_L2':a['interventions']['two_to_one/L2/pattern']['two']['mean']*100,
    'one_to_two_L2':a['interventions']['one_to_two/L2/pattern']['one']['mean']*100,
}
assert np.isclose(target_rates['two_to_one_L2'],45.4,atol=.05)
assert np.isclose(target_rates['one_to_two_L2'],60.92,atol=.05)
for model,stats in switch.items():
    ds=[np.load(ROOT/f'experiments/target_exchange/graph/{model}_fit{fit}.npz') for fit in (1,2)]
    labels=ds[0]['labels'];mask=(labels[:,0]!=labels[:,1])&(labels[:,0]!=labels[:,2])&(labels[:,1]!=labels[:,2])
    assert mask.sum()==stats['n_distinct']
    for direction,donor,target in [('two_to_one','two',2),('one_to_two','one',1)]:
        key=f'{direction}/L2/pattern';i=list(ds[0]['conditions']).index(key)
        actual=np.mean([(d['predictions'][i,mask]==labels[mask,target]).mean() for d in ds])
        assert np.isclose(actual,stats['interventions'][key][donor]['mean'],atol=1e-12)
out=ROOT/'outputs/audit';out.mkdir(parents=True,exist_ok=True)
(out/'submission.json').write_text(json.dumps({'composition':rows,'target_exchange_percent':target_rates,'scope':'saved predictions and summaries; no GPU rerun'},indent=2)+'\n')
print('Submission composition and target-exchange claims verified.')

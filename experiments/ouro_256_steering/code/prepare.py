import json
from pathlib import Path
P=Path(__file__).resolve().parents[1];old=Path('/data/paperexperiment/ouro_semantic_20260923');pairs=[];sources=[]
for p in old.glob('*.json'):
 d=json.loads(p.read_text())
 if isinstance(d,dict) and isinstance(d.get('pairs'),list):
  pairs.extend(x for x in d['pairs'] if 'base_graph' in x and 'source_graph' in x);sources.append(str(p))
(P/'exclude.json').write_text(json.dumps(dict(pairs=pairs,sources=sources)))
h={'41':[2,10,12,15],'43':[1,6,11,13,15],'47':[0,2,3,4,7,13,15]}
n={l:[i for i in range(16) if i not in hs][:len(hs)] for l,hs in h.items()}
conditions={k:dict(heads=h,calls=[3]) for k in ['selected_rescue_pattern','selected_damage_pattern','selected_rescue_value','selected_unrelated_pattern','selected_wrong_call_pattern','selected_self_pattern','selected_self_output']}
conditions['neighbor_rescue_pattern']=dict(heads=n,calls=[3])
(P/'config.json').write_text(json.dumps(dict(conditions=conditions,generate=['native','J','selected_rescue_pattern','selected_damage_pattern','selected_rescue_value','selected_unrelated_pattern','selected_wrong_call_pattern','neighbor_rescue_pattern']),indent=2))

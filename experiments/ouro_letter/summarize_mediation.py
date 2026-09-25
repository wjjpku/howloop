import collections,json,sys
from pathlib import Path
p=Path(sys.argv[1]);rows=[json.loads(x) for x in (p/'results.jsonl').read_text().splitlines()];g=collections.defaultdict(list)
for x in rows:g[x['scope'],x['condition']].append(x)
out=[]
for (scope,kind),rs in g.items():
 out.append(dict(scope=scope,condition=kind,n=len(rs),correct=sum(r['correct'] for r in rs),mean_probability=sum(r['probability'] for r in rs)/len(rs),max_identity_error=max((r.get('max_logit_error',0) for r in rs))))
(p/'summary.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

import json,sys,collections
from pathlib import Path
p=Path(sys.argv[1]);rows=[json.loads(x) for x in (p/'results.jsonl').read_text().splitlines()];g=collections.defaultdict(list)
for r in rows:g[r['condition']].append(r)
s={k:{'n':len(a),'correct':sum(x['correct'] for x in a),'p':sum(x['probability'] for x in a)/len(a)} for k,a in g.items()}
(p/'summary.json').write_text(json.dumps(s,indent=2))
print(json.dumps(s,indent=2))

import json,sys,collections
from pathlib import Path
p=Path(sys.argv[1]);g=collections.defaultdict(dict)
for line in (p/'results.jsonl').read_text().splitlines():
 r=json.loads(line);g[r['condition']][r['pair']]=r
names=[k[:-15] for k in g if k.endswith('_rescue_pattern')]
for k in names:
 a=g[k+'_rescue_pattern'];b=g[k+'_damage_pattern'];ix=sorted(a.keys()&b.keys())
 if not ix:continue
 nr=sum(a[i]['correct'] for i in ix);nd=sum(b[i]['correct'] for i in ix)
 print(k,'n',len(ix),'rescue',nr,'damage acc',nd,'prob',round(sum(a[i]['probability'] for i in ix)/len(ix),4),round(sum(b[i]['probability'] for i in ix)/len(ix),4))

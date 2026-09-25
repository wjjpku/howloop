import json,collections,re
from pathlib import Path
p=Path(__file__).resolve().parent;m=json.loads((p/'localize_ranked8/manifest.json').read_text());assert m['status']=='complete'
g=collections.defaultdict(list)
for r in map(json.loads,(p/'localize_ranked8/results.jsonl').read_text().splitlines()):g[r['condition']].append(r)
mean=lambda n:sum(r['probability'] for r in g[n])/len(g[n]);rank=[]
for l in [41,43,47]:
 for h in range(16):
  n=f'omitL{l}H{h}';score=mean('triple_rescue_pattern')-mean(n+'_rescue_pattern')+mean(n+'_damage_pattern')-mean('triple_damage_pattern');rank.append((score,l,h))
rank.sort(reverse=True);c={}
def add(n,sites,calls=[1,2,3]):
 heads={}
 for l,h in sites:heads.setdefault(str(l),[]).append(h)
 for k in ['rescue','damage']:c[n+'_'+k+'_pattern']={'heads':heads,'calls':calls}
for k in [1,2,3,4,5,6,8,10,12,16,20,24,32,40,48]:add(f'top{k}heads',[(l,h) for _,l,h in rank[:k]])
# Single heads: check whether necessity ranking conceals bidirectional singleton.
for _,l,h in rank[:12]:add(f'singleL{l}H{h}',[(l,h)])
(p/'compact.json').write_text(json.dumps({'conditions':c,'ranking':rank},indent=2));print(rank[:16])

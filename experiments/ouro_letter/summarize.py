import argparse,json,collections
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args();d=Path(a.directory);rows=[json.loads(s) for s in (d/'results.jsonl').read_text().splitlines()];groups=collections.defaultdict(list)
for x in rows:groups[(tuple(x['site']) if x['site'] else None,x['condition'])].append(x)
result=[]
for (site,condition),r in groups.items():
 n=len(r);counts=[sum(x['prediction']==x['target_ids'][i] for x in r) for i in range(3)]
 result.append(dict(site=site,condition=condition,n=n,base_answer=counts[0],source_answer=counts[1],counterfactual=counts[2],other=n-sum(counts),mean_target_probability=[sum(x['target_probabilities'][i] for x in r)/n for i in range(3)],both_clean_correct=sum(x['base_correct'] and x['source_correct'] for x in r)))
(d/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

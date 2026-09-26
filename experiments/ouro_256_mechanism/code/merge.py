import json,hashlib,sys
from pathlib import Path
P=Path(__file__).resolve().parents[1]
for kind,ncond,script in [('semantic',11,'parallel_semantics.py'),('restore',9,'parallel_restore.py')]:
 if len(sys.argv)>1 and kind not in sys.argv[1:]:continue
 out=P/(kind+'_merged');out.mkdir(exist_ok=True);rows=[];mans=[]
 for i in [0,1]:
  q=P/f'{kind}{i}';m=json.loads((q/'manifest.json').read_text());assert m['status']=='complete' and m['parameters_unchanged'];assert m['code_sha']==hashlib.sha256((P/'code'/script).read_bytes()).hexdigest();assert m['pairs_sha']==hashlib.sha256((P/f'{kind}128_{i}.json').read_bytes()).hexdigest();rows += [json.loads(x) for x in (q/'results.jsonl').read_text().splitlines()];mans.append(m)
 assert len(rows)==256*ncond;assert len({(r['pair'],r['condition']) for r in rows})==len(rows)
 for cond in {r['condition'] for r in rows}:assert {r['pair'] for r in rows if r['condition']==cond}==set(range(256))
 (out/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in sorted(rows,key=lambda r:(r['pair'],r['condition']))));(out/'manifest.json').write_text(json.dumps(dict(status='complete',shards=mans,n=256,rows=len(rows),parameters_unchanged=True),indent=2))
 print(kind,len(rows))

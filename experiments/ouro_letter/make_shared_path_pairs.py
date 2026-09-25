# Graph construction depends only on task structure, never model outcomes.
import json,random,argparse
from pathlib import Path
from transformers import AutoTokenizer
# Reuse the validated renderer without executing make_pairs.py's CLI.
ns={};code=Path(__file__).with_name('make_pairs.py').read_text();exec(code[:code.index('a=argparse.ArgumentParser()')],ns)
N,walk,enc=ns['N'],ns['walk'],ns['enc']
a=argparse.ArgumentParser();a.add_argument('--out',required=True);a.add_argument('--seed',type=int,default=2026092501);a.add_argument('--pairs',type=int,default=8);a.add_argument('--exclude',nargs='*',default=[]);args=a.parse_args()
tok=AutoTokenizer.from_pretrained('/data/wujiaju/models/Ouro-2.6B',local_files_only=True);r=random.Random(args.seed);out=[];seen=set();attempt=0
for path in args.exclude:
 for x in json.loads(Path(path).read_text())['pairs']:
  for f in ['base_graph','source_graph']:seen.add(tuple(x[f][n] for n in N))
while len(out)<args.pairs:
 attempt+=1;c=N.copy();r.shuffle(c);s=dict(zip(c,c[1:]+c[:1]));bc=c[:8]+[c[9],c[8]];b=dict(zip(bc,bc[1:]+bc[:1]));ss=c[0];sb=r.choice(N);ys=walk(s,ss,8);cf=walk(b,ss,8);yb=walk(b,sb,8)
 if len({yb,ys,cf})<3:continue
 sig=lambda g:tuple(g[n] for n in N)
 if sig(b) in seen or sig(s) in seen:continue
 assert all(walk(s,ss,k)==walk(b,ss,k) for k in range(8))
 assert cf==b[walk(s,ss,7)] and cf!=ys
 order=N.copy();r.shuffle(order);eb=enc(tok,b,sb,order);es=enc(tok,s,ss,order)
 if len(eb['ids'])!=len(es['ids']) or eb['slots']!=es['slots']:continue
 tids=[tok.encode(x,add_special_tokens=False)[0] for x in [yb,ys,cf]]
 if len(set(tids))<3:continue
 seen.update([sig(b),sig(s)]);out.append(dict(index=len(out),base=eb,source=es,base_graph=b,source_graph=s,base_start=sb,source_start=ss,base_answer=yb,source_answer=ys,counterfactual=cf,source_last_holder=walk(s,ss,7),secondary_start_counterfactual=cf,target_ids=tids,shared_prefix_hops=7))
p=Path(args.out);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(dict(seed=args.seed,attempts=attempt,construction='same source-start path through seven edges, different eighth recipient; base query differs',pairs=out),indent=2));print(json.dumps(dict(pairs=len(out),attempts=attempt)))

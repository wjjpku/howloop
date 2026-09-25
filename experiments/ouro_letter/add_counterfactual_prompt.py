import json,sys
from pathlib import Path
from transformers import AutoTokenizer
ns={};s=Path(__file__).with_name('make_pairs.py').read_text();exec(s[:s.index('a=argparse.ArgumentParser()')],ns)
tok=AutoTokenizer.from_pretrained('/data/wujiaju/models/Ouro-2.6B',local_files_only=True)
for path in sys.argv[1:]:
 p=Path(path);d=json.loads(p.read_text())
 for pair in d['pairs']:
  order=[line.split(' passes the letter to ')[0] for line in pair['base']['prompt'].splitlines() if ' passes the letter to ' in line]
  cf=ns['enc'](tok,pair['base_graph'],pair['source_start'],order);assert cf['slots']==pair['base']['slots'];pair['counterfactual_run']=cf
 out=p.with_name(p.stem+'_with_cf.json');out.write_text(json.dumps(d,indent=2));print(out)

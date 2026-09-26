from pathlib import Path
import json,hashlib,numpy as np
P=Path(__file__).resolve().parents[1];B=Path('/data/wujiaju/n10_migration_20260923');checks=[]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for n in 'ABCDE':
 assert (P/f'local/{n}/complete.json').exists()
 for hop in [1,2]:
  for f in [1,2]:
   sub=P/f'local/{n}/hop{hop}_seed{f}';id=json.loads((sub/'training_identity.json').read_text());old=json.loads((B/f'local/{n}/hop{hop}_seed{f}/training_identity.json').read_text());assert id['input_stream_sha256']==old['input_stream_sha256'];assert id['backbone_unchanged'];assert (sub/'evaluation/summary.json').exists()
 for f in [1,2]:
  meta=json.loads((P/f'graph/{n}_fit{f}.json').read_text());assert meta['weights_unchanged'];d=np.load(P/f'graph/{n}_fit{f}.npz');names=list(d['conditions']);count=0
  for i,k in enumerate(names):
   if str(k).endswith('/query_key'):
    j=names.index(str(k).replace('/query_key','/pattern'));assert np.array_equal(d['predictions'][i],d['predictions'][j]);count+=1
  c=json.loads((P/f'composition/{n}_fit{f}.json').read_text());assert c['first_step_matches_previous'] and c['weights_unchanged'];checks.append({'model':n,'fit':f,'qk_pattern_checks':count})
 for h in range(4):
  folder=P/f'battery/{n}/head{h}/confirmation';mf=json.loads((folder/'manifest.json').read_text());assert mf['status']=='complete' and mf['parameters_unchanged'];assert sha(folder/'events.jsonl.gz')==mf['predictions_sha256']
 ref=P/f'reference/{n}/confirmation';mf=json.loads((ref/'manifest.json').read_text());assert mf['status']=='complete';assert sha(ref/'events.jsonl.gz')==mf['predictions_sha256']
(P/'AUDIT.json').write_text(json.dumps({'status':'complete','training_fits':20,'matrix_fits':10,'composition_fits':10,'dense_head_batteries':20,'matched_reference_batteries':5,'checks':checks,'source_hashes':{p.name:sha(p) for p in (P/'code').glob('*.py')}},indent=2));print('All checks passed.')

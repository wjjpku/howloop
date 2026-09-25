from pathlib import Path
import json,random,hashlib,torch,shutil
R=Path(__file__).resolve().parent;B=Path('/data/wujiaju/n10_migration_20260923')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
d=json.loads((B/'datasets.json').read_text());used={tuple(g) for gs in d.values() for g in gs}
for panel in ['discovery','confirmation','smoke']:
 for g in d[panel]:
  for current in range(10):
   wrong=(current+3)%10
   for address,repl in [(i,(i+1)%10) for i in range(10)]+[(g[wrong],(g[wrong]+1)%10 if (g[wrong]+1)%10!=wrong else (g[wrong]+2)%10)]:
    h=g.copy();h[address],h[repl]=h[repl],h[address];used.add(tuple(h))
reserve=torch.load(B/'locks/donors.pt',weights_only=False,map_location='cpu')['successors'].tolist()
pool=sorted(set(map(tuple,reserve))-used);random.Random(2026092403).shuffle(pool);assert len(pool)>=1024
base=[list(g) for g in pool[:512]];donor=[list(g) for g in pool[512:1024]]
assert not (set(map(tuple,base+donor))&used)
(R/'datasets.json').write_text(json.dumps({'confirmation':base,'corrupted':donor}))
cp=B/'backbones/local_control_L6_seed6/best.pt';ctrl=B/'local/A/controllers.pt'
modelsummary=json.loads((cp.parent/'summary.json').read_text());assert sha(B/'locks/donors.pt') in modelsummary['extra_excluded_locks'].values()
for fit in [1,2]:
 identity=json.loads((B/f'local/A/hop1_seed{fit}/training_identity.json').read_text());print('J identity',fit,identity.keys())
s=json.loads((B/'local/A/head2/confirmation/manifest.json').read_text());assert s['backbone_sha256']==sha(cp) and s['controller_sha256']==sha(ctrl)
shutil.copytree(B/'code',R/'code',dirs_exist_ok=True)
c=(R/'code/evaluate_battery.py').read_text()
a=c.index('                    donor_graph = graph.clone()',c.index('# Related-graph donors'));b=c.index('                    donor_bundle =',a)
c=c[:a]+'''                    donor_graph = torch.tensor(datasets['corrupted'][first:first+len(graph_list)],device=device).repeat_interleave(cfg.node_count,0)
'''+c[b:]
a=c.index('                    donor_graph = graph.clone()',c.index('# Replace graph value vectors'));b=c.index('                    donor_run =',a)
c=c[:a]+'''                    donor_graph = torch.tensor(datasets['corrupted'][first:first+len(graph_list)],device=device).repeat_interleave(cfg.node_count,0)
'''+c[b:]
(R/'code/evaluate_battery.py').write_text(c)
manifest={'checkpoint':str(cp),'checkpoint_sha256':sha(cp),'checkpoint_step':s['backbone_step'],'controllers':str(ctrl),'controller_sha256':sha(ctrl),'head':2,'control_head':3,'fits':[1,2],'seed':2026092403,'fresh_raw_graphs':512,'fresh_corrupted_graphs':512,'starts':10,'unused_reserved_pool_size':len(pool),'training_exclusion_lock_sha256':sha(B/'locks/donors.pt'),'datasets_sha256':sha(R/'datasets.json'),'protocol':'New graph identities sampled without model-outcome inspection from unused pre-training-excluded reserve; arbitrary independent corrupted graph pairs, aligned node slots. Head 2 fixed from prior discovery; control 3 fixed as next head. Related to original holdout graphs by generation; not an independent model replication.'}
(R/'MANIFEST.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest))

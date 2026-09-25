from pathlib import Path
import json,random,hashlib,shutil,subprocess,sys,os,time
import torch
R=Path(__file__).resolve().parent;B=Path('/data/wujiaju/n10_migration_20260923');old=Path('/data/wujiaju/n10_fig4_fresh_20260924')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
d=json.loads((B/'datasets.json').read_text());used={tuple(g) for gs in d.values() for g in gs}
for panel in ['discovery','confirmation','smoke']:
 for g in d[panel]:
  for current in range(10):
   wrong=(current+3)%10
   for address,repl in [(i,(i+1)%10) for i in range(10)]+[(g[wrong],(g[wrong]+1)%10 if (g[wrong]+1)%10!=wrong else (g[wrong]+2)%10)]:
    h=g.copy();h[address],h[repl]=h[repl],h[address];used.add(tuple(h))
for gs in json.loads((old/'datasets.json').read_text()).values():used.update(map(tuple,gs))
pool=sorted(set(map(tuple,torch.load(B/'locks/donors.pt',weights_only=False)['successors'].tolist()))-used)
random.Random(2026092404).shuffle(pool)
data={'confirmation':pool[:512],'corrupted':pool[512:1024]};(R/'datasets.json').write_text(json.dumps(data))
shutil.copytree(old/'code',R/'code',dirs_exist_ok=True)
selection={'selected':['A','C'],'main':'A','contrast':'C','seed':2026092404,'basis':'Historical confirmation screening, not the new confirmation outcomes. A has strongest complete chain; C has near-perfect J behavior with weaker mediation. All five screening results retained.','pool':len(pool),'dataset_sha256':sha(R/'datasets.json'),'models':{}}
for name,seed in [('A',6),('C',4)]:
 cp=B/f'backbones/local_control_L6_seed{seed}/best.pt';ctrl=B/f'local/{name}/controllers.pt';head=json.loads((B/f'local/{name}/head_selection.json').read_text())['locked_head'];mf=json.loads((B/f'local/{name}/head{head}/confirmation/manifest.json').read_text())
 assert sha(cp)==mf['backbone_sha256'] and sha(ctrl)==mf['controller_sha256']
 assert sha(B/'locks/donors.pt') in json.loads((cp.parent/'summary.json').read_text())['extra_excluded_locks'].values()
 for f in [1,2]:
  ident=json.loads((B/f'local/{name}/hop1_seed{f}/training_identity.json').read_text())
  assert str(B/'locks/donors.pt') in str(ident['arguments']['train_exclude_locks'])
 selection['models'][name]={'checkpoint':str(cp),'controllers':str(ctrl),'head':head,'manifest':mf}
(R/'SELECTION_LOCK.json').write_text(json.dumps(selection,indent=2))
for name,m in selection['models'].items():
 mf=m['manifest'];cmd=[sys.executable,'-u',str(R/'code/evaluate_battery.py'),'--looplus',str(R/'code'),'--checkpoint',m['checkpoint'],'--checkpoint-sha256',mf['backbone_sha256'],'--checkpoint-step',str(mf['backbone_step']),'--controllers',m['controllers'],'--controller-sha256',mf['controller_sha256'],'--datasets',str(R/'datasets.json'),'--family','N10_selected_fresh_confirmation','--backbone-name',name,'--head',str(m['head']),'--panel','confirmation','--seeds','1','2','--out',str(R/name/'evaluation')]
 p=subprocess.Popen(cmd);(R/'RUN.json').write_text(json.dumps({'name':name,'pid':p.pid,'worker_pid':os.getpid(),'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'command':cmd,'state':'running'}));rc=p.wait();assert rc==0
(R/'RUN.json').write_text(json.dumps({'state':'complete','models':['A','C']}))

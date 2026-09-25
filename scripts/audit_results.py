#!/usr/bin/env python3
"""Recompute statistics from archived sample-level results; never edits evidence."""
from pathlib import Path
import argparse,csv,json,os,shutil,subprocess,sys,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'outputs/audit');a=p.parse_args();O=a.output.resolve();O.mkdir(parents=True,exist_ok=True)
checks={}
def same(a,b):
 if isinstance(a,dict):
  assert a.keys()==b.keys(),(a.keys(),b.keys())
  for k in a:same(a[k],b[k])
 elif isinstance(a,list):
  assert len(a)==len(b)
  for x,y in zip(a,b):same(x,y)
 elif isinstance(a,float):assert np.isclose(a,b,rtol=1e-9,atol=1e-10,equal_nan=True),(a,b)
 else:assert a==b,(a,b)
def run(src,dest,args=(),rewrite=None):
 dest.mkdir(parents=True,exist_ok=True);script=dest/src.name;s=src.read_text();script.write_text(rewrite(s) if rewrite else s)
 proc=subprocess.run([sys.executable,str(script),*map(str,args)],env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),capture_output=True,text=True)
 (dest/'run.log').write_text(proc.stdout+proc.stderr)
 if proc.returncode:raise RuntimeError((proc.stdout+proc.stderr)[-3000:])
# Full graph-cluster bootstrap from hash-verified per-example records, both selected backbones.
src=ROOT/'experiments/graph_mechanism';dest=O/'graph_mechanism'
for name in ['A','C']:shutil.copytree(src/name,dest/name,dirs_exist_ok=True)
run(src/'analyze.py',dest,['--names','AC'])
x=json.loads((dest/'analysis/summary.json').read_text());ref=json.loads((src/'analysis/summary.json').read_text())
for name in x['results']:same(x['results'][name],ref['results'][name])
checks['graph_fresh_bootstrap_matches']=True
# Rebuild the original five-backbone screening independently of the fresh A/C set.
src=ROOT/'experiments/n10/local';dest=O/'screening'
for name in 'ABCDE':
 head=json.loads((src/name/'head_selection.json').read_text())['locked_head']
 shutil.copytree(src/name/f'head{head}/confirmation',dest/name/'evaluation/confirmation',dirs_exist_ok=True)
run(ROOT/'experiments/graph_mechanism/screening/analyze.py',dest,['--names','ABCDE'])
x=json.loads((dest/'analysis/summary.json').read_text());ref=json.loads((ROOT/'experiments/graph_mechanism/screening/analysis/summary.json').read_text())
for name in x['results']:same(x['results'][name],ref['results'][name])
checks['five_backbone_screening']=True
# Ouro semantic route/content and block/restore: all pairs retained.
base=ROOT/'experiments/ouro_letter'
for directory,script in [('parallel_confirmation64','analyze_parallel_semantics.py'),('parallel_restore64','analyze_parallel_restore.py')]:
 source=base/'parallel_section'/directory;dest=O/directory;shutil.copytree(source,dest,dirs_exist_ok=True)
 shutil.copy2(base/'uncertainty.py',dest/'uncertainty.py');ref=json.loads((source/'analysis.json').read_text());run(base/script,dest,[dest]);same(json.loads((dest/'analysis.json').read_text()),ref);checks[directory]=True
# The broad 8192-graph cohort and all 101 extended parity lengths, including duplicate checks.
src=ROOT/'experiments/long_range';dest=O/'long_range';shutil.copytree(src/'results',dest/'results',dirs_exist_ok=True)
old=ROOT/'experiments/graph_continuation/raw'
run(src/'analyze.py',dest,rewrite=lambda s:s.replace('/Users/jiaju/Documents/looped_transformer_paper_repro_20260920/sec09_coverage/horizon_decomposition_20260921/raw',str(old)))
same(json.loads((src/'results/summary.json').read_text()),json.loads((dest/'results/summary.json').read_text()));checks['long_range_all_counts']=True
# Paired Ouro antonym counts at step0/500, template0. Score parsed pairs, not EOS.
ref=json.loads((ROOT/'data/plot/ouro_fixed4.json').read_text());panels={};keys=[]
for family,rel in [('final_only','ouro26_antonym_full_20260915/run'),('stepwise','ouro26_stepwise_pair_control_20260915')]:
 file=ROOT/'vendor/remote'/rel/'shared_j_lora128_k18/eval_outputs.jsonl';rows=[json.loads(s) for s in file.read_text().splitlines()];panels[family]={}
 for step in [0,500]:
  sub=[r for r in rows if r['step']==step and r['template']==0];assert len(sub)==512
  key=sorted((r['k'],r['sequence'],tuple(r['expected'])) for r in sub);assert len(set(key))==512;keys.append(key)
  for k in range(1,9):
   rr=[r for r in sub if r['k']==k];assert len(rr)==64
   count=sum(r['parsed']==r['expected'] for r in rr);expected=ref[family]['counts'][f't0_s{step}_k{k}']['correct'];assert count==expected
   panels[family][f's{step}_k{k}']=count
assert all(k==keys[0] for k in keys);checks['ouro_antonym_pairing_and_counts']=True
(O/'ouro_antonym_counts.json').write_text(json.dumps(panels,indent=2))
# KG cells are the completed length32 checkpoint's final evaluation row.
cap=json.loads((ROOT/'data/plot/capacity.json').read_text());kg={}
for label,key in [('Rank-48 affine','lora_r48'),('Dense affine','dense'),('MLP','mlp_256'),('Attention','attention')]:
 d=json.loads((ROOT/'experiments/kg/results_5k'/key/'results.json').read_text());row=d['heatmap'][-1];assert row['train_length']==32
 # Archived values have already been rounded to four decimal places, so compare at half a percentage hundredth.
 vals=np.array(row['accuracy'])[[1,3,5,7]]*100;assert np.allclose(vals,cap['KG_128x16'][label],atol=.006)
 kg[label]=vals.tolist()
checks['kg_last_curriculum_values']=True
(O/'kg_values.json').write_text(json.dumps(kg,indent=2))
# The three long parity curves have every length and no silent denominator mixing.
rows=list(csv.DictReader((ROOT/'data/plot/parity_long500/registered_accuracy.csv').open()));parity=[]
for seed in range(3):
 for variant in ['raw','J']:
  rr=sorted([r for r in rows if int(r['seed'])==seed and r['variant']==variant],key=lambda r:int(r['length']));assert [int(r['length']) for r in rr]==list(range(1,501))
  for lo,hi in [(20,40),(41,100),(101,500)]:
   vals=[float(r['exact_match']) for r in rr if lo<=int(r['length'])<=hi];parity.append(dict(seed=seed,variant=variant,min_length=lo,max_length=hi,accuracy=float(np.mean(vals)),n_per_length=64))
(O/'parity_accuracy.json').write_text(json.dumps(parity,indent=2));checks['parity_length_coverage']=True
(O/'audit_results.json').write_text(json.dumps({'checks':checks,'all_passed':all(checks.values()),'scope':'reaggregation of saved outputs; not retraining'},indent=2));print(json.dumps(checks,indent=2))

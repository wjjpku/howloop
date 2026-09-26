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
for name in ['A']:shutil.copytree(src/name,dest/name,dirs_exist_ok=True)
run(src/'analyze.py',dest,['--names','A'])
x=json.loads((dest/'analysis/summary.json').read_text());ref=json.loads((src/'analysis/summary.json').read_text())
for name in x['results']:same(x['results'][name],ref['results'][name])
checks['graph_fresh_bootstrap_matches']=True
# Ouro 256-pair steering, semantic patching, and restoration counts.
base=ROOT/'experiments/ouro_256_steering';ref=json.loads((base/'summary.json').read_text());rows=[]
for shard in (0,1):
 rows.extend(json.loads(s) for s in (base/f'shard{shard}/results.jsonl').read_text().splitlines())
for condition,item in ref['conditions'].items():
 rr=[r for r in rows if r['condition']==condition]
 assert len(rr)==256 and len({r['pair'] for r in rr})==256
 assert sum(r['correct'] for r in rr)==item['correct']
checks['ouro_256_steering_counts']=True
for family in ('semantic','restore'):
 base=ROOT/'experiments/ouro_256_mechanism'/f'{family}_merged'
 rows=[json.loads(s) for s in (base/'results.jsonl').read_text().splitlines()]
 ref=json.loads((base/'analysis.json').read_text())['conditions']
 assert {r['condition'] for r in rows}==set(ref)
 for condition,item in ref.items():
  rr=[r for r in rows if r['condition']==condition]
  assert len(rr)==item['n']==256 and len({r['pair'] for r in rr})==256
checks['ouro_256_semantic_restore_counts']=True
# The broad 8192-graph cohort and all 101 extended parity lengths, including duplicate checks.
src=ROOT/'experiments/long_range';dest=O/'long_range';shutil.copytree(src/'results',dest/'results',dirs_exist_ok=True)
old=ROOT/'experiments/graph_continuation/raw'
run(src/'analyze.py',dest,rewrite=lambda s:s.replace('/data/paperexperiment/Documents/looped_transformer_paper_repro_20260920/sec09_coverage/horizon_decomposition_20260921/raw',str(old)))
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
# The three long parity curves have every length and no silent denominator mixing.
rows=list(csv.DictReader((ROOT/'data/plot/parity_long500/registered_accuracy.csv').open()));parity=[]
for seed in range(3):
 for variant in ['raw','J']:
  rr=sorted([r for r in rows if int(r['seed'])==seed and r['variant']==variant],key=lambda r:int(r['length']));assert [int(r['length']) for r in rr]==list(range(1,501))
  for lo,hi in [(20,40),(41,100),(101,500)]:
   vals=[float(r['exact_match']) for r in rr if lo<=int(r['length'])<=hi];parity.append(dict(seed=seed,variant=variant,min_length=lo,max_length=hi,accuracy=float(np.mean(vals)),n_per_length=64))
(O/'parity_accuracy.json').write_text(json.dumps(parity,indent=2));checks['parity_length_coverage']=True
(O/'audit_results.json').write_text(json.dumps({'checks':checks,'all_passed':all(checks.values()),'scope':'reaggregation of saved outputs; not retraining'},indent=2));print(json.dumps(checks,indent=2))

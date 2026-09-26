#!/usr/bin/env python3
"""Materialize portable run code without changing the archived source snapshots.

Filesystem roots/interpreter paths are rewritten; the letter-walk trainer also
gets a checkpoint-stop flag that preserves the original 500-step LR schedule. Archived result flags
are deliberately not copied into a new training run.
"""
from pathlib import Path
import argparse,hashlib,json,shutil,sys
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work',type=Path,required=True);p.add_argument('--mode',choices=['replay','train'],default='replay');a=p.parse_args();W=a.work.resolve()
if W.exists() and any(W.iterdir()):raise SystemExit('Choose an empty work directory; existing runs are never overwritten.')
W.mkdir(parents=True,exist_ok=True);records=[]
maps=[(ROOT/'vendor/remote',W),
(ROOT/'experiments/n10/code',W/'n10_migration_20260923/code'),
(ROOT/'experiments/n10/locks',W/'n10_migration_20260923/locks'),
(ROOT/'experiments/graph_mechanism/code',W/'n10_selected_mechanism_20260924/code'),
(ROOT/'experiments/graph_mechanism_discovery/code',W/'n10_fig4_fresh_20260924/code'),
(ROOT/'experiments/ouro_antonym',W),
(ROOT/'experiments/kg',W/'kg_curriculum_controllers'),
(ROOT/'experiments/ouro_letter',W/'ouro_mechanism_20260923'),
(ROOT/'experiments/long_range',W/'fig6_retest_20260924'),
(ROOT/'experiments/pca',W/'n10_hop_pca_20260924'),
(ROOT/'experiments/composition',W/'continuous_composition_20260925'),
(ROOT/'experiments/target_exchange',W/'paper_strengthening_20260925/code')]
text_suffix={'.py','.sh','.json','.yaml','.yml','.md','.txt'}
def write(src,dst):
 # Preserve historical evidence in the repository; run dirs get only inputs/code.
 dst.parent.mkdir(parents=True,exist_ok=True);old=src.read_bytes();new=old
 if src.suffix in text_suffix:
  s=old.decode();s=s.replace('/data/wujiaju/.venvs/loopreasoner/bin/python',sys.executable)
  s=s.replace('/data/wujiaju',str(W))
  if src == ROOT/'vendor/remote/letter_walk_native_20260914/code/train_full.py':
   # The paper uses step200 of a schedule configured for500, not a200-step schedule.
   s=s.replace("p.add_argument('--resume',action='store_true');a=p.parse_args()", "p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int);a=p.parse_args()")
   s=s.replace("save(step,baseline,counts);task_eval(step)", "save(step,baseline,counts);task_eval(step)\n            if a.stop_after is not None and step>=a.stop_after:emit(dict(event='stopped_at_requested_checkpoint',step=step,planned_steps=a.steps));return")
  new=s.encode()
 dst.write_bytes(new)
 records.append({'source':str(src.relative_to(ROOT)),'target':str(dst.relative_to(W)),'original_sha256':hashlib.sha256(old).hexdigest(),'runtime_sha256':hashlib.sha256(new).hexdigest()})
for source,dest in maps:
 if not source.is_dir():raise FileNotFoundError(source)
 for src in sorted(source.rglob('*')):
  if not src.is_file() or '__pycache__' in src.parts or src.name.startswith('._'):continue
  rel=src.relative_to(source)
  is_code=src.suffix in {'.py','.sh','.yaml','.yml','.toml'}
  is_input=(source.name=='locks' or 'locks' in rel.parts or src.name in {'datasets.json','config.json','configuration.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.json','merges.txt'} or 'data' in rel.parts or 'configs' in rel.parts or 'pairs' in src.name or (len(rel.parts)==1 and src.suffix=='.json' and ('confirmation' in src.name or 'discovery' in src.name)))
  if is_code or is_input:write(src,dest/rel)
for srcdir,dstdir in [('graph_mechanism','n10_selected_mechanism_20260924'),('graph_mechanism_discovery','n10_fig4_fresh_20260924')]:
 for name in ['datasets.json','SELECTION_LOCK.json']:
  src=ROOT/'experiments'/srcdir/name
  if src.exists():write(src,W/dstdir/name)
for name in ['MANIFEST.json','datasets.json','locks_manifest.json']:
 write(ROOT/'experiments/n10'/name,W/'n10_migration_20260923'/name)
write(ROOT/'experiments/target_exchange/graph_data.json',W/'paper_strengthening_20260925/graph_data.json')
# Extension cohort lives in its own directory, not the original four-seed queue.
ext=W/'n10_migration_20260923/trajectory_seed_extension_20260924'
shutil.copytree(W/'n10_migration_20260923/code',ext/'code',dirs_exist_ok=True)
write(ROOT/'experiments/n10/trajectory_seed_extension_20260924/worker.py',ext/'worker.py')
# Ouro scripts depend on completed backbone manifests even for frozen evaluation.
for rel in ['ouro26_letter_full_20260915/run/manifest_train.json','ouro26_antonym_full_20260915/run/manifest_train.json']:
 source=ROOT/'vendor/remote'/rel
 if a.mode=='replay' and source.exists():write(source,W/rel)
(W/'runtime_manifest.json').write_text(json.dumps({'root':str(W),'rewrites':'data root/interpreter; letter-walk adds stop-after-checkpoint without changing planned LR schedule','files':records},indent=2))
print('Prepared',len(records),'files at',W)
print('No training was launched. Fetch checkpoints or run documented training stages next.')

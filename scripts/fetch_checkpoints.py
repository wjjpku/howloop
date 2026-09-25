#!/usr/bin/env python3
"""Fetch selected original weights from an authorized SSH archive and verify SHA256."""
from pathlib import Path
import argparse,hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work',required=True,type=Path);p.add_argument('--host',default='A100-80G-34200');p.add_argument('--group',required=True,choices=['n10','parity','graph','ouro','kg','all']);p.add_argument('--download',action='store_true',help='Without this flag, print a size manifest only.')
a=p.parse_args();prefix={'n10':'n10_migration','parity':'parity_input','graph':'paper2027_confirmatory','ouro':'ouro26_','kg':'kg-fj-'}
rows=[r for r in json.loads((ROOT/'provenance/checkpoints.json').read_text()) if r['exists'] and (a.group=='all' or (r['relative_path'].startswith(prefix[a.group]) or (a.group=='ouro' and r['relative_path'].startswith('models/Ouro-2.6B/'))))]
print(json.dumps({'files':len(rows),'bytes':sum(r['bytes'] for r in rows),'paths':[r['relative_path'] for r in rows]},indent=2))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
if a.download:
 for r in rows:
  dest=a.work.resolve()/r['relative_path'];dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists():
   if sha(dest)==r['sha256']:continue
   raise SystemExit(f'Existing file differs; refusing to overwrite: {dest}')
  tmp=dest.with_name(dest.name+'.partial')
  subprocess.run(['rsync','-az','--partial',a.host+':'+r['source'],str(tmp)],check=True)
  if sha(tmp)!=r['sha256']:raise SystemExit(f'Hash mismatch: {tmp}')
  tmp.rename(dest)
 print('All requested checkpoint hashes verified.')

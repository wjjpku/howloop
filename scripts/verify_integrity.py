#!/usr/bin/env python3
"""Verify frozen inputs, source snapshots, and complete manuscript figure coverage."""
from pathlib import Path
import hashlib,json,re,sys
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'provenance/SHA256SUMS.json').read_text());failed=[]
for name,expected in manifest.items():
 p=ROOT/name
 if not p.is_file():failed.append((name,'missing'));continue
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 if h.hexdigest()!=expected:failed.append((name,'hash mismatch'))
figs=json.loads((ROOT/'provenance/figures.json').read_text());used=set(re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',(ROOT/'paper/main.tex').read_text()))
assert used=={x['paper_asset'] for x in figs},'Incomplete paper figure mapping'
for row in figs:
 for name in row['inputs']+[row['script']]:
  assert (ROOT/name).exists(),name
assert len(used)==15
if failed:
 print(json.dumps(failed,indent=2));raise SystemExit(1)
print(f'Verified {len(manifest)} hashes and all {len(used)} manuscript figure assets.')

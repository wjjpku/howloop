import json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[1]
old=Path('/data/paperexperiment/ouro_expansion_20260926')
a=json.loads((old/'exclude.json').read_text())['pairs']+json.loads((old/'pairs256.json').read_text())['pairs']
(P/'exclude.json').write_text(json.dumps(dict(pairs=a)))

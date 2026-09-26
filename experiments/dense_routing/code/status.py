import json,subprocess
from pathlib import Path
P=Path(__file__).resolve().parents[1];out={}
for n in ['A','C','B','D','E']:
 root=P/'local'/n
 out[n]={'trained':len(list(root.glob('hop*/summary.json'))),'matrix_fits':len(list((P/'graph').glob(n+'_fit*.json'))),'composition_fits':len(list((P/'composition').glob(n+'_fit[12].json'))),'battery_heads':len(list((P/'battery'/n).glob('head*/confirmation/manifest.json'))),'complete':(root/'complete.json').exists()}
print(json.dumps(out,indent=2))

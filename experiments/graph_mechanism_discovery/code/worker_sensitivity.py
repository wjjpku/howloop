import json,subprocess,sys,hashlib
from pathlib import Path
import torch
R=Path(__file__).resolve().parents[1];C=R/'code';ck=R/'backbones/local_control_L6_seed6/best.pt';out=R/'regularized/A';out.mkdir(parents=True,exist_ok=True);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();step=torch.load(ck,weights_only=False)['step'];head=json.loads((R/'local/A/head_selection.json').read_text())['locked_head']
def run(cmd):print(json.dumps(list(map(str,cmd))),flush=True);subprocess.run(list(map(str,cmd)),cwd=C,check=True)
if not (out/'manifest.json').exists():run([sys.executable,'-u','train_regularized.py','--looplus',C,'--checkpoint',ck,'--checkpoint-sha256',sha(ck),'--checkpoint-step',step,'--out',out,'--seeds',1,2,'--rank',48])
run([sys.executable,'-u','evaluate_battery.py','--looplus',C,'--checkpoint',ck,'--checkpoint-sha256',sha(ck),'--checkpoint-step',step,'--controllers',out/'controllers.pt','--datasets',R/'datasets.json','--family','N10_regularized_SVD48','--backbone-name','A','--head',head,'--panel','confirmation','--seeds',1,2,'--out',out/'evaluation'])

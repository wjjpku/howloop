import sys,json,subprocess,hashlib
from pathlib import Path
import torch
P=Path(__file__).resolve().parents[1];B=Path('/data/wujiaju/n10_migration_20260923');torch.set_num_threads(4)
for name in sys.argv[1:]:
 seed={'A':6,'B':3,'C':4,'D':5,'E':7}[name];cp=B/f'backbones/local_control_L6_seed{seed}/best.pt';head=json.loads((B/f'local/{name}/head_selection.json').read_text())['locked_head'];step=torch.load(cp,map_location='cpu',weights_only=False)['step'];ctrl=B/f'local/{name}/controllers.pt';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 subprocess.run(list(map(str,[sys.executable,'-u','/data/wujiaju/n10_fig4_fresh_20260924/code/evaluate_battery.py','--looplus',B/'code','--checkpoint',cp,'--checkpoint-sha256',sha(cp),'--checkpoint-step',step,'--controllers',ctrl,'--controller-sha256',sha(ctrl),'--rank',48,'--datasets','/data/wujiaju/n10_selected_mechanism_20260924/datasets.json','--family','N10_lowrank_paired_reference','--backbone-name',name,'--head',head,'--panel','confirmation','--seeds',1,2,'--out',P/'reference'/name])),check=True)

 for hop in [1,2]:
  for fit in [1,2]:
   sub=P/f'local/{name}/hop{hop}_seed{fit}'
   if not (sub/'summary.json').exists():continue
   subprocess.run(list(map(str,[sys.executable,'-u',P/'code/train_dense.py','evaluate','--checkpoint',cp,'--controller',sub/'best_controller.pt','--locked-test',B/'locks/confirmation.pt','--out-dir',sub/'evaluation','--test-seed',20260923,'--batch-size',250])),check=True)

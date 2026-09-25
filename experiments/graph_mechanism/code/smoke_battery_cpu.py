import subprocess,sys,hashlib,json
from pathlib import Path
import torch
R=Path(__file__).resolve().parents[1];torch.set_num_threads(4)
cp=R/'smoke_snapshot/best.pt';state=torch.load(R/'smoke_s1_cpu/best_controller.pt',weights_only=False)['controller_state_dict'];w=torch.diag(state['diagonal'])+state['A']@state['B'];maps={}
for seed in [1,2]:
 for name in ['one','two']:maps[f'seed{seed}_J_{name}_beh_rank48']={'weight':w,'bias':state['bias']}
p=R/'smoke_maps.pt';torch.save(maps,p)
# Duplicated maps are implementation fixtures only, never experiment fits.
command=[sys.executable,'-u',str(R/'code/evaluate_battery.py'),'--looplus',str(R/'code'),'--checkpoint',str(cp),'--checkpoint-sha256',hashlib.sha256(cp.read_bytes()).hexdigest(),'--checkpoint-step',str(torch.load(cp,weights_only=False)['step']),'--controllers',str(p),'--datasets',str(R/'datasets.json'),'--family','implementation_fixture_only','--backbone-name','smoke','--head','0','--panel','smoke','--seeds','1','--out',str(R/'smoke_battery_cpu'),'--device','cpu']
subprocess.run(command,check=True)

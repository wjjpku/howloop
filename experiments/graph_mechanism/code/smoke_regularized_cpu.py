import hashlib,subprocess,sys
from pathlib import Path
import torch
R=Path(__file__).resolve().parents[1];ck=R/'smoke_snapshot/best.pt';step=torch.load(ck,weights_only=False)['step'];subprocess.run([sys.executable,'-u',str(R/'code/train_regularized.py'),'--looplus',str(R/'code'),'--checkpoint',str(ck),'--checkpoint-sha256',hashlib.sha256(ck.read_bytes()).hexdigest(),'--checkpoint-step',str(step),'--out',str(R/'smoke_regularized_cpu'),'--seeds','1','--steps','2','--cal-batches','1','--eval-batches','1','--batch-size','16','--rank','48','--device','cpu'],check=True)

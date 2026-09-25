import argparse,json,subprocess,sys,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];C=R/'code';p=argparse.ArgumentParser();p.add_argument('--shard',type=int,required=True);a=p.parse_args();py=sys.executable
def run(cmd):print(json.dumps({'event':'launch','command':list(map(str,cmd))}),flush=True);subprocess.run(list(map(str,cmd)),cwd=C,check=True)
for i,seed in enumerate(range(100,112)):
 if i%2!=a.shard:continue
 checkpoint=R/'backbones'/f'continuation_L8_seed{seed}'/'best.pt'
 for fit in (1,2):
  out=R/'continuation'/f'backbone{seed}_fit{fit}';out.mkdir(parents=True,exist_ok=True)
  if not (out/'summary.json').exists():
   run([py,'-u','-m','reasoning_loop.paper2027_graph_g3_controller','train','--checkpoint',checkpoint,'--out-dir',out,'--seed',fit,'--validation-seed',10000+fit,'--validation-batch-size',250,'--validation-lock',R/'locks/selection.pt','--train-exclude-locks',*sorted((R/'locks').glob('*.pt')),'--protocol-id','paper2027.n10.continuation.v1'])
  modes=['raw','full']+(['D_only','no_AB','identity_D','AB_only','no_bias','mean_D','shuffle_D','spectrum_matched_random_delta','executor_off','batch_shuffle'] if fit==1 else [])
  for mode in modes:
   dest=out/f'eval_{mode}'
   if not (dest/'summary.json').exists():
    run([py,'-u','-m','reasoning_loop.paper2027_graph_g3_controller','evaluate','--checkpoint',checkpoint,'--controller',out/'best_controller.pt','--locked-test',R/'locks/confirmation.pt','--out-dir',dest,'--mode',mode,'--test-seed',20260923,'--batch-size',250,'--require-final-test-lock-excluded','--protocol-id','paper2027.n10.continuation.eval.v1'])

import sys,os,json,time,subprocess,hashlib,argparse
from pathlib import Path
import torch
P=Path(__file__).resolve().parents[1];B=Path('/data/wujiaju/n10_migration_20260923');C=P/'code'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(cmd):print(json.dumps({'event':'launch','command':list(map(str,cmd))}),flush=True);subprocess.run(list(map(str,cmd)),check=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--models',nargs='+',required=True);a=ap.parse_args();py=sys.executable;torch.set_num_threads(4)
 for name in a.models:
  out=P/'local'/name;out.mkdir(parents=True,exist_ok=True);seed={'A':6,'B':3,'C':4,'D':5,'E':7}[name];cp=B/f'backbones/local_control_L6_seed{seed}/best.pt'
  if (out/'complete.json').exists():continue
  (out/'run.json').write_text(json.dumps({'state':'running','pid':os.getpid(),'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'time':time.time()}))
  maps={};ids=[]
  for hop in [1,2]:
   for fit in [1,2]:
    sub=out/f'hop{hop}_seed{fit}'
    if not (sub/'summary.json').exists():run([py,'-u',C/'train_dense.py','train','--checkpoint',cp,'--out-dir',sub,'--target-hop',hop,'--seed',fit,'--rank',256,'--validation-seed',10000+fit,'--validation-lock',B/'locks/selection.pt','--train-exclude-locks',*sorted((B/'locks').glob('*.pt')),'--validation-batch-size',500])
    ident=json.loads((sub/'training_identity.json').read_text());ref=json.loads((B/f'local/{name}/hop{hop}_seed{fit}/training_identity.json').read_text());assert ident['input_stream_sha256']==ref['input_stream_sha256'],'training stream differs'
    sd=torch.load(sub/'best_controller.pt',map_location='cpu',weights_only=False)['controller_state_dict'];maps[f'seed{fit}_J_{"one" if hop==1 else "two"}_beh_rank256']=sd;ids.append({'hop':hop,'fit':fit,'matched_input_stream':True,'best_sha256':sha(sub/'best_controller.pt')})
  torch.save(maps,out/'controllers.pt');(out/'export.json').write_text(json.dumps({'parameterization':'hW+b','fits':ids,'compatibility_key_rank':256},indent=2))
  run([py,'-u',C/'graph_matrix.py','--models',name])
  run([py,'-u',C/'composition.py','--models',name])
  # Preserve preselected head for matched comparison; also retain all-head sensitivity.
  head=json.loads((B/f'local/{name}/head_selection.json').read_text())['locked_head'];step=torch.load(cp,map_location='cpu',weights_only=False)['step']
  for h in range(4):
   dest=P/'battery'/name/f'head{h}'
   run([py,'-u','/data/wujiaju/n10_fig4_fresh_20260924/code/evaluate_battery.py','--looplus',B/'code','--checkpoint',cp,'--checkpoint-sha256',sha(cp),'--checkpoint-step',step,'--controllers',out/'controllers.pt','--controller-sha256',sha(out/'controllers.pt'),'--rank',256,'--datasets','/data/wujiaju/n10_selected_mechanism_20260924/datasets.json','--family','N10_dense_affine_matched','--backbone-name',name,'--head',h,'--panel','confirmation','--seeds',1,2,'--out',dest])
  (out/'complete.json').write_text(json.dumps({'state':'complete','locked_head':head,'time':time.time(),'pid':os.getpid(),'gpu':os.environ['CUDA_VISIBLE_DEVICES']}))
if __name__=='__main__':main()

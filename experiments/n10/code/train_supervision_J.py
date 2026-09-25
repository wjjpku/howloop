"""Fixed eight executor calls, eight post-unit shared affine insertions, final-only target f^n."""
import argparse,os,time,json,hashlib
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
import torch.nn.functional as F
from supervision_utils import AffineJ,readout,save_json,sha,load_checkpoint,fixed_depth_batch
R=Path(__file__).resolve().parent.parent/"supervision"
TARGETS=[9,10,12,15]
@torch.no_grad()
def prepare_first_loop(m,c,d,device):
 s=torch.tensor(d['graphs'],device=device).repeat_interleave(10,0);start=torch.arange(10,device=device).repeat(len(d['graphs']));order=torch.tensor(d['order'],device=device).repeat_interleave(10,0)
 tokens,path,_,_=fixed_depth_batch(c,len(start),device,path_positions=25,successors=s,start=start)
 tokens[:,1:31]=tokens[:,1:31].reshape(-1,10,3).gather(1,order[:,:,None].expand(-1,-1,3)).reshape(-1,30)
 states=[]
 for lo in range(0,len(tokens),64):
  x=m.token_embed(tokens[lo:lo+64])+m.pos_embed[None];states.append(m.apply_loop(x,loop_index=0))
 return torch.cat(states),torch.cat((start[:,None],path),1),tokens
def run_seven(m,j,x,mode='J'):
 pre=x
 for t in range(1,8):
  pre=x if mode=='noJ' else j(x)
  x=pre if mode=='J_only' else m.apply_loop(pre,loop_index=t)
 return (x if mode=="noJ" else j(x)),x
@torch.no_grad()
def evaluate(m,j,data,n,mode='J'):
 source,path,_=data;correct=0;loss=0.;post=0;total=len(source)
 for lo in range(0,total,64):
  x=source[lo:lo+64].clone();ys=path[lo:lo+64,n]
  if mode=='shuffled_answer':x[:,-1]=x[:,-1].roll(10,0)
  out,pre=run_seven(m,j,x,mode);z=readout(m,out);correct+=int(z.argmax(-1).eq(ys).sum());loss+=float(F.cross_entropy(z,ys,reduction='sum'));post+=int(readout(m,pre).argmax(-1).eq(ys).sum())
 return {'accuracy':correct/total,'loss':loss/total,'correct':correct,'n':total,'pre_final_J_target_accuracy':post/total}
def frozen_digest(m):
 h=hashlib.sha256()
 for k,v in m.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def run(args):
 torch.set_num_threads(4);device=torch.device(args.device)
 if device.type=='cuda':torch.cuda.set_per_process_memory_fraction((8*1024**3)/torch.cuda.get_device_properties(device).total_memory,0)
 torch.manual_seed(916000+args.seed);dest=R/'fixed8_runs'/(args.run_name or f'seed{args.seed}_n{args.target}');dest.mkdir(parents=True,exist_ok=False)
 source=Path(args.checkpoint);m,c,p=load_checkpoint(source,device);assert c.max_loops==8 and c.node_count==10 and p['step'] in [20000,30000];m.eval().requires_grad_(False);initial_digest=frozen_digest(m)
 j=AffineJ(c.d_model).to(device);initial_J_meta=None
 if args.init_j:
  cp=torch.load(args.init_j,map_location=device,weights_only=False);j.load_state_dict(cp['J']);initial_J_meta={'path':args.init_j,'sha256':sha(Path(args.init_j)),'source_step':cp['step'],'optimizer_and_sampling_rng':'reset; source best lacks optimizer and RNG'}
 opt=torch.optim.AdamW(j.parameters(),lr=args.lr,betas=(.9,.95),eps=1e-8,weight_decay=0.)
 ds=json.loads((R/'datasets.json').read_text());data={name:prepare_first_loop(m,c,ds[name],device) for name in ds};meta={'seed':args.seed,'target_n':args.target,'executor_loops':8,'J_insertions':8,'J_position':'answer token only','J_shared_across_boundaries':True,'J_form':'h+deltaW h+b','J_initialization':'identity','J_parameters':sum(x.numel() for x in j.parameters()),'init_J':initial_J_meta,'checkpoint':str(source),'checkpoint_sha256':sha(source),'backbone_step':p['step'],'frozen_backbone':True,'edge_layout':'randomized, dataset order matched across both backbones','optimizer_reset':False,'steps':args.steps,'batch_size':args.batch_size,'lr':args.lr,'optimizer':'AdamW betas .9 .95 eps1e-8 wd0','gradient_clip':1.,'precision':'float32','query_depth_token':8,'supervision':'only loop8 CE against f^n(start)','device':str(device),'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'pid':os.getpid(),'started':time.time(),'status':'running','dataset_sha256':sha(R/'datasets.json'),'stop_acc':args.stop_acc,'stop_count':args.stop_count,'eval_every':args.eval_every};save_json(dest/'manifest.json',meta);print('START',json.dumps(meta),flush=True)
 baseline={k:evaluate(m,j,data[k],args.target,'noJ') for k in ['validation','test','cycle10']};save_json(dest/'baseline.json',baseline)
 gen=torch.Generator(device=device).manual_seed(917000+args.seed);history=[];best=(-1.,float('inf'));train_x,train_path,_=data['train']
 high_count=0;stop_reason='budget_exhausted'
 for step in range(1,args.steps+1):
  ix=torch.randint(len(train_x),(args.batch_size,),device=device,generator=gen);x=train_x[ix];ys=train_path[ix,args.target];opt.zero_grad(set_to_none=True);out,_=run_seven(m,j,x);z=readout(m,out);loss=F.cross_entropy(z,ys)
  if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
  loss.backward();gn=torch.nn.utils.clip_grad_norm_(j.parameters(),1.,error_if_nonfinite=True);opt.step()
  if step==1 or step%100==0 or step==args.steps:
   row={'step':step,'loss':float(loss.detach()),'batch_accuracy':float(z.argmax(-1).eq(ys).float().mean()),'gradient_norm':float(gn),'elapsed_seconds':time.time()-meta['started']}
   if step==1 or step%args.eval_every==0 or step==args.steps:
    val=evaluate(m,j,data['validation'],args.target);row['validation']=val
    high_count=high_count+1 if val['accuracy']>=args.stop_acc else 0
    if val['accuracy']>best[0] or (val['accuracy']==best[0] and val['loss']<best[1]):
     best=(val['accuracy'],val['loss']);tmp=dest/'best.tmp';torch.save({'J':j.state_dict(),'step':step,'validation':val,'meta':meta},tmp);tmp.replace(dest/'best.pt')
   history.append(row);save_json(dest/'history.json',history);print(json.dumps(row),flush=True)
   if high_count>=args.stop_count:
    stop_reason='validation_threshold_reached';break
 final={'J':j.state_dict(),'optimizer':opt.state_dict(),'step':step,'meta':meta,'sampling_rng':gen.get_state(),'torch_rng':torch.get_rng_state()};tmp=dest/'final.tmp';torch.save(final,tmp);tmp.replace(dest/'final.pt')
 results={'baseline':baseline,'final':{},'best':{}}
 for name in ['final','best']:
  cp=torch.load(dest/f'{name}.pt',map_location=device,weights_only=False);j.load_state_dict(cp['J']);res={'step':cp['step']}
  for split in ['train','validation','test','cycle10']:res[split]=evaluate(m,j,data[split],args.target)
  res['executor_off_test']=evaluate(m,j,data['test'],args.target,'J_only');res['shuffled_answer_test']=evaluate(m,j,data['test'],args.target,'shuffled_answer')
  random_j=AffineJ(c.d_model).to(device);rg=torch.Generator(device=device).manual_seed(919000+args.seed)
  with torch.no_grad():
   for a,b in zip(random_j.parameters(),j.parameters()):
    v=torch.randn(a.shape,generator=rg,device=device);a.copy_(v*b.norm()/v.norm().clamp_min(1e-12))
  res['norm_matched_random_J_test']=evaluate(m,random_j,data['test'],args.target);results[name]=res
 assert frozen_digest(m)==initial_digest;assert all(p.grad is None for p in m.parameters());save_json(dest/'results.json',results);meta.update(status='complete',finished=time.time(),backbone_unchanged_verified=True,actual_steps=step,stop_reason=stop_reason);save_json(dest/'manifest.json',meta);print('COMPLETE',json.dumps(results['final']['test']),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--target',type=int,choices=TARGETS,required=True);ap.add_argument('--device',default='cuda');ap.add_argument('--steps',type=int,default=5000);ap.add_argument('--batch-size',type=int,default=128);ap.add_argument('--lr',type=float,default=1e-4);ap.add_argument('--eval-every',type=int,default=500);ap.add_argument('--stop-acc',type=float,default=1.1);ap.add_argument('--stop-count',type=int,default=2);ap.add_argument('--init-j');ap.add_argument('--run-name');run(ap.parse_args())

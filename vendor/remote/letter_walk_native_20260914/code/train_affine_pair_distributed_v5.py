"""Matched L4/L8 shared dense affine fits on a frozen step-200 Ouro backbone."""
import argparse,hashlib,json,math,os,random,shutil,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from transformers import AutoModelForCausalLM,AutoTokenizer
from train_full import MODEL,task,encode_task

BASE=Path('/data/paperexperiment/ouro26_letter_full_20260915')
CK_SHA='ed5b5bff0825de33d5f3a48267bc3b6b721cd6596db03ae22551eb4037aa0f44'

class Affine(nn.Module):
    def __init__(self,d):
        super().__init__();self.delta=nn.Parameter(torch.zeros(d,d));self.bias=nn.Parameter(torch.zeros(d))
    def forward(self,h):return h+torch.nn.functional.linear(h,self.delta,self.bias)

def main():
    p=argparse.ArgumentParser();p.add_argument('--loops',type=int,choices=[4,8],required=True);p.add_argument('--steps',type=int,default=500);p.add_argument('--resume',action='store_true');p.add_argument('--no-general-early-stop',action='store_true');p.add_argument('--verify-distributed-step',action='store_true');a=p.parse_args()
    world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'));local_rank=int(os.environ.get('LOCAL_RANK','0'))
    gpu=os.environ['CUDA_VISIBLE_DEVICES'];assert gpu==('7,6' if world==2 else ('7' if a.loops==8 else '5'))
    assert world in (1,2) and (world==1 or a.loops==8)
    torch.cuda.set_device(local_rank)
    if world>1:dist.init_process_group('nccl',device_id=torch.device('cuda',local_rank))
    root=Path('/data/paperexperiment/ouro26_affine_pair_20260915')/f'L{a.loops}'
    if rank==0:root.mkdir(parents=True,exist_ok=a.resume)
    if world>1:dist.barrier()
    def emit(row):
        if rank!=0:return
        print(json.dumps(row),flush=True)
        with (root/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    torch.set_num_threads(4);torch.manual_seed(20260915)
    ckpath=BASE/'run/checkpoint.pt';digest=hashlib.sha256()
    with ckpath.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):digest.update(b)
    assert digest.hexdigest()==CK_SHA
    ck=torch.load(ckpath,map_location='cpu',weights_only=False,mmap=True);assert ck['step']==200
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(MODEL,trust_remote_code=True,local_files_only=True,torch_dtype=torch.float32,attn_implementation='sdpa')
    model.load_state_dict(ck['model'],strict=True);del ck
    model.requires_grad_(False);model=model.to('cuda')
    assert model.config.attention_dropout==0
    model.config.total_ut_steps=a.loops;model.model.total_ut_steps=a.loops;model.config.use_cache=False
    affine=Affine(model.config.hidden_size).to('cuda')
    opt=torch.optim.AdamW(affine.parameters(),lr=1e-4,weight_decay=0)
    versions={n:p._version for n,p in model.named_parameters()}
    general={s:{src:np.load(BASE/'data'/f'{s}_{src}.npy') for src in ['documents','stories','code']} for s in ['train','validation']}
    enabled=True;trace=[];audit=False;boundary_grads={}
    def hook(module,args,kwargs):
        stage=kwargs.get('current_ut',0)
        if enabled and stage>0:
            trace.append(stage)
            h=affine(args[0])
            if audit and h.requires_grad:
                h.register_hook(lambda grad,s=stage:boundary_grads.__setitem__(s,float(grad.float().norm())))
            return (h,)+args[1:],kwargs
        return args,kwargs
    handle=model.model.layers[0].register_forward_pre_hook(hook,with_kwargs=True)
    def batch(rows):
        n=max(len(r[0]) for r in rows);pad=tok.pad_token_id or tok.eos_token_id
        ids=torch.tensor([r[0]+[pad]*(n-len(r[0])) for r in rows],device='cuda')
        labels=torch.tensor([r[1]+[-100]*(n-len(r[1])) for r in rows],device='cuda')
        mask=torch.tensor([[1]*len(r[0])+[0]*(n-len(r[0])) for r in rows],device='cuda')
        return ids,labels,mask
    def loss(rows):
        ids,labels,mask=batch(rows)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            _,states,_=model.model(input_ids=ids,attention_mask=mask,use_cache=False)
            logits=model.lm_head(states[-1][:,:-1,:])
            return torch.nn.functional.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),labels[:,1:].reshape(-1),ignore_index=-100)
    manifest=dict(pid=os.getpid(),gpu=gpu,world_size=world,parallelism='task microbatches sharded; affine gradients averaged once per update; general replay replicated',loops=a.loops,checkpoint=str(ckpath),checkpoint_step=200,checkpoint_sha256=CK_SHA,
      controller='shared dense affine h + delta_W h + b, initialized exact identity',controller_parameters=sum(p.numel() for p in affine.parameters()),
      boundary='before first decoder layer of each loop except first; after previous loop RMSNorm',all_tokens=True,applications=a.loops-1,
      backbone_frozen=True,gate_frozen=True,gate_used=False,final_only=True,tbptt=False,
      train_k=list(range(1,9)),examples_per_update=16,per_k_per_update=2,templates=5,seed=20260915,
      optimizer='AdamW FP32 controller/state',lr=1e-4,warmup=50,steps=a.steps,general_input_fraction=.2,
      loss='0.8 task mean answer CE + 0.2 general token CE',general_guard='monitor only; user authorized continuation' if a.no_general_early_stop else 'stop if own initial NLL + 0.3 exceeded',
      eval='16 fixed held-out graphs per k; fixed template0; final held-out template5; not strict depth OOD',
      precision='FP32 frozen weights, BF16 autocast',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if rank==0:(root/(f'manifest_resume_{os.getpid()}.json' if a.resume else 'manifest.json')).write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(event='rank_ready',rank=rank,pid=os.getpid(),physical_gpu=gpu.split(',')[local_rank])),flush=True)
    emit(dict(event='loaded',loops=a.loops,controller_params=manifest['controller_parameters']))
    model.eval();ids,labels,mask=batch([encode_task(tok,9100000,4,0)])
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        enabled=False;plain=model(input_ids=ids,attention_mask=mask,exit_at_step=a.loops-1,use_cache=False).logits
        enabled=True;trace.clear();identity=model(input_ids=ids,attention_mask=mask,exit_at_step=a.loops-1,use_cache=False).logits
        diff=float((plain-identity).abs().max());assert diff==0,diff
        assert trace==list(range(1,a.loops)),trace
    del plain,identity
    emit(dict(event='identity_and_boundary_check_passed',max_logit_difference=diff,boundaries=trace.copy()))
    model.train();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    def general_eval(step):
        model.eval();values={}
        with torch.no_grad():
            for src in general['validation']:
                values[src]=sum(float(loss([(arr.tolist(),arr.tolist())])) for arr in general['validation'][src][:4])/4
        model.train();nll=sum(values[src]*w for src,w in zip(['documents','stories','code'],[.7,.2,.1]));emit(dict(event='general_eval',step=step,nll=nll,by_source=values));return nll
    def task_eval(step,templates=(0,)):
        if world>1 and rank!=0:
            dist.barrier();return
        model.eval();model.gradient_checkpointing_disable();rows=[]
        with torch.inference_mode():
            for template in templates:
                for k in range(1,9):
                    for i in range(16):
                        prompt,answer,_=task(9100000+i,k,template)
                        ids=tok.apply_chat_template([dict(role='user',content=prompt)],add_generation_prompt=True,return_tensors='pt').to('cuda')
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            out=model.generate(ids,max_new_tokens=16,do_sample=False,use_cache=True,exit_at_step=a.loops-1,pad_token_id=tok.pad_token_id or tok.eos_token_id)
                        tokens=out[0,ids.shape[1]:].tolist();text=tok.decode(tokens,skip_special_tokens=True).strip()
                        row=dict(step=step,template=template,k=k,graph=i,expected=answer,generated=text,generated_ids=tokens,correct=text.strip('.!\n ').lower()==answer.lower(),length_limit_reached=len(tokens)==16)
                        rows.append(row)
                        with (root/'eval_outputs.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        emit(dict(event='task_eval',step=step,acc={f't{t}_k{k}':sum(r['correct'] for r in rows if r['template']==t and r['k']==k)/16 for t in templates for k in range(1,9)}))
        model.train();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        if world>1:dist.barrier()
    def frozen_check():
        assert all(not p.requires_grad and p.grad is None and p._version==versions[n] for n,p in model.named_parameters())
    def save(step,baseline,counts):
        frozen_check();assert shutil.disk_usage(root).free>2*2**30
        if rank!=0:
            if world>1:dist.barrier()
            return
        tmp=root/'checkpoint.tmp';torch.save(dict(affine=affine.state_dict(),optimizer=opt.state_dict(),step=step,baseline_nll=baseline,counts=counts,backbone_sha256=CK_SHA,loops=a.loops,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),tmp)
        os.replace(tmp,root/'checkpoint.pt');emit(dict(event='checkpoint',step=step,backbone_unchanged=True))
        if world>1:dist.barrier()
    start=0;counts=dict(task_input=0,general_input=0,task_supervised=0,general_supervised=0)
    if a.resume:
        ck=torch.load(root/'checkpoint.pt',map_location='cpu',weights_only=False);assert ck['backbone_sha256']==CK_SHA and ck['loops']==a.loops
        affine.load_state_dict(ck['affine']);opt.load_state_dict(ck['optimizer']);start=ck['step'];baseline=ck['baseline_nll'];counts=ck['counts'];torch.set_rng_state(ck['rng']);torch.cuda.set_rng_state(ck['cuda_rng']);del ck
        emit(dict(event='resumed',step=start,general_early_stop=not a.no_general_early_stop,optimizer_restored=True))
    else:baseline=general_eval(0);task_eval(0)
    held_graphs={task(9100000+i,1,0)[2] for i in range(16)}
    for step in range(start+1,a.steps+1):
        begin=time.time();opt.zero_grad(set_to_none=True);rows=[]
        for j in range(16):
            seed=300000000+step*100+j;row=encode_task(tok,seed,j%8+1,(step*16+j)%5)
            while row[2] in held_graphs:seed+=100000000;row=encode_task(tok,seed,j%8+1,(step*16+j)%5)
            rows.append(row)
        nt=sum(len(r[0]) for r in rows);remaining=round(nt/4);r=random.Random(8200000+step);grows=[]
        while remaining>=2:
            src=r.choices(['documents','stories','code'],[.7,.2,.1])[0];arr=r.choice(general['train'][src]);length=min(remaining,512);tokens=arr[:length].tolist();grows.append((tokens,tokens));remaining-=length
        data_hash=hashlib.sha256(json.dumps([[(x[0],x[1]) for x in rows],grows]).encode()).hexdigest()
        task_ce=0.;general_ce=0.;boundary_grads.clear()
        for j in range(rank*2,16,2*world):
            audit=step==1 and j==rank*2;trace.clear();val=loss(rows[j:j+2]);(val*.8/8).backward();task_ce+=float(val.detach())*world/8
            if audit:
                assert set(boundary_grads)==set(range(1,a.loops)),boundary_grads
                assert all(v>0 and math.isfinite(v) for v in boundary_grads.values()),boundary_grads
                emit(dict(event='boundary_gradient_check_passed',step=step,boundary_grad_norms=boundary_grads.copy()))
            audit=False
        glabels=sum(len(x[0])-1 for x in grows)
        for row in grows:
            val=loss([row]);w=(len(row[0])-1)/glabels;(val*.2*w/world).backward();general_ce+=float(val.detach())*w
        frozen_check();assert all(p.grad is not None for p in affine.parameters())
        if world>1:
            for param in affine.parameters():dist.all_reduce(param.grad)
            metric=torch.tensor([task_ce,general_ce],device='cuda',dtype=torch.float64);dist.all_reduce(metric);metric/=world;task_ce,general_ce=metric.tolist()
            if a.verify_distributed_step and step==start+1:
                if rank==0:
                    distributed_grads=[param.grad.clone() for param in affine.parameters()];opt.zero_grad(set_to_none=True)
                    for j in range(0,16,2):(loss(rows[j:j+2])*.8/8).backward()
                    for row in grows:
                        w=(len(row[0])-1)/glabels
                        (loss([row])*.2*w).backward()
                    serial_grads=[param.grad.clone() for param in affine.parameters()]
                    errors=[float((s-d).norm()/d.norm().clamp_min(1e-12)) for s,d in zip(serial_grads,distributed_grads)]
                    opt.zero_grad(set_to_none=True)
                    for j in range(0,16,2):(loss(rows[j:j+2])*.8/8).backward()
                    for row in grows:
                        w=(len(row[0])-1)/glabels
                        (loss([row])*.2*w).backward()
                    repeats=[float((param.grad-s).norm()/s.norm().clamp_min(1e-12)) for param,s in zip(affine.parameters(),serial_grads)]
                    emit(dict(event='gradient_parity_diagnostic',step=step,distributed_serial_l2=errors,serial_repeat_l2=repeats))
                    for param,reference,error,repeat in zip(affine.parameters(),distributed_grads,errors,repeats):
                        assert error<max(1e-4,3*repeat) and error<.003,(error,repeat)
                        param.grad.copy_(reference)
                    emit(dict(event='distributed_serial_gradient_parity_passed',step=step,relative_l2_errors=errors,serial_repeat_l2=repeats))
                dist.barrier()
        norm=float(torch.nn.utils.clip_grad_norm_(affine.parameters(),1.,error_if_nonfinite=True))
        lr=1e-4*min(step/50,1.)*(1 if step<=50 else .1+.9*.5*(1+math.cos(math.pi*(step-50)/max(1,a.steps-50))))
        for group in opt.param_groups:group['lr']=lr
        opt.step();counts['task_input']+=nt;counts['general_input']+=sum(len(x[0]) for x in grows);counts['task_supervised']+=sum(sum(y!=-100 for y in x[1][1:]) for x in rows);counts['general_supervised']+=glabels
        emit(dict(event='train',step=step,task_loss=task_ce,general_loss=general_ce,lr=lr,grad_norm=norm,delta_fro=float(affine.delta.norm()),bias_norm=float(affine.bias.norm()),data_sha256=data_hash,counts=counts,seconds=time.time()-begin,peak_gib=torch.cuda.max_memory_allocated()/2**30))
        if torch.cuda.mem_get_info()[0]<16*2**30:raise RuntimeError('GPU reserve breached')
        if step==2 or (world>1 and step==start+1):save(step,baseline,counts)
        if step%100==0 or step==a.steps:
            nll=general_eval(step);save(step,baseline,counts);task_eval(step)
            if nll>baseline+.3 and not a.no_general_early_stop:emit(dict(event='stopped_general_regression',step=step,baseline=baseline,nll=nll));return
    task_eval(a.steps,templates=(5,));frozen_check();emit(dict(event='complete',backbone_unchanged=True))

if __name__=='__main__':main()

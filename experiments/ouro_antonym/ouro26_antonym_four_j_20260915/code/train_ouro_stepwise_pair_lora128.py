"""Shared affine J on frozen stepwise-supervised Ouro, requested k1..8, pair-only targets."""
import argparse,hashlib,json,math,os,random,shutil,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from transformers import AutoModelForCausalLM,AutoTokenizer
from train_ouro_full import MODEL
from ouro_stepwise_pair_task import example,parse as parse_fields
from datetime import timedelta

def task(seed,k,template):
    prompt,answer,sig,_=example(seed,k,k,template)
    return prompt,answer,sig

def encode_task(tok,seed,k,template):
    prompt,answer,sig=task(seed,k,template)
    prefix=tok.apply_chat_template([dict(role='user',content=prompt)],tokenize=True,add_generation_prompt=True)
    full=tok.apply_chat_template([dict(role='user',content=prompt),dict(role='assistant',content=answer)],tokenize=True)
    assert full[:len(prefix)]==prefix
    return full,[-100]*len(prefix)+full[len(prefix):],sig
from ouro_eval_panel import EVAL_SEEDS
import re

BASE=Path('/data/wujiaju/ouro26_stepwise_pair_control_20260915')
CK_SHA='807db90d207ea86f428b7dfb9602badc4615181c1c0f39fbd555ac2b1917fd72'

class Affine(nn.Module):
    def __init__(self,d):
        super().__init__();self.A=nn.Parameter(torch.empty(128,d));self.B=nn.Parameter(torch.zeros(d,128));self.bias=nn.Parameter(torch.zeros(d))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20260915);nn.init.kaiming_uniform_(self.A,a=math.sqrt(5))
    @property
    def delta(self):return self.B@self.A
    def forward(self,h):return h+torch.nn.functional.linear(torch.nn.functional.linear(h,self.A),self.B,self.bias)

def main():
    p=argparse.ArgumentParser();p.add_argument('--loops',type=int,choices=[4],default=4);p.add_argument('--steps',type=int,default=500);p.add_argument('--resume',action='store_true');p.add_argument('--no-general-early-stop',action='store_true');p.add_argument('--verify-distributed-step',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--min-k',type=int,choices=[1],default=1);a=p.parse_args()
    world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'));local_rank=int(os.environ.get('LOCAL_RANK','0'))
    gpu=os.environ['CUDA_VISIBLE_DEVICES'];assert len(gpu.split(','))==world
    assert world in (1,2,3)
    torch.cuda.set_device(local_rank)
    if world>1:dist.init_process_group('nccl',timeout=timedelta(minutes=90),device_id=torch.device('cuda',local_rank))
    assert json.loads((BASE/'backbone_evaluation.json').read_text())['gate_passed']
    root=BASE/'shared_j_lora128_k18'
    if rank==0:root.mkdir(parents=True,exist_ok=a.resume)
    if world>1:dist.barrier()
    def emit(row):
        if rank!=0:return
        print(json.dumps(row),flush=True)
        with (root/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    torch.set_num_threads(4);torch.manual_seed(20260915)
    ckpath=BASE/'checkpoint.pt';digest=hashlib.sha256()
    with ckpath.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):digest.update(b)
    assert digest.hexdigest()==CK_SHA
    ck=torch.load(ckpath,map_location='cpu',weights_only=False,mmap=True);assert ck['step']==500
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(MODEL,trust_remote_code=True,local_files_only=True,torch_dtype=torch.float32,attn_implementation='sdpa')
    model.load_state_dict(ck['model'],strict=True);del ck
    model.requires_grad_(False);model=model.to('cuda')
    assert model.config.attention_dropout==0
    model.config.total_ut_steps=a.loops;model.model.total_ut_steps=a.loops;model.config.use_cache=False
    affine=Affine(model.config.hidden_size).to('cuda')
    opt=torch.optim.AdamW(affine.parameters(),lr=1e-4,weight_decay=0)
    versions={n:p._version for n,p in model.named_parameters()}
    general={s:{src:np.load(Path('/data/wujiaju/ouro26_letter_full_20260915/data')/f'{s}_{src}.npy') for src in ['documents','stories','code']} for s in ['train','validation']}
    enabled=True;trace=[];audit=False;boundary_grads={}
    stage_counter=0
    def reset_counter(module,args,kwargs):
        nonlocal stage_counter
        stage_counter=0
    def norm_hook(module,args,output):
        nonlocal stage_counter
        stage=stage_counter;stage_counter+=1
        assert stage<4
        if enabled:
            trace.append(stage);h=affine(output)
            if audit and h.requires_grad:
                h.register_hook(lambda grad,s=stage:boundary_grads.__setitem__(s,float(grad.float().norm())))
            return h
        return output
    handle0=model.model.register_forward_pre_hook(reset_counter,with_kwargs=True)
    handle=model.model.norm.register_forward_hook(norm_hook)
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
    manifest=dict(pid=os.getpid(),gpu=gpu,world_size=world,parallelism='task and replay microbatches sharded; affine gradients summed once per update at unchanged global loss weights',loops=a.loops,checkpoint=str(ckpath),checkpoint_step=500,checkpoint_sha256=CK_SHA,
      controller='one shared rank128 affine h + B A h + b, applied four times; A Kaiming uniform, B/b zero; alpha=rank=128; scale1; dropout0',controller_parameters=sum(p.numel() for p in affine.parameters()),
      boundary='after each loop RMSNorm, including loop4 before frozen LM head',all_tokens=True,applications=4,
      output_contract='deleted pair only',backbone_training='per-loop t=1..4 supervision, fixed requested4 prompt',backbone_frozen=True,gate_frozen=True,gate_used=False,final_only=True,tbptt=False,
      train_k=list(range(a.min_k,9)),examples_per_update=16,per_k_per_update=16//(9-a.min_k),templates=5,seed=20260915,
      optimizer='AdamW FP32 controller/state',lr=1e-4,warmup=50,steps=a.steps,general_input_fraction=.2,
      loss='0.8 task mean answer CE + 0.2 general token CE',general_guard='monitor only; user authorized continuation' if a.no_general_early_stop else 'stop if own initial NLL + 0.3 exceeded',
      eval='64 fresh topology-heldout sequences per k; template0 primary paired comparison; template5 additional; k5-8 supervised for J, not J depth-OOD',
      precision='FP32 frozen weights, BF16 autocast',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if rank==0:(root/(f'manifest_resume_{os.getpid()}.json' if a.resume else 'manifest.json')).write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(event='rank_ready',rank=rank,pid=os.getpid(),physical_gpu=gpu.split(',')[local_rank])),flush=True)
    emit(dict(event='loaded',loops=a.loops,controller_params=manifest['controller_parameters']))
    model.eval();ids,labels,mask=batch([encode_task(tok,9100000,4,0)])
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        enabled=False;plain=model(input_ids=ids,attention_mask=mask,exit_at_step=a.loops-1,use_cache=False).logits
        enabled=True;trace.clear();identity=model(input_ids=ids,attention_mask=mask,exit_at_step=a.loops-1,use_cache=False).logits
        diff=float((plain-identity).abs().max());assert diff==0,diff
        assert trace==list(range(4)),trace
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
        model.eval();model.gradient_checkpointing_disable();records=[]
        old_side=tok.padding_side;tok.padding_side='left'
        with torch.inference_mode():
            for template in templates:
                for k in range(1,9):
                    indices=list(range(rank,64,world))
                    for off in range(0,len(indices),4):
                        subset=indices[off:off+4]
                        examples=[task(EVAL_SEEDS[i],k,template) for i in subset]
                        prompts=[tok.apply_chat_template([dict(role='user',content=e[0])],add_generation_prompt=True,tokenize=True) for e in examples]
                        b=tok.pad(dict(input_ids=prompts),padding=True,return_tensors='pt').to('cuda')
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            out=model.generate(**b,max_new_tokens=160,do_sample=False,use_cache=True,exit_at_step=3,pad_token_id=tok.pad_token_id)
                        for i,e,tokens in zip(subset,examples,out[:,b.input_ids.shape[1]:].tolist()):
                            text=tok.decode(tokens,skip_special_tokens=True).strip()
                            parsed=parse_fields(text);expected=parse_fields(e[1])
                            records.append(dict(step=step,template=template,k=k,sequence=i,expected=expected,generated=text,parsed=parsed,generated_ids=tokens,correct=parsed==expected,content_correct=parsed is not None and parsed[0]==expected[0],fields=[parsed is not None and parsed[j]==expected[j] for j in range(1)]))
        if world>1:
            gathered=[None]*world;dist.all_gather_object(gathered,records);records=[r for rs in gathered for r in rs]
        if rank==0:
            with (root/'eval_outputs.jsonl').open('a') as f:
                for row in records:f.write(json.dumps(row)+'\n')
        emit(dict(event='task_eval',step=step,acc={f't{t}_k{k}':sum(r['correct'] for r in records if r['template']==t and r['k']==k)/64 for t in templates for k in range(1,9)},content_acc={f't{t}_k{k}':sum(r['content_correct'] for r in records if r['template']==t and r['k']==k)/64 for t in templates for k in range(1,9)}))
        emit(dict(event='field_accuracy',step=step,n_per_cell=64,fields={f't{t}_k{k}':[sum(r['fields'][j] for r in records if r['template']==t and r['k']==k)/64 for j in range(1)] for t in templates for k in range(1,9)}))
        tok.padding_side=old_side;model.train();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
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
    else:
        baseline=general_eval(0)
        task_eval(0)
    held_graphs={task(9100000+i,1,0)[2] for i in range(16)}
    for step in range(start+1,(2 if a.smoke else a.steps)+1):
        begin=time.time();opt.zero_grad(set_to_none=True);rows=[]
        for j in range(16):
            k=j%(9-a.min_k)+a.min_k
            seed=300000000+step*100+j;row=encode_task(tok,seed,k,(step*16+j)%5)
            while row[2] in held_graphs:seed+=100000000;row=encode_task(tok,seed,k,(step*16+j)%5)
            rows.append(row)
        nt=sum(len(r[0]) for r in rows);remaining=round(nt/4);r=random.Random(8200000+step);grows=[]
        while remaining>=2:
            src=r.choices(['documents','stories','code'],[.7,.2,.1])[0];arr=r.choice(general['train'][src]);length=min(remaining,512);tokens=arr[:length].tolist();grows.append((tokens,tokens));remaining-=length
        data_hash=hashlib.sha256(json.dumps([[(x[0],x[1]) for x in rows],grows]).encode()).hexdigest()
        task_ce=0.;general_ce=0.;boundary_grads.clear()
        for j in range(rank*2,16,2*world):
            audit=step==1 and j==rank*2;trace.clear();val=loss(rows[j:j+2]);(val*.8/8).backward();task_ce+=float(val.detach())*world/8
            if audit:
                assert set(boundary_grads)==set(range(4)),boundary_grads
                assert all(v>0 and math.isfinite(v) for v in boundary_grads.values()),boundary_grads
                emit(dict(event='boundary_gradient_check_passed',step=step,boundary_grad_norms=boundary_grads.copy()))
            audit=False
        glabels=sum(len(x[0])-1 for x in grows)
        for micro,row in enumerate(grows):
            if micro%world!=rank:continue
            val=loss([row]);w=(len(row[0])-1)/glabels;(val*.2*w).backward();general_ce+=float(val.detach())*w*world
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
    if not a.smoke:task_eval(a.steps,templates=(5,))
    frozen_check();emit(dict(event='complete',smoke=a.smoke,backbone_unchanged=True))
    if world>1:dist.destroy_process_group()

if __name__=='__main__':main()



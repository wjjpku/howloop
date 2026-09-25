"""Four-loop deep supervision from raw pretrained Ouro; no J in this stage."""
import argparse,json,math,os,random,shutil,time,hashlib
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers.optimization import Adafactor
from train_ouro_full import MODEL
from ouro_stepwise_task import encode,example,parse,selftest
from ouro_eval_panel import EVAL_SEEDS

ROOT=Path('/data/wujiaju/ouro26_stepwise_control_20260915')

def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--steps',type=int,default=500);a=p.parse_args()
    selftest();assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))==1
    ROOT.mkdir(parents=True,exist_ok=a.resume)
    if shutil.disk_usage(ROOT).free<24_000_000_000:raise RuntimeError('Need atomic checkpoint reserve')
    torch.set_num_threads(4);torch.manual_seed(20260915)
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(MODEL,trust_remote_code=True,local_files_only=True,torch_dtype=torch.float32,attn_implementation='sdpa').to('cuda')
    model.requires_grad_(True);model.model.early_exit_gate.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.config.use_cache=False
    opt=Adafactor([x for x in model.parameters() if x.requires_grad],lr=1e-5,scale_parameter=False,relative_step=False,warmup_init=False,beta1=None,weight_decay=0)
    general={split:{src:np.load(Path('/data/wujiaju/ouro26_letter_full_20260915/data')/f'{split}_{src}.npy') for src in ['documents','stories','code']} for split in ['train','validation']}
    counts=dict(task_input=0,general_input=0,task_labels=0,general_labels=0);start=0
    if a.resume:
        ck=torch.load(ROOT/'checkpoint.pt',map_location='cpu',weights_only=False,mmap=True)
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);start=ck['step'];counts=ck['counts'];torch.set_rng_state(ck['rng']);torch.cuda.set_rng_state(ck['cuda_rng']);del ck
    manifest=dict(pid=os.getpid(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],initial_model=MODEL,resume=a.resume,steps=a.steps,
        supervision='at each loop t=1..4, [removed pair | sequence before step t | sequence after step t]',
        requested_k_during_backbone=4,prompt_stage_invariant=True,no_oracle_previous_stage_input=True,
        teacher_forcing='only own output prefix per branch; no earlier-loop target concatenation',
        backward='full BPTT through t loops for each supervised branch; no detach/TBPTT',
        branch_implementation='separate forwards needed for different autoregressive targets; prefixes with t loops equal prefixes of four-loop unroll',
        loop_loss_weights=[.25]*4,task_sequences_per_update=8,templates=5,gate_frozen=True,J=False,
        lr=1e-5,warmup=50,optimizer='Adafactor FP32 factored no momentum',general_fraction=.2,general_mix=[.7,.2,.1],
        loss='.8 mean of 4 stage losses + .2 general token CE; replay fraction counts actual branch input tokens',
        checkpoint_retention=1,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/f'manifest_{"resume" if a.resume else "smoke"}.json').write_text(json.dumps(manifest,indent=2))
    def emit(r):
        print(json.dumps(r),flush=True)
        with (ROOT/'metrics.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
    emit(dict(event='loaded',**manifest))
    def loss(row,depth,audit=False):
        model.model.total_ut_steps=depth
        ids=torch.tensor([row[0]],device='cuda');labels=torch.tensor([row[1]],device='cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            _,states,_=model.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)
            if audit:
                for h in states:h.retain_grad()
            # Only answer positions reach the vocabulary head, avoiding huge prompt logits.
            keep=labels[0,1:]!=-100
            logits=model.lm_head(states[-1][0,:-1][keep]).float()
            val=torch.nn.functional.cross_entropy(logits,labels[0,1:][keep])
        return val,states if audit else None
    def eval_general(step):
        model.eval();vals={}
        with torch.no_grad():
            for src in general['validation']:
                vals[src]=float(np.mean([float(loss((r.tolist(),r.tolist()),4)[0]) for r in general['validation'][src][:2]]))
        emit(dict(event='general_validation',step=step,by_source=vals,nll=sum(vals[s]*w for s,w in zip(vals,[.7,.2,.1]))));model.train()
    def evaluate(step,final=False):
        model.eval();model.gradient_checkpointing_disable();tok.padding_side='left'
        result=[];seeds=EVAL_SEEDS if final else EVAL_SEEDS[:16]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            for depth in range(1,5):
                model.model.total_ut_steps=depth
                for off in range(0,len(seeds),4):
                    rr=[example(seed,4,depth,0) for seed in seeds[off:off+4]]
                    ids=[tok.apply_chat_template([dict(role='user',content=r[0])],tokenize=True,add_generation_prompt=True) for r in rr]
                    b=tok.pad(dict(input_ids=ids),padding=True,return_tensors='pt').to('cuda')
                    out=model.generate(**b,exit_at_step=depth-1,max_new_tokens=160,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id)
                    for seed,r,ids in zip(seeds[off:off+4],rr,out[:,b.input_ids.shape[1]:]):
                        text=tok.decode(ids,skip_special_tokens=True).strip();parsed=parse(text)
                        record=dict(step=step,seed=seed,loop=depth,expected=r[3],generated=text,parsed=parsed,
                            fields=[parsed is not None and parsed[i]==r[3][i] for i in range(3)],exact=parsed==r[3])
                        result.append(record)
                        with (ROOT/'eval_outputs.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        acc={str(t):dict(exact=float(np.mean([r['exact'] for r in result if r['loop']==t])),
            fields=[float(np.mean([r['fields'][i] for r in result if r['loop']==t])) for i in range(3)]) for t in range(1,5)}
        emit(dict(event='task_eval',step=step,n_per_loop=len(seeds),acc=acc))
        if final:
            (ROOT/'backbone_evaluation.json').write_text(json.dumps(dict(step=step,acc=acc,
                gate_passed=all(v['exact']>=.95 for v in acc.values()),
                gate='all four loops >=95% three-field exact match on 64 topology-heldout sequences; necessary, not sufficient for mechanism conclusion'),indent=2))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.train()
    def save(step):
        if shutil.disk_usage(ROOT).free<12_000_000_000:raise RuntimeError('Insufficient atomic save reserve')
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step,counts=counts,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),protocol=manifest),ROOT/'checkpoint.tmp')
        os.replace(ROOT/'checkpoint.tmp',ROOT/'checkpoint.pt');emit(dict(event='checkpoint',step=step))
    if not a.resume:eval_general(0)
    for step in range(start+1,(2 if a.smoke else a.steps)+1):
        began=time.time();model.train();opt.zero_grad(set_to_none=True);stage_losses=[];nt=0;nl=0;audits=[]
        for depth in range(1,5):
            stage_loss=0
            for i in range(8):
                row=encode(tok,100000000+step*100+i,depth,(step*8+i)%5)
                audit=step==1 and i==0
                val,states=loss(row,depth,audit)
                (val*.8/32).backward();stage_loss+=float(val.detach())/8
                if audit:
                    norms=[float(h.grad.float().norm()) for h in states];assert all(x>0 and math.isfinite(x) for x in norms);audits.append(norms)
                nt+=len(row[0]);nl+=sum(x!=-100 for x in row[1][1:])
            stage_losses.append(stage_loss)
        rng=random.Random(8000000+step);remaining=round(nt/4);replay=[]
        while remaining>=2:
            src=rng.choices(['documents','stories','code'],[.7,.2,.1])[0];n=min(512,remaining)
            tokens=rng.choice(general['train'][src])[:n].tolist();replay.append(tokens);remaining-=n
        ng=sum(len(r) for r in replay);labels=sum(len(r)-1 for r in replay);gl=0
        for tokens in replay:
            val,_=loss((tokens,tokens),4);w=(len(tokens)-1)/labels;(val*.2*w).backward();gl+=float(val.detach())*w
        missing=[n for n,p in model.named_parameters() if p.requires_grad and p.grad is None];assert not missing,missing
        norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True))
        lr=1e-5*min(step/50,1)* (1 if step<=50 else .1+.9*.5*(1+math.cos(math.pi*(step-50)/max(1,a.steps-50))))
        for group in opt.param_groups:group['lr']=lr
        opt.step();counts['task_input']+=nt;counts['general_input']+=ng;counts['task_labels']+=nl;counts['general_labels']+=labels
        emit(dict(event='train',step=step,stage_losses=stage_losses,general_loss=gl,lr=lr,grad_norm=norm,
            bptt_audit=audits or None,counts=counts,seconds=time.time()-began,peak_gib=torch.cuda.max_memory_allocated()/2**30))
        if torch.cuda.mem_get_info()[0]<16*2**30:raise RuntimeError('GPU reserve breached')
        if not a.smoke and (step%100==0 or step==a.steps):save(step);eval_general(step);evaluate(step,final=step==a.steps)
    if a.smoke:
        save(2)
        ck=torch.load(ROOT/'checkpoint.pt',map_location='cpu',weights_only=False,mmap=True);assert ck['step']==2 and ck['optimizer']['state'];del ck
        emit(dict(event='smoke_passed',checkpoint_readback=True))
    else:emit(dict(event='budget_complete',stage='backbone only; J training not started',gate_file=str(ROOT/'backbone_evaluation.json')))

if __name__=='__main__':main()

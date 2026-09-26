"""Frozen numerical Qwen: common-H4 suffix necessity audit, no optimization."""
import argparse
from collections import defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(16 << 20), b''): digest.update(part)
    return digest.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def panel(rows):
    # Original paired_bank is task-major, base-case-major, then k=1..8.
    assert len(rows) == 1280
    out = []
    for ti in range(5):
        for bi in [0,16]:
            chunk = rows[ti*256+bi*8:ti*256+(bi+1)*8]
            assert [r['steps'] for r in chunk] == list(range(1,9))
            assert len({r['task'] for r in chunk}) == 1
            for ki,r in enumerate(chunk):
                out.append(dict(panel_id=len(out), original_index=ti*256+bi*8+ki,
                    **{k:r[k] for k in ['task','start','steps','prompt','answer']},
                    archived_generated=r['generated']))
    return out


def audit_class(parent, torch):
    class Audit(parent):
        arm = 'normal'

        def forward(self, input_ids, attention_mask=None, *, loops=None,
                    labels=None, return_boundaries=False, last_only=False):
            assert labels is None, 'inference only'
            loops = self.loops if loops is None else int(loops)
            hidden = self.base.model.embed_tokens(input_ids)
            if hidden.device.type == 'cuda' and torch.is_autocast_enabled('cuda'):
                hidden = hidden.to(torch.get_autocast_dtype('cuda'))
            if attention_mask is None: attention_mask = torch.ones_like(input_ids)
            pos = (attention_mask.long().cumsum(-1)-1).clamp_min(0)
            length = input_ids.shape[1]
            cache_position = torch.arange(length, device=input_ids.device)
            causal = torch.ones(length,length,dtype=torch.bool,device=hidden.device).tril()
            causal = causal[None,None] & attention_mask[:,None,None,:].bool()
            pe = self.base.model.rotary_emb(hidden,pos)
            kwargs = dict(attention_mask=causal,position_ids=pos,past_key_value=None,
                use_cache=False,cache_position=cache_position,position_embeddings=pe)
            boundaries = []
            for loop in range(1,loops+1):
                use_j = self.use_j and loop > 1 and not (self.arm=='no_j_after4' and loop>4)
                if use_j: hidden = self.controller(hidden)
                skip = (self.arm=='no_f_after4' and loop>4) or self.arm==f'skip_f{loop}'
                if not skip:
                    before = hidden
                    for layer in self.base.model.layers: hidden = layer(hidden,**kwargs)
                    if self.arm=='half_f_after4' and loop>4:
                        hidden = before + .5*(hidden-before)
                if return_boundaries: boundaries.append(hidden)
            hidden = self.base.model.norm(hidden)
            if last_only: hidden = hidden[:,-1:,:]
            logits = self.base.lm_head(hidden)
            return SimpleNamespace(logits=logits,loss=None,boundaries=boundaries)
    return Audit


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--root',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--selftest',action='store_true')
    p.add_argument('--full-panel',action='store_true')
    p.add_argument('--seconds',type=int,default=7200)
    a = p.parse_args()
    sys.path.insert(0,str(a.source))
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,Qwen3Config,Qwen3ForCausalLM
    from model import LoopedQwen3
    from data import prompt_ids,batch
    from tasks import parse_first_answer
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    Audit = audit_class(LoopedQwen3,torch)
    arms = ['normal','no_j_after4','no_f_after4','skip_f5','skip_f6','skip_f7','skip_f8','half_f_after4']
    if a.full_panel:
        arms = ['normal','no_j_after4','no_f_after4','prefix4']
    evaluated_arms = arms if a.full_panel else arms+['raw4','raw8']
    if a.selftest:
        torch.manual_seed(20260912)
        config = Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,
            num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=16)
        config._attn_implementation='sdpa'
        model = Audit(Qwen3ForCausalLM(config),loops=8,train_j=True,activation_checkpointing=False).eval()
        with torch.no_grad():
            model.controller.weight.normal_(0,.01)
            ids=torch.tensor([[0,0,1,2,3],[1,2,3,4,5]])
            mask=ids.ne(0).long()
            baseline=LoopedQwen3.forward(model,ids,mask,return_boundaries=True,last_only=True)
            for arm in arms:
                model.arm=arm
                result=model(ids,mask,return_boundaries=True,last_only=True)
                assert torch.equal(result.boundaries[3],baseline.boundaries[3]), arm
                if arm=='normal': assert torch.equal(result.logits,baseline.logits)
            # Identity replacement of F5 has exactly seven stack executions.
            count=[0]
            handle=model.base.model.layers[0].register_forward_hook(lambda *args: count.__setitem__(0,count[0]+1))
            model.arm='skip_f5'; model(ids,mask,last_only=True)
            handle.remove(); assert count[0]==7
        print(json.dumps(dict(selftest='passed',normal_exact=True,common_H4_all_arms=True,skip_count=7)))
        return
    assert os.environ.get('CUDA_VISIBLE_DEVICES') is not None
    assert torch.cuda.device_count()==1
    a.output.mkdir(parents=True,exist_ok=False)
    begin=time.monotonic()
    status=dict(status='preparing',pid=os.getpid(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],training=False,torch=torch.__version__)
    write(a.output/'status.json',status)
    def event(name,**kw):
        value=dict(event=name,seconds=time.monotonic()-begin,**kw)
        with (a.output/'events.jsonl').open('a') as f: f.write(json.dumps(value)+'\n')
        print(json.dumps(value),flush=True)
    def guard():
        if time.monotonic()-begin>a.seconds: raise TimeoutError('bounded Qwen deadline')
        free,total=torch.cuda.mem_get_info()
        if free < max(16*2**30,.2*total): raise RuntimeError('GPU reserve breached')
    try:
        original_rows=json.loads((a.root/'formal/affine/summary.json').read_text())['test']['rows']
        examples=panel(original_rows)
        if a.full_panel:
            examples=[dict(panel_id=i,original_index=i,
                **{k:r[k] for k in ['task','start','steps','prompt','answer']},
                archived_generated=r['generated']) for i,r in enumerate(original_rows)]
        write(a.output/'panel.json',examples)
        hashes={}
        for name,directory,expected in [
            ('backbone','baseline','2d07a4256fdc327f5ba796f35cce4614697c2f74b4d32072089cf35a8c9fabbd'),
            ('controller','affine','e652e59973ec1fe92ec00c73c7fe87a6390c52343387e5fe493b5727d2bada09')]:
            hashes[name]=sha(a.root/f'formal/{directory}/latest.pt')
            assert hashes[name]==expected,(name,hashes[name])
            event('hash_verified',component=name,sha256=hashes[name])
        status.update(status='loading',hashes=hashes,panel_sha256=sha(a.output/'panel.json'),
            arms=evaluated_arms,parameter_dtype='float32',compute_dtype='bfloat16',
            source_hashes={n:sha(a.source/n) for n in ['model.py','data.py','semantic_data.py','tasks.py']},
            audit_source_sha256=sha(Path(__file__)))
        write(a.output/'status.json',status)
        tok=AutoTokenizer.from_pretrained(a.root/'original',local_files_only=True)
        base=AutoModelForCausalLM.from_pretrained(a.root/'original',local_files_only=True,
            torch_dtype=torch.float32,attn_implementation='sdpa',low_cpu_mem_usage=True)
        state=torch.load(a.root/'formal/baseline/latest.pt',map_location='cpu',weights_only=False,mmap=True)
        base.load_state_dict(state['model'],strict=True)
        assert state['update']==9500
        del state; gc.collect()
        model=Audit(base,loops=8,train_j=True,activation_checkpointing=False)
        state=torch.load(a.root/'formal/affine/latest.pt',map_location='cpu',weights_only=False)
        assert state['update']==12000
        model.controller.load_state_dict(state['model'],strict=True)
        del state
        model.eval().requires_grad_(False).distribute([torch.device('cuda:0')])
        assert all(p.dtype==torch.float32 for p in model.parameters())
        params=list(model.parameters()); versions=[p._version for p in params]
        def tokenize(rs):
            items=[prompt_ids(tok,r['prompt']) for r in rs]
            return batch([(x,[-100]*len(x)) for x in items],tok.pad_token_id,'cuda:0',left=True)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            b=tokenize(examples[:2])
            old=LoopedQwen3.forward(model,b['input_ids'],b['attention_mask'],return_boundaries=True,last_only=True)
            for arm in arms:
                model.arm=arm
                new=model(b['input_ids'],b['attention_mask'],return_boundaries=True,last_only=True)
                assert torch.equal(new.boundaries[3],old.boundaries[3]),arm
                if arm=='normal': assert torch.equal(new.logits,old.logits),'normal differs from source'
                guard()
            del old,new
            event('smoke_passed',normal_exact=True,common_H4_all_arms=True,
                peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20)
            status.update(status='evaluating',smoke_passed=True)
            write(a.output/'status.json',status)
            allrows=[]
            for arm in evaluated_arms:
                model.arm=arm if not arm.startswith('raw') else 'normal'
                model.use_j=not arm.startswith('raw')
                loops=4 if arm in ['raw4','prefix4'] else 8
                for first in range(0,len(examples),5):
                    guard()
                    rs=examples[first:first+5]; b=tokenize(rs)
                    output=model.generate(b['input_ids'],b['attention_mask'],max_new_tokens=12,
                        eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id,loops=loops)
                    for r,ids in zip(rs,output[:,b['input_ids'].shape[1]:].tolist()):
                        if tok.eos_token_id in ids: ids=ids[:ids.index(tok.eos_token_id)]
                        generated=tok.decode(ids,skip_special_tokens=True)
                        parsed=parse_first_answer(generated,r['task'])
                        expected=r['answer'].lower() if r['task']=='weekday' else int(r['answer'])
                        row=dict(**r,arm=arm,generated=generated,parsed=parsed,
                            parsed_correct=parsed==expected,
                            whole_correct=generated.strip().casefold()==str(r['answer']).strip().casefold(),
                            archive_same=generated==r['archived_generated'])
                        allrows.append(row)
                        with (a.output/'predictions.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
                    event('batch',arm=arm,complete=first+len(rs),peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20)
                def stats(rr):
                    return dict(n=len(rr),whole=sum(r['whole_correct'] for r in rr),parsed=sum(r['parsed_correct'] for r in rr))
                sums={}
                for armname in {r['arm'] for r in allrows}:
                    rr=[r for r in allrows if r['arm']==armname]
                    sums[armname]=dict(all=stats(rr),short=stats([r for r in rr if r['steps']<=4]),
                        long=stats([r for r in rr if r['steps']>4]),
                        per_task_long={t:stats([r for r in rr if r['task']==t and r['steps']>4]) for t in {r['task'] for r in rr}})
                write(a.output/'summary.json',sums)
                event('arm_complete',arm=arm,summary=sums[arm])
        assert all(p._version==v for p,v in zip(params,versions))
        status.update(status='complete',frozen_verified=True,elapsed_seconds=time.monotonic()-begin,
            predictions_sha256=sha(a.output/'predictions.jsonl'),summary_sha256=sha(a.output/'summary.json'))
        write(a.output/'status.json',status)
    except BaseException as exc:
        status.update(status='failed',error=repr(exc),elapsed_seconds=time.monotonic()-begin)
        write(a.output/'status.json',status)
        raise


if __name__=='__main__': main()

"""Paired head-output census and held-out causal group ablation. No training."""
import argparse, hashlib, json, os, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from train_ouro_full import MODEL, task
from train_ouro_four_j import Affine, CK_SHA
from ouro_eval_panel import EVAL_SEEDS
from score_ouro_content import parse

ROOT = Path('/data/paperexperiment/ouro26_antonym_heads_20260915')
BASE = Path('/data/paperexperiment/ouro26_antonym_full_20260915/run/checkpoint.pt')
JPATH = Path('/data/paperexperiment/ouro26_antonym_four_j_20260915/fit/checkpoint.pt')
JSHA = '203712ced24e6dd3be01bc275a2fc5983a3372db9bde380d08b7db4069972e0f'
ARMS = {'j4': (True, 4), 'plain4': (False, 4), 'j8': (True, 8)}

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''): h.update(b)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--mode', choices=['collect', 'ablate', 'boundary'], default='collect')
    p.add_argument('--limit', type=int, default=64)
    a = p.parse_args()
    assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) == 1
    root = ROOT / a.arm / a.mode
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4); torch.manual_seed(20260915)
    assert digest(BASE) == CK_SHA and digest(JPATH) == JSHA
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True,
        local_files_only=True, torch_dtype=torch.float32, attn_implementation='sdpa')
    ck = torch.load(BASE, map_location='cpu', weights_only=False, mmap=True)
    assert ck['step'] == 500
    model.load_state_dict(ck['model'], strict=True); del ck
    model = model.to('cuda').eval().requires_grad_(False)
    j = Affine(2048)
    ck = torch.load(JPATH, map_location='cpu', weights_only=False)
    assert ck['step'] == 500 and ck['backbone_sha256'] == CK_SHA
    j.load_state_dict(ck['affine']); del ck
    j = j.to('cuda').eval().requires_grad_(False)
    use_j, k = ARMS[a.arm]
    assert model.model.total_ut_steps == 4
    versions = [x._version for x in model.parameters()]
    j_sites = set(range(4)); j_stage = [0]
    def reset_j(m, args, kwargs): j_stage[0] = 0
    def apply_j(m, args, out):
        site = j_stage[0]; j_stage[0] += 1
        assert site < 4
        return j(out) if site in j_sites else out
    model.model.register_forward_pre_hook(reset_j, with_kwargs=True)
    if use_j: model.model.norm.register_forward_hook(apply_j)
    current = [0]; capture = [False]; selected = set(); vectors = {}; rms = {}
    def stage_hook(m, args, kwargs): current[0] = int(kwargs['current_ut'])
    def head_hook(layer):
        def hook(m, args):
            h = args[0]
            v = h.reshape(h.shape[0], h.shape[1], 16, 128)
            if capture[0]:
                vectors[current[0], layer] = v[0, -1].float().cpu().numpy()
                rms[current[0], layer] = v.float().square().mean((0, 1, 3)).sqrt().cpu().numpy()
            indices = [head for loop, lay, head in selected if loop == current[0] and lay == layer]
            if indices:
                v = v.clone(); v[:, :, indices, :] = 0
                return (v.reshape_as(h),)
        return hook
    for layer, block in enumerate(model.model.layers):
        block.self_attn.register_forward_pre_hook(stage_hook, with_kwargs=True)
        block.self_attn.o_proj.register_forward_pre_hook(head_hook(layer))
    manifest = dict(arm=a.arm, mode=a.mode, k=k, use_j=use_j, physical_loops=4,
        backbone_sha256=CK_SHA, j_sha256=JSHA, gpu=os.environ['CUDA_VISIBLE_DEVICES'],
        seeds=EVAL_SEEDS, discovery_indices=list(range(16)), confirmation_indices=list(range(16,64)),
        metric='pre-o_proj head vectors at final prompt position; RMS over all prompt positions',
        ablation='zero selected loop/layer/head outputs before o_proj at all positions and generated tokens',
        caveat='post-attention RMSNorm remains active; zero ablation includes its rescaling effect',
        script_sha256=digest(Path(__file__)), training=False)
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2))
    def emit(row):
        print(json.dumps(row), flush=True)
        with (root/'results.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    def prompt_ids(seed):
        prompt, answer, _ = task(seed, k, 0)
        ids = tok.apply_chat_template([dict(role='user',content=prompt)], tokenize=True, add_generation_prompt=True)
        return torch.tensor([ids], device='cuda'), answer
    def generate(ids, answer):
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), exit_at_step=3,
            max_new_tokens=64, do_sample=False, use_cache=True, pad_token_id=tok.pad_token_id)
        text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        parsed = parse(dict(kind='cancellation', generated=text))
        return dict(generated=text, parsed=parsed, correct=parsed==answer)
    started = time.time()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        if a.mode == 'collect':
            vv, rr = [], []
            for i, seed in enumerate(EVAL_SEEDS[:a.limit]):
                ids, answer = prompt_ids(seed)
                capture[0] = True; vectors.clear(); rms.clear()
                _, states, _ = model.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                logp = model.lm_head(states[-1][:,-1]).float().log_softmax(-1)
                assert len(vectors) == 192
                vv.append(np.stack([vectors[l,d] for l in range(4) for d in range(48)]).reshape(4,48,16,128))
                rr.append(np.stack([rms[l,d] for l in range(4) for d in range(48)]).reshape(4,48,16))
                capture[0] = False
                target = tok.encode(answer, add_special_tokens=False)[0]
                row = dict(i=i, seed=seed, answer=answer, prompt_tokens=ids.shape[1],
                    first_target_logp=logp[0,target].item(), **generate(ids,answer))
                emit(row); del states, logp
            np.savez_compressed(root/'activations.npz', vectors=np.stack(vv), rms=np.stack(rr))
        elif a.mode == 'ablate':
            groups = json.loads((ROOT/'groups.json').read_text())
            for name, sites in groups.items():
                selected.clear(); selected.update(tuple(x) for x in sites)
                for i in range(16, min(64, a.limit)):
                    ids, answer = prompt_ids(EVAL_SEEDS[i])
                    emit(dict(group=name, i=i, answer=answer, **generate(ids,answer)))
        else:
            assert use_j
            for name, sites in [('last_j_only',[3]),('internal_j_only',[0,1,2])]:
                j_sites.clear(); j_sites.update(sites)
                for i in range(16, min(64,a.limit)):
                    ids, answer = prompt_ids(EVAL_SEEDS[i])
                    emit(dict(group=name, j_sites=sites, i=i, answer=answer, **generate(ids,answer)))
    assert versions == [x._version for x in model.parameters()]
    (root/'complete.json').write_text(json.dumps(dict(seconds=time.time()-started,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30, backbone_unchanged=True)))
    print('COMPLETE', flush=True)

if __name__ == '__main__': main()

"""Frozen L8.H5 address counterfactual and downstream-J mediation; no training."""
import importlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from diagnose_ouro_heads import BASE, JPATH, JSHA, digest
from train_ouro_four_j import Affine, CK_SHA
from train_ouro_full import MODEL, task
from task import OPPOSITE
from localize_ouro_l8h5_combinations import EVAL_SEEDS as PREVIOUS_SEEDS
from ouro_eval_panel import EVAL_SEEDS as ORIGINAL_SEEDS
from score_ouro_content import parse

ROOT = Path('/data/wujiaju/ouro_native_causal_20260916')


def main():
    ROOT.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(20260916)
    assert digest(BASE) == CK_SHA and digest(JPATH) == JSHA
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    def example(seed, k):
        prompt, answer, sig = task(seed, k, 0)
        text = tok.apply_chat_template([dict(role='user', content=prompt)], tokenize=False, add_generation_prompt=True)
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = tok.apply_chat_template([dict(role='user', content=prompt)], tokenize=True, add_generation_prompt=True)
        assert ids == enc.input_ids
        start = text.index('Word sequence: ') + len('Word sequence: ')
        end = text.index('.', start)
        positions = [i for i, (a, b) in enumerate(enc.offset_mapping) if a < end and b > start]
        words = text[start:end].split()
        assert len(positions) == len(words) == 24
        live = list(range(24))
        trajectory = []
        while live:
            index = next(i for i in range(len(live)-1) if OPPOSITE[words[live[i]]] == words[live[i+1]])
            trajectory.append(live[index:index+2])
            del live[index:index+2]
        return dict(seed=seed, k=k, ids=ids, positions=positions, words=words,
                    trajectory=trajectory, answer=answer, signature=sig)

    # Labels and pairing are selected using the symbolic oracle only, never model outputs.
    seen = {task(s, 4, 0)[2] for s in set(PREVIOUS_SEEDS + ORIGINAL_SEEDS)}
    pairs = []
    candidates = iter(range(9120000, 9199000))
    def fresh():
        for seed in candidates:
            ex = example(seed, 4)
            if ex['signature'] not in seen:
                seen.add(ex['signature'])
                return ex
        raise RuntimeError('exhausted held-out seeds')
    for index in range(64):
        recipient = fresh()
        while True:
            donor = fresh()
            if donor['positions'] != recipient['positions'] or len(donor['ids']) != len(recipient['ids']):
                continue
            labels = {}
            for k in (4, 8):
                r = example(recipient['seed'], k)
                cf = ' '.join(r['words'][p] for p in donor['trajectory'][k-1])
                labels[k] = cf
                if len({r['answer'], donor['answer'], cf}) != 3:
                    break
            else:
                break
        pairs.append(dict(recipient=recipient['seed'], donor=donor['seed'],
                          counterfactual=labels, split='discovery' if index < 16 else 'confirmation'))

    manifest = dict(pid=os.getpid(), gpu=os.environ['CUDA_VISIBLE_DEVICES'], backbone_sha256=CK_SHA,
                    J_sha256=JSHA, script_sha256=digest(Path(__file__)), pairs=pairs,
                    loops=4, training=False, head='L8.H5 one-based', position='24 input word tokens',
                    phase='prefill only; includes first generated-token decision',
                    fixed_protocol=True, selection='No outcome-based selection; report all predeclared conditions on both splits',
                    address_test='Donor attention probabilities times RECIPIENT values; recipient query retained',
                    address_limitation='Transfers candidate pairing topology; counterfactual may not be a lexical antonym pair. Not an established state variable.',
                    rescue='Bypass J2 or J3, restore only L8.H5 word-position activity in immediately following physical loop',
                    controls=['self attention patch', 'normal-J same-input path', 'native no-J same-input path',
                              'wrong-semantic native donor path', 'native attention-only path'],
                    claim_gate='Native reuse requires semantic confirmation AND specific native restoration; normal-J restoration alone is insufficient')
    (ROOT/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(dict(event='manifest', **manifest)), flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=True,
                    torch_dtype=torch.float32, attn_implementation='sdpa')
    ck = torch.load(BASE, map_location='cpu', weights_only=False, mmap=True)
    model.load_state_dict(ck['model']); del ck
    model.to('cuda').eval().requires_grad_(False)
    model.model.total_ut_steps = model.config.total_ut_steps = 4
    controller = Affine(2048)
    ck = torch.load(JPATH, map_location='cpu', weights_only=False)
    controller.load_state_dict(ck['affine']); del ck
    controller.to('cuda').eval().requires_grad_(False)
    versions = [p._version for p in list(model.parameters()) + list(controller.parameters())]
    state = {}
    def reset(module, args, kwargs):
        ids = kwargs.get('input_ids', args[0] if args else None)
        state.update(loop=0, prefill=ids.shape[1] > 1)
    def norm(module, args, hidden):
        state['loop'] += 1
        return controller(hidden) if state['use_j'] and state['loop'] != state['skip'] else hidden
    model.model.register_forward_pre_hook(reset, with_kwargs=True)
    model.model.norm.register_forward_hook(norm)
    mod = importlib.import_module(type(model.model.layers[7].self_attn).__module__)
    original = mod.ALL_ATTENTION_FUNCTIONS['sdpa']
    def attention(module, q, key, value, mask, **kwargs):
        result = original(module, q, key, value, mask, **kwargs)
        if module.layer_idx != 7 or not state['prefill']:
            return result
        loop = state['loop'] + 1
        positions = state['positions']
        kh = key.repeat_interleave(q.shape[1] // key.shape[1], dim=1)
        vh = value.repeat_interleave(q.shape[1] // value.shape[1], dim=1)
        logits = (q[:, 4].float() @ kh[:, 4].float().transpose(-1, -2)) * kwargs['scaling']
        if mask is None:
            logits.masked_fill_(torch.ones_like(logits, dtype=torch.bool).triu(1), -float('inf'))
        elif mask.dtype == torch.bool:
            logits.masked_fill_(~mask[:, 0, :, :key.shape[2]], -float('inf'))
        else:
            logits += mask[:, 0, :, :key.shape[2]].float()
        prob = logits.softmax(-1)[:, positions]
        output = result[0]
        state['capture'][loop] = dict(prob=prob.detach().clone(), output=output[:, positions, 4].detach().clone())
        patch = state['patch']
        if patch is None or loop not in patch['loops']:
            return result
        source = patch['source'][loop]
        output = output.clone()
        if patch['kind'] == 'output':
            output[:, positions, 4] = source['output']
        else:
            assert source['prob'].shape == prob.shape
            delta = (source['prob'] - prob) @ vh[:, 4].float()
            output[:, positions, 4] += delta.to(output.dtype)
        state['hits'] += 1
        return (output, result[1])
    mod.ALL_ATTENTION_FUNCTIONS.register('ouro_native_causal', attention)
    model.config._attn_implementation = 'ouro_native_causal'

    def run(ex, use_j, skip=None, patch=None):
        if torch.cuda.mem_get_info()[0] < 16 * 2**30:
            raise RuntimeError('GPU reserve breached')
        state.update(use_j=use_j, skip=skip, patch=patch, positions=ex['positions'], capture={}, hits=0)
        ids = torch.tensor([ex['ids']], device='cuda')
        generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), exit_at_step=3,
                        max_new_tokens=64, do_sample=False, use_cache=True, pad_token_id=tok.pad_token_id)
        text = tok.decode(generated[0, ids.shape[1]:], skip_special_tokens=True).strip()
        parsed = parse(dict(kind='cancellation', generated=text))
        if patch is not None:
            assert state['hits'] == len(patch['loops'])
        return dict(generated=text, parsed=parsed, correct=parsed == ex['answer'], expected=ex['answer'],
                    hits=state['hits'], tokens=generated.shape[1]-ids.shape[1]), state['capture']

    totals = defaultdict(lambda: dict(n=0, correct=0, cf=0, donor_copy=0))
    start = time.time()
    def save(pair, arm, condition, row, **extra):
        rec = dict(**pair, arm=arm, condition=condition, **row, **extra)
        with (ROOT/'records.jsonl').open('a') as f:
            f.write(json.dumps(rec)+'\n')
        key = '|'.join([pair['split'], arm, condition])
        t = totals[key]; t['n'] += 1; t['correct'] += row['correct']
        t['cf'] += row['parsed'] == extra.get('cf_label', '<none>')
        t['donor_copy'] += row['parsed'] == extra.get('donor_label', '<none>')

    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for index, pair in enumerate(pairs):
            native_ex = example(pair['recipient'], 4)
            donor_ex = example(pair['donor'], 4)
            native, native_activity = run(native_ex, False)
            donor, donor_activity = run(donor_ex, False)
            for arm, use_j, k in [('plain4', False, 4), ('j4', True, 4), ('j8', True, 8)]:
                ex = example(pair['recipient'], k)
                assert ex['positions'] == donor_ex['positions'] and len(ex['ids']) == len(donor_ex['ids'])
                normal, activity = (native, native_activity) if arm == 'plain4' else run(ex, use_j)
                common = dict(native_correct=native['correct'], donor_correct=donor['correct'],
                              normal_correct=normal['correct'], cf_label=pair['counterfactual'][k], donor_label=donor_ex['answer'])
                save(pair, arm, 'normal', normal, **common)
                identity, _ = run(ex, use_j, patch=dict(kind='prob', loops=[1,2], source=activity))
                assert identity['generated'] == normal['generated'], 'self patch changed generation'
                save(pair, arm, 'self_probability_L12', identity, **common)
                for loops in ([1], [2], [1,2]):
                    row, _ = run(ex, use_j, patch=dict(kind='prob', loops=loops, source=donor_activity))
                    save(pair, arm, 'donor_address_L'+''.join(map(str, loops)), row, **common)
                if arm != 'j8':
                    continue
                for skip in (2,3):
                    bypass, _ = run(ex, True, skip=skip)
                    save(pair, arm, f'skipJ{skip}', bypass, **common)
                    for label, source, kind in [('normal_output', activity, 'output'),
                                                ('native_output', native_activity, 'output'),
                                                ('wrong_native_output', donor_activity, 'output'),
                                                ('native_probability', native_activity, 'prob')]:
                        row, _ = run(ex, True, skip=skip, patch=dict(kind=kind, loops=[skip+1], source=source))
                        save(pair, arm, f'skipJ{skip}_restore_{label}', row, bypass_correct=bypass['correct'], **common)
            print(json.dumps(dict(event='pair_complete', done=index+1, total=len(pairs), split=pair['split'],
                                  elapsed_s=time.time()-start, peak_GiB=torch.cuda.max_memory_allocated()/2**30)), flush=True)
            (ROOT/'progress.json').write_text(json.dumps(dict(done=index+1,total=64,totals=totals), indent=2))
    assert versions == [p._version for p in list(model.parameters()) + list(controller.parameters())]
    summary = dict(status='complete', parameters_unchanged=True, elapsed_s=time.time()-start, totals=totals,
                   interpretation='Raw fixed-protocol counts. Semantic reuse is not established automatically by completion.')
    (ROOT/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()

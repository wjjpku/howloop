"""CPU-only checks of task oracle and split, without importing model libraries."""
import ast
import hashlib
import random
from pathlib import Path
from task import make_sequence, trace, OPPOSITE, bank

source=Path(__file__).with_name('train_ouro_full.py').read_text()
tree=ast.parse(source)
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('topology','partition','task')]
ns=dict(hashlib=hashlib,random=random,make_sequence=make_sequence,trace=trace,OPPOSITE=OPPOSITE)
exec(compile(ast.Module(body=nodes,type_ignores=[]),'<task-only>','exec'),ns)
ns['HISTORICAL']={ns['topology'](r['words']) for r in bank() if r['kind']=='cancellation'}
train=set();val=set()
for held in (False,True):
    for i in range(256):
        seed=(9100000 if held else 100000000)+i
        signatures=[]
        for k in range(1,5):
            prompt,answer,sig=ns['task'](seed,k,i%5)
            words=prompt.split('Word sequence: ',1)[1].split('.',1)[0].split()
            assert len(words)==24 and len(set(words))==24
            assert answer==' '.join(trace(words)[k-1]['removed'])
            assert sig not in ns['HISTORICAL']
            assert (80<=ns['partition'](words)<90) if held else ns['partition'](words)<80
            signatures.append(sig)
        assert len(set(signatures))==1
        (val if held else train).add(signatures[0])
assert not train.intersection(val)
assert 'logits=model.lm_head(hidden[3]' in source
assert 'early_exit_gate.requires_grad_(False)' in source
assert '.detach()' not in source.split('def loss(',1)[1].split('held_graphs=',1)[0]
print('PASS: 2048 oracle cases; train/validation topology disjoint; same sequence across k; final-loop-only loss selected. GPU/backprop verification still required.')

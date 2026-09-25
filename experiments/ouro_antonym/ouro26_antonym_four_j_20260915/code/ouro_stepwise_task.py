"""Same topology split as earlier Ouro tasks, with three-field step targets."""
from train_ouro_full import task as old_task
from task import trace,OPPOSITE

def example(seed,requested=4,stage=4,template=0):
    old,_,sig=old_task(seed,1,template)
    words=old.split('Word sequence: ',1)[1].split('.',1)[0].split()
    history=trace(words);r=history[stage-1]
    before=words if stage==1 else history[stage-2]['remaining']
    fields=[' '.join(r['removed']),' '.join(before),' '.join(r['remaining'])]
    rule=old.split('. ',1)[1].split(' Which two words',1)[0]
    prompt=('Word sequence: '+' '.join(words)+'. '+rule+
        f' Requested number of deletions: {requested}. Report the current deletion update as '
        '[deleted pair | sequence before this deletion | sequence after this deletion]. '
        'Write only these three fields, preserving word order.')
    return prompt,'['+' | '.join(fields)+']',sig,fields

def encode(tok,seed,stage,template=0):
    prompt,answer,sig,_=example(seed,4,stage,template)
    prefix=tok.apply_chat_template([dict(role='user',content=prompt)],tokenize=True,add_generation_prompt=True)
    full=tok.apply_chat_template([dict(role='user',content=prompt),dict(role='assistant',content=answer)],tokenize=True)
    assert full[:len(prefix)]==prefix
    return full,[-100]*len(prefix)+full[len(prefix):],sig

def parse(text):
    text=text.strip().lower()
    if not (text.startswith('[') and text.endswith(']')):return None
    parts=[' '.join(x.split()) for x in text[1:-1].split('|')]
    if len(parts)!=3 or any(w not in OPPOSITE for p in parts for w in p.split()):return None
    if len(parts[0].split())!=2:return None
    return parts

def selftest():
    for seed in [100001001,100001002,9101000,9101001]:
        previous=None;prompt0=None
        for stage in range(1,9):
            prompt,answer,_,f=example(seed,4,stage)
            assert parse(answer)==f and len(f[1].split())==26-2*stage
            assert len(f[2].split())==24-2*stage
            before=f[1].split();idx=next(i for i in range(len(before)-1) if OPPOSITE[before[i]]==before[i+1])
            assert before[idx:idx+2]==f[0].split()
            assert before[:idx]+before[idx+2:]==f[2].split()
            if previous is not None:assert previous==f[1] and prompt==prompt0
            previous=f[2];prompt0=prompt
    assert parse('[hot cold | hot cold | bogus]') is None
    print('oracle, invariant prompt, adjacency, lengths and parser tests passed')

if __name__=='__main__':selftest()

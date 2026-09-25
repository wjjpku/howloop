"""Matched stepwise task: remove before/after fields, keep only the current pair."""
from ouro_stepwise_task import example as three_field_example
from task import OPPOSITE

def example(seed,requested=4,stage=4,template=0):
    prompt,_,sig,fields=three_field_example(seed,requested,stage,template)
    prompt=prompt.split(' Report the current deletion update as ',1)[0]+(
        ' Report only the two words deleted in the current deletion update. '
        'Write only the pair, preserving word order.')
    return prompt,fields[0],sig,[fields[0]]

def parse(text):
    words=text.strip().lower().split()
    if len(words)!=2 or any(w not in OPPOSITE for w in words):return None
    return [' '.join(words)]

def encode(tok,seed,stage,template=0):
    prompt,answer,sig,_=example(seed,4,stage,template)
    prefix=tok.apply_chat_template([dict(role='user',content=prompt)],tokenize=True,add_generation_prompt=True)
    full=tok.apply_chat_template([dict(role='user',content=prompt),dict(role='assistant',content=answer)],tokenize=True)
    assert full[:len(prefix)]==prefix
    return full,[-100]*len(prefix)+full[len(prefix):],sig

def selftest():
    for seed in [100001001,100001002,9101000,9101001]:
        for template in range(5):
            prompts=[]
            for stage in range(1,9):
                prompt,answer,sig,fields=example(seed,4,stage,template)
                old,_,oldsig,oldfields=three_field_example(seed,4,stage,template)
                assert parse(answer)==fields==[oldfields[0]] and sig==oldsig
                assert prompt.split(' Report ')[0]==old.split(' Report ')[0]
                prompts.append(prompt)
            assert len(set(prompts))==1
    assert parse('hot cold extra') is None
    print('Pair-only oracle, matched topology/rule, stage-invariant prompt and parser passed')

if __name__=='__main__':selftest()

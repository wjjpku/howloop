"""Semantic k and physical loop count are independent; reuse historical start domains."""
from dataclasses import replace
from collections import Counter
import random
from tasks import TASK_NAMES, WEEKDAYS, build_enumerated_bank, make_example


def semantic_example(example, k, template=0):
    if not 1 <= k <= 8 or not 0 <= template < 5:
        raise ValueError((k, template))
    value=example.start
    trace=[]
    for _ in range(k):
        if example.task=='successor': value+=1
        elif example.task=='weekday': value=WEEKDAYS[(WEEKDAYS.index(value)+1)%7]
        elif example.task=='doubling': value*=2
        elif example.task=='fibonacci': value=(value[1],value[0]+value[1])
        elif example.task=='collatz': value=value//2 if value%2==0 else 3*value+1
        else: raise ValueError(example.task)
        trace.append(value[1] if example.task=='fibonacci' else value)
    canonical=example.prompt.replace('exactly 04',f'exactly {k:02d}')
    start=str(example.start)
    if example.task=='fibonacci': start=f'{example.start[0]}, {example.start[1]}'
    action={'successor':'apply the integer successor operation',
            'weekday':'move forward by one calendar day', 'doubling':'double the value',
            'fibonacci':'advance the Fibonacci pair recurrence',
            'collatz':'apply the standard Collatz step'}[example.task]
    target='second value of the final pair' if example.task=='fibonacci' else ('final weekday' if example.task=='weekday' else 'final value')
    bodies=[canonical,
        f'Begin at {start}. Repeat this operation {k:02d} times: {action}. Give the {target}.',
        f'Take {start} as the initial state. After you {action} exactly {k:02d} times, what is the {target}?',
        f'Perform {k:02d} iterations starting from {start}. At each iteration, {action}. Return the {target}.',
        f'Initial state: {start}. Operation: {action}. Number of iterations: {k:02d}. What is the {target}?']
    prompt=bodies[template]
    if template: prompt+=' Answer directly with only the answer. Answer:'
    return replace(example,steps=k,prompt=prompt,answer=str(trace[-1]),trace=tuple(trace))


def paired_bank(max_k, count_per_task=32, seed=2026090802, enumerated=False):
    rng=random.Random(seed)
    base=build_enumerated_bank() if enumerated else [make_example(t,rng) for t in TASK_NAMES for _ in range(count_per_task)]
    # Each starting state has all k prompts, in the same template.
    return [semantic_example(e,k,i%5) for i,e in enumerate(base) for k in range(1,max_k+1)]


class BalancedSampler:
    """One example/task/batch; each K task batches cover every (task,k) equally."""
    def __init__(self,max_k,seed):
        self.max_k=max_k
        self.rng=random.Random(seed)
        self.decks={t:[] for t in TASK_NAMES}
        self.counts=Counter()

    def batch(self):
        batch=[]
        for task in TASK_NAMES:
            if not self.decks[task]:
                self.decks[task]=list(range(1,self.max_k+1));self.rng.shuffle(self.decks[task])
            k=self.decks[task].pop()
            template=self.rng.randrange(5)
            batch.append(semantic_example(make_example(task,self.rng),k,template))
            self.counts[f'{task}/k{k}']+=1
        self.rng.shuffle(batch)
        return batch

    def state_dict(self):
        return dict(rng=self.rng.getstate(),decks={k:list(v) for k,v in self.decks.items()},counts=dict(self.counts))

    def load_state_dict(self,state):
        self.rng.setstate(state['rng']);self.decks={k:list(v) for k,v in state['decks'].items()};self.counts=Counter(state['counts'])


def summarize_rows(rows):
    def stats(rs):
        n=len(rs);c=sum(r['exact'] for r in rs)
        strict=sum(r['generated'].strip().casefold()==str(r['answer']).casefold() for r in rs)
        return dict(correct=c,total=n,accuracy=c/n,answer_only_accuracy=strict/n)
    ks=sorted({r['steps'] for r in rows})
    return dict(overall=stats(rows),
        per_k={str(k):stats([r for r in rows if r['steps']==k]) for k in ks},
        per_task_k={f'{t}/k{k}':stats([r for r in rows if r['task']==t and r['steps']==k]) for t in TASK_NAMES for k in ks})


def gate(metrics):
    return all(v['accuracy']>=.99 and v['answer_only_accuracy']>=.99 for v in metrics['per_task_k'].values())

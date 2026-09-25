"""Deterministic lexical cancellation; no model-dependent answer generation."""
import random
import re

PAIRS = [('hot', 'cold'), ('near', 'far'), ('open', 'closed'),
         ('heavy', 'light'), ('young', 'old'), ('rich', 'poor'),
         ('happy', 'sad'), ('clean', 'dirty'), ('full', 'empty'),
         ('wet', 'dry'), ('fast', 'slow'), ('alive', 'dead')]
OPPOSITE = {a: b for pair in PAIRS for a, b in (pair, pair[::-1])}


def trace(words):
    live = list(enumerate(words))
    result = []
    for k in range(1, len(words) // 2 + 1):
        index = next((i for i in range(len(live) - 1)
                      if OPPOSITE.get(live[i][1]) == live[i + 1][1]), None)
        if index is None:
            break
        left, right = live[index:index + 2]
        enclosed = [r['dependency_depth'] for r in result
                    if left[0] < r['original_indices'][0] < right[0]]
        del live[index:index + 2]
        result.append(dict(k=k, removed=[left[1], right[1]],
                           original_indices=[left[0], right[0]],
                           dependency_depth=1 + max(enclosed, default=0),
                           remaining=[w for _, w in live]))
    return result


def make_sequence(rng):
    # Inserting matched adjacent pairs generates nested AND concatenated
    # structures. Repeated labels are excluded to keep lexical truth unambiguous.
    labels = rng.sample(PAIRS, len(PAIRS))
    words = []
    for pair in labels:
        pos = rng.randrange(len(words) + 1)
        words[pos:pos] = pair if rng.randrange(2) else pair[::-1]
    return words


def bank(seed=20260909, count=16):
    rng = random.Random(seed)
    rows = []
    for i, (a, b) in enumerate(PAIRS):
        for orientation, (u, v) in enumerate(((a, b), (b, a))):
            for truth in (True, False):
                other = v if truth else PAIRS[(i + 1) % len(PAIRS)][orientation]
                rows.append(dict(id=f'lexical-{i}-{orientation}-{truth}', kind='lexical',
                    prompt=f'Are "{u}" and "{other}" antonyms in their usual senses? '
                           'Answer only yes or no.', answer='yes' if truth else 'no'))
    seen = set()
    for sequence_id in range(count):
        while True:
            words = make_sequence(rng)
            signature = tuple(words)
            steps = trace(words)
            if signature not in seen and len(steps) == 12 and max(
                    r['dependency_depth'] for r in steps[:8]) >= 3:
                seen.add(signature)
                break
        for r in steps[:8]:
            k = r['k']
            rows.append(dict(id=f'cancel-{sequence_id}-{k}', kind='cancellation',
                sequence_id=sequence_id, words=words, k=k,
                dependency_depth=r['dependency_depth'], trace=steps,
                prompt='Word sequence: ' + ' '.join(words) + '. '
                    'Repeatedly delete the leftmost adjacent pair of antonyms, '
                    'then join the remaining words without changing their order. '
                    f'Which two words are deleted on deletion number {k}? '
                    'Answer with only those two words in their original left-to-right order.',
                answer=' '.join(r['removed'])))
    return rows


def score(row, text):
    # Permit case/punctuation differences but NOT answer hidden in a long CoT.
    normalized = ' '.join(re.findall(r'[a-z]+', text.lower()))
    return dict(exact=normalized == row['answer'],
                strict=text.strip().lower() == row['answer'], normalized=normalized)


def summarize(rows):
    def metrics(rs):
        return dict(correct=sum(r['exact'] for r in rs), total=len(rs),
                    accuracy=sum(r['exact'] for r in rs) / len(rs) if rs else None,
                    strict_accuracy=sum(r['strict'] for r in rs) / len(rs) if rs else None)
    return dict(lexical=metrics([r for r in rows if r['kind'] == 'lexical']),
        cancellation=metrics([r for r in rows if r['kind'] == 'cancellation']),
        per_k={str(k): metrics([r for r in rows if r.get('k') == k]) for k in range(1, 9)},
        dependency_ge_3=metrics([r for r in rows if r.get('dependency_depth', 0) >= 3]))

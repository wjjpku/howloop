import hashlib
import json
from pathlib import Path
import random
import sys
import numpy as np
import torch

# Existing mathematics/templates are reused, not modified by this experiment.
root = Path(__file__).resolve().parent.parent
sys.path.extend([str(root / "nope_semantic_k_4to8_20260908"),
                 str(root / "nope_1p1b_4loop_baseline_20260905")])
from semantic_data import BalancedSampler, paired_bank, summarize_rows, gate
from tasks import parse_first_answer, TASK_NAMES


def prompt_ids(tokenizer, prompt):
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False)


def supervised_item(tokenizer, prompt, answer):
    prefix = prompt_ids(tokenizer, prompt)
    target = tokenizer.encode(str(answer), add_special_tokens=False) + [tokenizer.eos_token_id]
    return prefix + target, [-100] * len(prefix) + target


def batch(items, pad, device, left=False):
    width = max(len(x[0]) for x in items)
    ids, labels, masks = [], [], []
    for tokens, targets in items:
        n = width - len(tokens)
        ids.append(([pad]*n+tokens) if left else (tokens+[pad]*n))
        labels.append(([-100]*n+targets) if left else (targets+[-100]*n))
        masks.append(([0]*n+[1]*len(tokens)) if left else ([1]*len(tokens)+[0]*n))
    return {"input_ids": torch.tensor(ids, device=device),
            "attention_mask": torch.tensor(masks, device=device),
            "labels": torch.tensor(labels, device=device)}


class Replay:
    def __init__(self, path):
        path = Path(path)
        self.meta = json.loads((path / "manifest.json").read_text())
        self.arrays = {f"{s}_{split}": np.memmap(path / f"{s}_{split}.bin", mode="r", dtype=np.uint32)
                       for s in ("natural", "code") for split in ("train", "val")}
        self.chats = {split: [json.loads(x) for x in (path / f"chat_{split}.jsonl").read_text().splitlines()]
                      for split in ("train", "val")}

    def sample(self, source, rng, split="train", length=256):
        if source == "chat":
            row = self.chats[split][rng.randrange(len(self.chats[split]))]
            return row["ids"], row["labels"]
        arr = self.arrays[f"{source}_{split}"]
        start = rng.randrange(len(arr) - length + 1)
        ids = arr[start:start+length].astype(np.int64).tolist()
        return ids, list(ids)


@torch.no_grad()
def evaluate(model, tokenizer, examples, loops, device, batch_size=5):
    was_training = model.training
    model.eval()
    rows = []
    for start in range(0, len(examples), batch_size):
        es = examples[start:start+batch_size]
        items = [(prompt_ids(tokenizer, e.prompt), []) for e in es]
        items = [(x, [-100]*len(x)) for x, _ in items]
        b = batch(items, tokenizer.pad_token_id, device, left=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model.generate(b["input_ids"], b["attention_mask"],
                    max_new_tokens=12, eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id, loops=loops)
        for e, ids in zip(es, output[:, b["input_ids"].shape[1]:].tolist()):
            if tokenizer.eos_token_id in ids:
                ids = ids[:ids.index(tokenizer.eos_token_id)]
            generated = tokenizer.decode(ids, skip_special_tokens=True)
            parsed = parse_first_answer(generated, e.task)
            expected = e.answer.lower() if e.task == "weekday" else int(e.answer)
            rows.append(dict(task=e.task, start=e.start, steps=e.steps, prompt=e.prompt,
                             answer=e.answer, generated=generated, parsed=parsed,
                             exact=parsed == expected))
    model.train(was_training)
    return dict(rows=rows, **summarize_rows(rows))

"""Full-weight four-loop adaptation followed by frozen-backbone shared dense J.

One process, layer model-parallel GPUs. No DTensors, parameter replication,
LoRA, weight quantization, truncated BPTT, or policy-gradient objective.
"""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
from datetime import datetime, timezone

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
from model import LoopedQwen3
from data import (BalancedSampler, Replay, batch, evaluate, gate, paired_bank,
                  prompt_ids, supervised_item)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(8*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)
        file.flush()
        os.fsync(file.fileno())
    tmp.replace(path)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(v) for v in value)
    return 0


def save_state(path, state, reserve_gib=15):
    # Keep the old fully recoverable checkpoint until the new one is fsynced.
    estimated = int(tensor_bytes(state) * 1.03) + 8 * 2**20
    if shutil.disk_usage(path.parent).free < estimated + reserve_gib * 2**30:
        raise RuntimeError("Cannot atomically save checkpoint with disk reserve; old checkpoint retained")
    tmp = path.with_suffix(".pt.tmp")
    with tmp.open("wb") as file:
        torch.save(cpu_tree(state), file)
        file.flush()
        os.fsync(file.fileno())
    tmp.replace(path)


def build_model(args, baseline):
    if args.tiny:
        config = Qwen3Config(vocab_size=1024, hidden_size=192, intermediate_size=384,
            num_hidden_layers=6, num_attention_heads=6, num_key_value_heads=2,
            head_dim=32, max_position_embeddings=512, attention_dropout=0.0)
        config._attn_implementation = "sdpa"
        base = Qwen3ForCausalLM(config)
    else:
        # Loading the locked FP32 checkpoint directly preserves the baseline
        # exactly; no BF16 export/reload rounding is inserted before J.
        base = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
            torch_dtype=torch.float32,
            attn_implementation="sdpa", low_cpu_mem_usage=True)
    if not baseline and not args.smoke:
        parent = args.output / "baseline"
        summary = json.loads((parent / "summary.json").read_text())
        if not summary["gate_passed"]:
            raise RuntimeError("Baseline did not pass its gate; J is not authorized on a failed baseline")
        checkpoint = parent / "latest.pt"
        if sha(checkpoint) != summary["checkpoint_sha256"]:
            raise RuntimeError("Locked baseline checkpoint changed")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        base = base.float()
        base.load_state_dict(state["model"])
        del state
        gc.collect()
    model = LoopedQwen3(base, loops=4 if baseline else 8, train_j=not baseline)
    devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    if len(devices) != 3:
        raise RuntimeError("This measured launch requires exactly 3 explicitly pinned GPUs")
    return model.distribute(devices)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stage", choices=["baseline", "affine"], default="baseline")
    p.add_argument("--updates", type=int, default=12000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=20260908)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--tiny", action="store_true")
    args = p.parse_args()
    if args.tiny and not args.smoke:
        raise ValueError("Tiny models are engineering tests only")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Physical GPUs must be explicitly pinned")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    baseline = args.stage == "baseline"
    out = args.output / args.stage
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        print("STAGE_ALREADY_FINISHED", flush=True)
        return
    def emit(value):
        value = dict(utc=utc(), **value)
        with (out / "events.jsonl").open("a") as file:
            file.write(json.dumps(value, ensure_ascii=False)+"\n")
        print(json.dumps(value, ensure_ascii=False), flush=True)
    model = build_model(args, baseline)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if baseline:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(parameters, lr=1e-5, betas=(.9, .95),
            weight_decay=.01, min_8bit_size=16384)
    else:
        optimizer = torch.optim.AdamW(parameters, lr=1e-6, betas=(.9, .95), weight_decay=0)
        assert {id(p) for p in parameters} == {id(p) for p in model.controller.parameters()}
    device = model.base.model.embed_tokens.weight.device
    tok = None if args.tiny else AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tok is not None and tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    corpus = None if args.tiny else Replay(args.data)
    sampler = BalancedSampler(4 if baseline else 8, args.seed)
    rng = random.Random(args.seed+1)
    start, streak, counts, stream = 0, 0, {}, "0"*64
    identity = dict(stage=args.stage, seed=args.seed, original_revision="b968826d9c46dd6066d109eabc6255188de91218",
        model_hash=None if args.tiny else sha(args.model / "model.safetensors.index.json"),
        data_hash=None if args.tiny else sha(args.data / "manifest.json"), tiny=args.tiny,
        code_hashes={n:sha(Path(__file__).parent / n) for n in ("model.py", "train.py", "data.py")})
    checkpoint = out / "latest.pt"
    target_module = model.base if baseline else model.controller

    def restore(state):
        if state["identity"] != identity:
            raise RuntimeError("Checkpoint protocol/source identity differs")
        target_module.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        sampler.load_state_dict(state["sampler"])
        rng.setstate(state["rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])

    if checkpoint.exists() and not args.smoke:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        restore(state)
        start, streak, counts, stream = state["update"], state["streak"], state["counts"], state["stream"]
        del state
        gc.collect()
    elif (out / "manifest.json").exists():
        raise RuntimeError("Partial run without checkpoint: inspect before restarting")
    write_json(out / ("resume_manifest.json" if start else "manifest.json"),
        dict(identity=identity, pid=os.getpid(), cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
             parallelism="single-process layer model parallel", loops=model.loops,
             semantic_k=list(range(1,model.loops+1)), supervision="only final loop answer+EOS",
             full_bptt=True, parameter_dtype="FP32",
             compute_dtype="BF16", optimizer="AdamW8bit" if baseline else "AdamW32bit",
             trainable_parameters=sum(p.numel() for p in parameters),
             mix_updates=dict(task=.7,natural=.14,code=.06,chat=.1), microbatch=5,
             warmup=300 if baseline else 100, max_updates=args.updates,
             resume_update=start, utc=utc()))

    def state_dict(update):
        return dict(identity=identity, update=update, streak=streak, counts=counts, stream=stream,
            model=target_module.state_dict(), optimizer=optimizer.state_dict(),
            sampler=sampler.state_dict(), rng=rng.getstate(), torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all())

    def step(b, lr):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = lr
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**b).loss
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite loss")
        loss.backward()
        if not baseline and any(p.grad is not None for p in model.base.parameters()):
            raise RuntimeError("Frozen backbone gradient")
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
        if not torch.isfinite(norm):
            raise FloatingPointError("nonfinite gradient")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return float(loss.detach()), float(norm)

    def language():
        result = {}
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for source in ("natural", "code", "chat"):
                gen = random.Random(args.seed+401)
                losses = []
                for _ in range(4):
                    items = [corpus.sample(source, gen, split="val") for _ in range(5)]
                    b = batch(items, tok.pad_token_id, device)
                    losses.append(float(model(**b).loss))
                result[source] = sum(losses)/len(losses)
        return result

    def compact(metrics):
        return {k:v for k,v in metrics.items() if k != "rows"}

    def dialogues():
        prompts = ["Explain why the sky is blue in one sentence.", "你好，请用两句话介绍一下你自己。",
            "Translate into Chinese: Reading helps us understand the world.",
            "Give two practical ways to organize a busy day."]
        model.eval()
        rows = []
        for prompt in prompts:
            ids = torch.tensor([prompt_ids(tok,prompt)], device=device)
            with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
                generated = model.generate(ids, torch.ones_like(ids), 64, tok.eos_token_id, tok.pad_token_id)
            answer = tok.decode(generated[0,ids.shape[1]:], skip_special_tokens=True)
            rows.append(dict(prompt=prompt, generated=answer))
        return rows

    for i in range(3):
        torch.cuda.reset_peak_memory_stats(i)
    if args.smoke:
        results = []
        for index, source in enumerate(("task", "natural", "chat")):
            if args.tiny:
                ids = torch.randint(3, 1024, (2, 32 if index == 0 else 96), device=device)
                b = dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids.clone())
                if index == 0:
                    b["labels"][:, :-4] = -100
            else:
                if source == "task":
                    items = [supervised_item(tok,e.prompt,e.answer) for e in sampler.batch()]
                elif source == "chat":
                    rows = sorted(corpus.chats["train"], key=lambda x:len(x["ids"]), reverse=True)[:5]
                    items = [(x["ids"],x["labels"]) for x in rows]
                else:
                    items = [corpus.sample(source,rng) for _ in range(5)]
                b = batch(items,tok.pad_token_id,device)
            begin = time.monotonic()
            loss,norm = step(b,1e-5 if baseline else 1e-6)
            results.append(dict(source=source,loss=loss,grad_norm=norm,seconds=time.monotonic()-begin,
                shape=list(b["input_ids"].shape),peak_mib=[torch.cuda.max_memory_allocated(i)/2**20 for i in range(3)]))
            emit(dict(event="smoke_step",**results[-1]))
        # Verify exact full-state continuation, including quantized optimizer state.
        if args.tiny:
            save_state(checkpoint,state_dict(3),reserve_gib=1)
            next_loss,_ = step(b,1e-5 if baseline else 1e-6)
            expected = {n:p.detach().clone() for n,p in target_module.named_parameters()}
            restore(torch.load(checkpoint,map_location="cpu",weights_only=False))
            actual_loss,_ = step(b,1e-5 if baseline else 1e-6)
            assert next_loss == actual_loss, (next_loss,actual_loss)
            assert all(torch.equal(expected[n],p) for n,p in target_module.named_parameters())
        write_json(out/"summary.json",dict(smoke=True, passed=True, results=results,
            exact_resume_verified=args.tiny, utc=utc()))
        return

    val = paired_bank(model.loops, count_per_task=16, seed=args.seed+1000)
    test = paired_bank(8, count_per_task=32, seed=args.seed+2000)
    if not (out/"initial.json").exists():
        if baseline:
            model.loops = 1
            initial = dict(original_tasks=evaluate(model,tok,test,1,device),
                           original_language=language(), original_dialogues=dialogues())
            model.loops = 4
            initial["raw4"] = evaluate(model,tok,val,4,device)
        else:
            model.use_j = False
            raw = evaluate(model,tok,test,8,device)
            probe = paired_bank(8,count_per_task=1,seed=args.seed+900)
            a = evaluate(model,tok,probe,8,device)
            model.use_j = True
            b = evaluate(model,tok,probe,8,device)
            assert [r["generated"] for r in a["rows"]] == [r["generated"] for r in b["rows"]]
            initial = dict(raw8=raw, identity_j_parity=True)
        write_json(out/"initial.json",initial)
        emit(dict(event="initial_complete",stage=args.stage))
    begin = time.monotonic()
    warmup = 300 if baseline else 100
    peak_lr = 1e-5 if baseline else 1e-6
    update = start
    for update in range(start+1,args.updates+1):
        draw = rng.random()
        source = "task" if draw < .7 else "chat" if draw >= .9 else "natural" if draw < .84 else "code"
        if source == "task":
            examples = sampler.batch()
            items = [supervised_item(tok,e.prompt,e.answer) for e in examples]
        else:
            items = [corpus.sample(source,rng) for _ in range(5)]
        b = batch(items,tok.pad_token_id,device)
        counts[source+"_updates"] = counts.get(source+"_updates",0)+1
        counts[source+"_supervised_tokens"] = counts.get(source+"_supervised_tokens",0)+int(b["labels"][:,1:].ne(-100).sum())
        stream = hashlib.sha256((stream+source+json.dumps(items)).encode()).hexdigest()
        lr = peak_lr*min(1.,update/warmup)
        loss,norm = step(b,lr)
        if update == start+1 or update%10 == 0:
            emit(dict(event="train",update=update,source=source,loss=loss,grad_norm=norm,lr=lr,
                seconds=time.monotonic()-begin,counts=counts,task_k_counts=dict(sampler.counts),
                peak_mib=[torch.cuda.max_memory_allocated(i)/2**20 for i in range(3)]))
        if update%args.eval_every == 0 or update == args.updates:
            metrics = evaluate(model,tok,val,model.loops,device)
            streak = streak+1 if gate(metrics) else 0
            write_json(out/"validation_latest.json",metrics)
            general = language()
            write_json(out/"dialogues_latest.json",dialogues())
            emit(dict(event="eval",update=update,metrics=compact(metrics),pass_streak=streak,language=general))
            save_state(checkpoint,state_dict(update))
            emit(dict(event="checkpoint_saved",update=update,bytes=checkpoint.stat().st_size))
            if streak >= 2 and update >= 500:
                break
    test_metrics = evaluate(model,tok,[e for e in test if e.steps <= model.loops],model.loops,device)
    if baseline:
        write_json(out/"no_j8_test.json",evaluate(model,tok,test,8,device))
        write_json(out/"baseline4_all_k_test.json",evaluate(model,tok,test,4,device))
    checkpoint_sha = sha(checkpoint)
    summary = dict(stage=args.stage,updates=update,gate_passed=streak>=2,pass_streak=streak,
        test=test_metrics,counts=counts,task_k_counts=dict(sampler.counts),stream=stream,
        checkpoint_sha256=checkpoint_sha,language=language(),dialogues=dialogues(),utc=utc())
    write_json(out/"summary.json",summary)
    emit(dict(event="finished",updates=update,gate_passed=summary["gate_passed"],test=test_metrics["overall"]))


if __name__ == "__main__":
    main()

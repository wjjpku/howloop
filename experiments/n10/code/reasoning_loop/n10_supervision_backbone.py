"""Train a final-only N8 D8L8 Graph backbone with graph-disjoint locks.

This is intentionally a compact, dedicated trainer instead of a flag layered
onto the exploratory Graph trainer.  Its only admissible data sampler rejects
the pre-created selection and final-test permutation locks on every update.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch
from reasoning_loop.graph_path_loop import (
    GraphPathConfig,
    LoopedGraphPathTransformer,
    build_optimizer_param_groups,
    cosine_lr,
    count_parameters,
    pick_device,
    set_optimizer_base_lr,
    set_seed,
)
from reasoning_loop.paper2027_graph_g4_protocol import (
    load_unique_lock,
    merged_forbidden_codes,
    sample_training_permutations,
    sha256,
)


PROTOCOL_ID = "paper2027.graph.n10.matched_supervision.v1"


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _initial_cfg(args: argparse.Namespace) -> GraphPathConfig:
    cfg = GraphPathConfig(
        node_count=10, max_depth=8, d_model=256, n_heads=4, d_mlp=1024,
        n_layers=2, max_loops=args.loops, dropout=0.0, inner_norm_style="pre_layernorm",
        readout_norm_style="layernorm", block_style="legacy",
        block_schedule="all_blocks",
    )
    if args.node_count != cfg.node_count or args.max_depth != cfg.max_depth:
        raise ValueError("G4 is registered only for N10 D8")
    return cfg


def _locked_endpoint_accuracy(
    *, model: LoopedGraphPathTransformer, cfg: GraphPathConfig,
    lock: dict[str, object], device: torch.device, batch_size: int, amp: bool,
) -> float:
    successors_all = lock["successors"]
    if not isinstance(successors_all, torch.Tensor):
        raise ValueError("lock lacks successors")
    graphs = successors_all.shape[0]
    if batch_size % cfg.node_count:
        raise ValueError("evaluation batch must preserve graph clusters")
    correct = 0
    total = 0
    model.eval()
    autocast_device = "cuda" if device.type == "cuda" else device.type
    graphs_per_batch = batch_size // cfg.node_count
    with torch.inference_mode():
        for first in range(0, graphs, graphs_per_batch):
            last = min(graphs, first + graphs_per_batch)
            successors = successors_all[first:last].repeat_interleave(cfg.node_count, dim=0).to(device)
            start = torch.arange(cfg.node_count, device=device).repeat(last - first)
            tokens, targets, _, _ = fixed_depth_batch(
                cfg, successors.shape[0], device, path_positions=cfg.max_depth,
                successors=successors, start=start,
            )
            order = torch.rand(len(tokens), cfg.node_count, device=device).argsort(-1)
            tokens[:,1:1+3*cfg.node_count] = tokens[:,1:1+3*cfg.node_count].reshape(-1,cfg.node_count,3).gather(1,order[:,:,None].expand(-1,-1,3)).reshape(len(tokens),-1)
            with torch.autocast(
                device_type=autocast_device, dtype=torch.bfloat16,
                enabled=amp and device.type == "cuda",
            ):
                logits = model.forward_all(tokens, max_loops=cfg.max_loops)["logits_by_loop"][:, -1]
            correct += int(logits.argmax(dim=-1).eq(targets[:, -1]).sum())
            total += targets.shape[0]
    return correct / total


def _write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["step", "lr", "train_final_ce", "selection_endpoint_accuracy", "elapsed_sec"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def train(args: argparse.Namespace) -> dict[str, Any]:
    device = pick_device(args.device)
    cfg = _initial_cfg(args)
    if args.steps < 1 or args.eval_every < 1:
        raise ValueError("steps and eval_every must be positive")
    if args.out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_dir}")

    selection_lock = load_unique_lock(args.selection_lock)
    final_lock = load_unique_lock(args.final_test_lock)
    if selection_lock["node_count"] != cfg.node_count or final_lock["node_count"] != cfg.node_count:
        raise ValueError("lock/model node-count mismatch")
    if selection_lock["role"] != "selection" or final_lock["role"] != "final_test":
        raise ValueError("G4 requires correctly role-labelled selection and final locks")
    forbidden_codes = merged_forbidden_codes(
        [args.selection_lock, args.final_test_lock, *args.extra_exclude_locks], node_count=cfg.node_count
    )
    if not args.extra_exclude_locks and forbidden_codes.numel() != int(selection_lock["permutations"]) + int(final_lock["permutations"]):
        raise ValueError("held-out locks must be mutually disjoint")

    args.out_dir.mkdir(parents=True)
    set_seed(args.seed + 1009 * args.loops if args.depth_mode == "uniform" else args.seed)
    model = LoopedGraphPathTransformer(cfg).to(device)
    initial_digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        initial_digest.update(name.encode()); initial_digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    input_digest = hashlib.sha256()
    groups = build_optimizer_param_groups(
        model, base_lr=args.learning_rate,
        component_lr_scales={"embedding": 1.0, "attention": 1.0, "mlp": 1.0, "norm": 1.0, "readout": 1.0},
        block_lr_scales=None,
    )
    optimizer = torch.optim.AdamW(
        groups, lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8,
        weight_decay=args.weight_decay,
    )
    metadata: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "supervision": args.supervision, "edge_layout": "randomized triplets",
        "args": {"trajectory_aux_weight": 1.0 if args.supervision == "per_loop" else 0.0, "trajectory_aux_jump": 1},
        "config": asdict(cfg),
        "training": {"loss": "per-call path CE" if args.supervision == "per_loop" else f"final-only endpoint CE at call {cfg.max_loops}", "depth_mode": args.depth_mode, "steps": args.steps,
                     "batch_size": args.batch_size, "learning_rate": args.learning_rate,
                     "weight_decay": args.weight_decay, "warmup_steps": args.warmup_steps,
                     "trajectory_aux_weight": 0.0, "aux_loss": 0.0},
        "seed": args.seed, "pid": os.getpid(), "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "parameter_count": count_parameters(model),
        "selection_lock": str(args.selection_lock),
        "selection_lock_sha256": sha256(args.selection_lock),
        "final_test_lock": str(args.final_test_lock),
        "final_test_lock_sha256": sha256(args.final_test_lock),
        "extra_excluded_locks": {str(p): sha256(p) for p in args.extra_exclude_locks},
        "withheld_graphs": int(forbidden_codes.numel()),
        "eligible_training_graphs": math.factorial(cfg.node_count) - int(forbidden_codes.numel()),
        "sampler": "uniform rejection from all N! permutation graphs excluding both locks",
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    autocast_device = "cuda" if device.type == "cuda" else device.type
    best_selection = -1.0
    best_step = 0
    history: list[dict[str, Any]] = []
    started = time.time()
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    for step in range(1, args.steps + 1):
        model.train()
        lr = cosine_lr(step - 1, base_lr=args.learning_rate, total_steps=args.steps, warmup_steps=args.warmup_steps)
        set_optimizer_base_lr(optimizer, lr)
        successors = sample_training_permutations(
            batch_size=args.batch_size, node_count=cfg.node_count,
            forbidden_codes=forbidden_codes, device=device,
        )
        tokens, targets, _, _ = fixed_depth_batch(
            cfg, args.batch_size, device, path_positions=cfg.max_depth, successors=successors
        )
        order = torch.rand(len(tokens), cfg.node_count, device=device).argsort(-1)
        tokens[:,1:1+3*cfg.node_count] = tokens[:,1:1+3*cfg.node_count].reshape(-1,cfg.node_count,3).gather(1,order[:,:,None].expand(-1,-1,3)).reshape(len(tokens),-1)
        target = targets[:, -1]
        if args.depth_mode == "uniform":
            depth = torch.randint(1, cfg.max_depth + 1, (args.batch_size,), device=device)
            tokens[:, -2] = cfg.depth_token_base + depth - 1
            target = targets.gather(1, (depth - 1)[:, None]).squeeze(1)
        input_digest.update(tokens.detach().cpu().numpy().tobytes())
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"):
            all_logits = model.forward_all(tokens, max_loops=cfg.max_loops)["logits_by_loop"]
            logits = all_logits[:, -1]
            loss = F.cross_entropy(all_logits.reshape(-1,cfg.node_count),targets.reshape(-1)) if args.supervision == "per_loop" else F.cross_entropy(logits, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            selection_accuracy = _locked_endpoint_accuracy(
                model=model, cfg=cfg, lock=selection_lock, device=device,
                batch_size=args.eval_batch_size, amp=args.amp,
            )
            row = {"step": step, "lr": lr, "train_final_ce": float(loss.detach().cpu()),
                   "selection_endpoint_accuracy": selection_accuracy, "elapsed_sec": time.time() - started}
            history.append(row)
            _write_history(args.out_dir / "history.csv", history)
            (args.out_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
            if selection_accuracy > best_selection:
                best_selection, best_step = selection_accuracy, step
                _atomic_save({"protocol_id": PROTOCOL_ID, "model": model.state_dict(), "config": asdict(cfg),
                              "step": step, "selection_metrics": row, "excluded_locks": {
                                  "selection_sha256": sha256(args.selection_lock), "final_test_sha256": sha256(args.final_test_lock)},
                              "parameter_count": count_parameters(model)}, args.out_dir / "best.pt")
            print(f"[G4 step={step:05d}] loss={float(loss.detach().cpu()):.4f} selection_endpoint={selection_accuracy:.4f}", flush=True)
    final_selection = _locked_endpoint_accuracy(
        model=model, cfg=cfg, lock=selection_lock, device=device, batch_size=args.eval_batch_size, amp=args.amp
    )
    summary = {"status": "complete", "initial_parameter_sha256": initial_digest.hexdigest(), "training_input_sha256": input_digest.hexdigest(), "elapsed_sec": time.time()-started, "peak_memory_gib": torch.cuda.max_memory_allocated(device)/2**30 if device.type == "cuda" else 0, "protocol_id": PROTOCOL_ID, "best_step": best_step,
               "best_selection_endpoint_accuracy": best_selection,
               "final_selection_endpoint_accuracy": final_selection, **metadata}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _atomic_save({"protocol_id": PROTOCOL_ID, "model": model.state_dict(), "config": asdict(cfg),
                  "step": args.steps, "selection_metrics": {"endpoint_accuracy": final_selection},
                  "excluded_locks": {"selection_sha256": sha256(args.selection_lock),
                                      "final_test_sha256": sha256(args.final_test_lock)},
                  "parameter_count": count_parameters(model)}, args.out_dir / "final.pt")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supervision",choices=["final_only","per_loop"],required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--final-test-lock", type=Path, required=True)
    parser.add_argument("--extra-exclude-locks", nargs="*", type=Path, default=[])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--node-count", type=int, default=10)
    parser.add_argument("--loops", type=int, choices=[6,8], default=8)
    parser.add_argument("--depth-mode", choices=["uniform","fixed"], default="uniform")
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=1_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.3)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    torch.set_num_threads(4)
    train(parse_args())

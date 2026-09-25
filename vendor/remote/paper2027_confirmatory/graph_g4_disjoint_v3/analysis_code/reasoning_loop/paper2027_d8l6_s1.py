"""Matched pure-CE D8L6 stage-selection test (appendix-only case study).

Both requested stages start from the same raw h6, use the same D+AB+b
parameterization, initialization, update count, optimizer, and held-out graph
stream.  They differ only in whether the *post-executor* target is f9 or f10.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_loop import GraphPathConfig, pick_device, set_seed
from reasoning_loop.graph_path_temporal_intervention import logits_from_raw_state
from reasoning_loop.paper2027_graph_g3_controller import DiagonalLowRankGraphController
from scripts.evaluate_paper2027_graph_g1 import make_or_load_locked_test, write_csv


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(cfg: GraphPathConfig) -> None:
    expected = {"node_count": 8, "max_depth": 8, "d_model": 256, "n_heads": 4, "d_mlp": 1024, "n_layers": 2, "max_loops": 6, "block_schedule": "all_blocks"}
    mismatch = {key: (value, getattr(cfg, key)) for key, value in expected.items() if getattr(cfg, key) != value}
    if mismatch:
        raise ValueError(f"S1 requires D8L6 seed-class checkpoint: {mismatch}")


def initial(model: torch.nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    return model.token_embed(tokens) + model.pos_embed.unsqueeze(0)


def terminal_h6(model: torch.nn.Module, tokens: torch.Tensor, cfg: GraphPathConfig) -> torch.Tensor:
    state = initial(model, tokens)
    for loop in range(cfg.max_loops):
        state = model.apply_loop(state, loop_index=loop)
    return state


def target_for_hop(targets: torch.Tensor, cfg: GraphPathConfig, hop: int) -> torch.Tensor:
    if hop not in (1, 2):
        raise ValueError("target hop must be one or two")
    return targets[:, cfg.max_depth + hop - 1]


def payload(controller: DiagonalLowRankGraphController, checkpoint: Path, *, hop: int, seed: int, update: int, validation: float) -> dict[str, Any]:
    return {"kind": "paper2027_d8l6_s1", "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint), "target_hop": hop, "seed": seed, "update": update, "validation_post_executor_accuracy": validation, "parameterization": "h(D+AB)+b", "placement": "once before frozen F from raw h6", "objective": "pure post-executor CE", "controller_state_dict": {key: value.detach().cpu() for key, value in controller.state_dict().items()}}


@torch.no_grad()
def validate(model, cfg: GraphPathConfig, controller, *, hop: int, device: torch.device, seed: int, batches: int, batch_size: int) -> float:
    cpu_state = torch.get_rng_state(); cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    try:
        set_seed(seed); correct = total = 0
        for _ in range(batches):
            tokens, targets, _, _ = fixed_depth_batch(cfg, batch_size, device, path_positions=cfg.max_depth + 2)
            state = terminal_h6(model, tokens, cfg)
            output = model.apply_loop(controller(state), loop_index=cfg.max_loops)
            prediction = logits_from_raw_state(model, output).argmax(dim=-1)
            target = target_for_hop(targets, cfg, hop)
            correct += int(prediction.eq(target).sum()); total += batch_size
        return correct / total
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None: torch.cuda.set_rng_state(cuda_state, device)


def train(args: argparse.Namespace) -> dict[str, Any]:
    device = pick_device(args.device); model, cfg, _ = load_checkpoint(args.checkpoint, device); verify(cfg)
    model.requires_grad_(False)
    controller = DiagonalLowRankGraphController(cfg.d_model, args.rank).to(device)
    optimizer = torch.optim.AdamW(controller.parameters(), lr=args.learning_rate, weight_decay=0.0)
    set_seed(args.seed); best = -math.inf; rows = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for update in range(1, args.updates + 1):
        tokens, targets, _, _ = fixed_depth_batch(cfg, args.batch_size, device, path_positions=cfg.max_depth + 2)
        with torch.no_grad(): state = terminal_h6(model, tokens, cfg)
        output = model.apply_loop(controller(state), loop_index=cfg.max_loops)
        loss = F.cross_entropy(logits_from_raw_state(model, output).float(), target_for_hop(targets, cfg, args.target_hop))
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(controller.parameters(), args.grad_clip); optimizer.step()
        if update % args.validation_every == 0 or update == args.updates:
            accuracy = validate(model, cfg, controller, hop=args.target_hop, device=device, seed=args.validation_seed, batches=args.validation_batches, batch_size=args.validation_batch_size)
            rows.append({"update": update, "pure_post_executor_ce": float(loss.detach()), "validation_post_executor_accuracy": accuracy})
            if accuracy > best:
                best = accuracy; torch.save(payload(controller, args.checkpoint, hop=args.target_hop, seed=args.seed, update=update, validation=accuracy), args.out_dir / "best_controller.pt")
    torch.save(payload(controller, args.checkpoint, hop=args.target_hop, seed=args.seed, update=args.updates, validation=accuracy), args.out_dir / "final_controller.pt")
    write_csv(args.out_dir / "training.csv", rows)
    result = {"status": "complete", "protocol_id": "paper2027.d8l6.s1.matched_pure_ce.v1", "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint), "target_hop": args.target_hop, "controller_seed": args.seed, "rank": args.rank, "updates": args.updates, "validation": accuracy}
    (args.out_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); return result


def load_controller(path: Path, checkpoint: Path, device: torch.device) -> tuple[DiagonalLowRankGraphController, dict[str, Any]]:
    item = torch.load(path, map_location="cpu", weights_only=False)
    if item.get("kind") != "paper2027_d8l6_s1" or item.get("checkpoint_sha256") != sha256(checkpoint): raise ValueError("invalid or mismatched S1 controller")
    controller = DiagonalLowRankGraphController(256, int(item["controller_state_dict"]["A"].shape[1])).to(device); controller.load_state_dict(item["controller_state_dict"]); controller.eval(); return controller, item


def _counts(pred: torch.Tensor, endpoint: torch.Tensor, one: torch.Tensor, two: torch.Tensor) -> dict[str, int]:
    keep = endpoint.ne(one) & endpoint.ne(two) & one.ne(two)
    values = pred[keep]
    return {"examples": int(pred.numel()), "distinct_examples": int(keep.sum()), "endpoint": int(values.eq(endpoint[keep]).sum()), "one": int(values.eq(one[keep]).sum()), "two": int(values.eq(two[keep]).sum()), "other": int((values.ne(endpoint[keep]) & values.ne(one[keep]) & values.ne(two[keep])).sum())}


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    data: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        for key, value in row.items():
            if key not in {"permutation", "mode", "readout"}: data[(row["mode"], row["readout"])][key] += int(value)
    return [{"mode": mode, "readout": readout, **values, **{f"{name}_fraction": values[name] / values["distinct_examples"] for name in ("endpoint", "one", "two", "other")}} for (mode, readout), values in sorted(data.items())]


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = pick_device(args.device); model, cfg, _ = load_checkpoint(args.checkpoint, device); verify(cfg)
    controller, item = load_controller(args.controller, args.checkpoint, device)
    locked = make_or_load_locked_test(cfg=cfg, path=args.locked_test, permutations=args.permutations, seed=args.test_seed)
    rows = []; group = args.batch_size // cfg.node_count
    for first in range(0, args.permutations, group):
        last = min(args.permutations, first + group); successors = locked["successors"][first:last].repeat_interleave(cfg.node_count, 0).to(device); starts = torch.arange(cfg.node_count, device=device).repeat(last - first)
        tokens, targets, _, _ = fixed_depth_batch(cfg, successors.shape[0], device, path_positions=cfg.max_depth + 2, successors=successors, start=starts)
        state = terminal_h6(model, tokens, cfg); boundary = controller(state)
        variants = {"raw": state, "full": boundary, "batch_shuffle": boundary.reshape(last-first,cfg.node_count,*boundary.shape[1:]).roll(1,0).reshape_as(boundary)}
        endpoint, one, two = targets[:, 7], targets[:, 8], targets[:, 9]
        for mode, value in variants.items():
            for readout, output in (("pre_executor", value), ("post_executor", model.apply_loop(value, loop_index=cfg.max_loops))):
                pred = logits_from_raw_state(model, output).argmax(dim=-1)
                for local in range(last-first):
                    begin, end = local*cfg.node_count, (local+1)*cfg.node_count; rows.append({"permutation": first+local, "mode": mode, "readout": readout, **_counts(pred[begin:end], endpoint[begin:end], one[begin:end], two[begin:end])})
    aggregate = summarize(rows); args.out_dir.mkdir(parents=True, exist_ok=True); write_csv(args.out_dir / "permutation_clusters.csv", rows); write_csv(args.out_dir / "aggregate.csv", aggregate)
    result = {"status":"complete","protocol_id":"paper2027.d8l6.s1.matched_pure_ce.v1","checkpoint":str(args.checkpoint),"controller":str(args.controller),"controller_target_hop":item["target_hop"],"locked_test":str(args.locked_test),"locked_test_sha256":sha256(args.locked_test),"aggregate_rows":aggregate,"claim_boundary":"single-backbone appendix case study; post-executor target is the only one-vs-two training difference"}
    (args.out_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n",encoding="utf-8"); return result


def parse_args() -> argparse.Namespace:
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest="action",required=True)
    train_p=sub.add_parser("train"); train_p.add_argument("--checkpoint",type=Path,required=True); train_p.add_argument("--out-dir",type=Path,required=True); train_p.add_argument("--target-hop",type=int,choices=(1,2),required=True); train_p.add_argument("--seed",type=int,required=True); train_p.add_argument("--rank",type=int,default=48); train_p.add_argument("--updates",type=int,default=8000); train_p.add_argument("--batch-size",type=int,default=128); train_p.add_argument("--learning-rate",type=float,default=1e-4); train_p.add_argument("--grad-clip",type=float,default=1.0); train_p.add_argument("--validation-every",type=int,default=400); train_p.add_argument("--validation-seed",type=int,required=True); train_p.add_argument("--validation-batches",type=int,default=8); train_p.add_argument("--validation-batch-size",type=int,default=256); train_p.add_argument("--device",default="cuda")
    eval_p=sub.add_parser("evaluate"); eval_p.add_argument("--checkpoint",type=Path,required=True); eval_p.add_argument("--controller",type=Path,required=True); eval_p.add_argument("--locked-test",type=Path,required=True); eval_p.add_argument("--out-dir",type=Path,required=True); eval_p.add_argument("--permutations",type=int,default=512); eval_p.add_argument("--test-seed",type=int,default=2026099501); eval_p.add_argument("--batch-size",type=int,default=256); eval_p.add_argument("--device",default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    arguments=parse_args(); print(json.dumps(train(arguments) if arguments.action=="train" else evaluate(arguments),indent=2,sort_keys=True))

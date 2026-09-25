"""Evaluate a trained age-specific J bank with matched routing controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from reasoning_loop.graph_path_depth_circuit import load_checkpoint
from reasoning_loop.graph_path_loop import pick_device
from reasoning_loop.train_graph_path_age_specific_j_bank import (
    AgeSpecificJBank,
    _write_csv,
    evaluate_products,
    evaluate_random_trajectories,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--bank-artifact", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=820001)
    parser.add_argument("--trajectories", type=int, default=56)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-max-backs", type=int, default=28)
    parser.add_argument("--unseen-max-backs", type=int, default=40)
    parser.add_argument("--train-max-total-backs", type=int)
    parser.add_argument("--unseen-max-total-backs", type=int)
    parser.add_argument("--max-consecutive-backs", type=int)
    parser.add_argument("--composition-examples", type=int, default=256)
    parser.add_argument("--composition-batch-size", type=int, default=64)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=(
            "learned",
            "exact",
            "identity",
            "wrong_stage",
            "reverse_stage",
            "shared_J8",
        ),
    )
    return parser.parse_args()


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    model.eval()
    phase = json.loads(args.phase_summary.read_text(encoding="utf-8"))
    phase_positions = [
        int(value) for value in phase["trajectory_positions_including_initial"]
    ]
    payload = torch.load(args.bank_artifact, map_location="cpu", weights_only=False)
    bank = AgeSpecificJBank(
        dimension=cfg.d_model,
        rank=int(payload["rank"]),
        map_architecture=payload.get("map_architecture", "diagonal_lora"),
    )
    bank.load_state_dict(payload["state_dict"])
    bank = bank.to(device).frozen()
    positions = tuple(int(value) for value in payload["positions"])
    rows = []
    summary = []
    for split, max_extra, max_total, offset in (
        (
            "train_like",
            None if args.train_max_total_backs is not None else args.train_max_backs,
            args.train_max_total_backs,
            1,
        ),
        (
            "longer_unseen",
            None if args.unseen_max_total_backs is not None else args.unseen_max_backs,
            args.unseen_max_total_backs,
            2,
        ),
    ):
        split_rows, split_summary = evaluate_random_trajectories(
            model=model,
            cfg=cfg,
            bank=bank,
            phase_positions=phase_positions,
            positions=positions,
            device=device,
            trajectories=args.trajectories,
            batch_size=args.batch_size,
            max_extra_backs=max_extra,
            max_total_backs=max_total,
            max_consecutive_backs=args.max_consecutive_backs,
            seed=args.seed + offset,
            split=split,
            conditions=tuple(args.conditions),
        )
        rows.extend(split_rows)
        summary.extend(split_summary)
    products = evaluate_products(
        model=model,
        cfg=cfg,
        bank=bank,
        phase_positions=phase_positions,
        positions=positions,
        device=device,
        examples=args.composition_examples,
        batch_size=args.composition_batch_size,
        seed=args.seed + 3,
    )
    _write_csv(args.out_dir / "random_trajectory_evaluation.csv", rows)
    _write_csv(args.out_dir / "random_trajectory_summary.csv", summary)
    _write_csv(args.out_dir / "matrix_product_evaluation.csv", products)
    result = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "bank_artifact": str(args.bank_artifact),
        "bank_initialization": payload.get("initialization"),
        "backbone_loss_placement": "final-only CE at loop 8",
        "trained_loop_count": cfg.max_loops,
        "shared_block_count": cfg.n_layers,
        "effective_training_depth": cfg.max_loops * cfg.n_layers,
        "summary": summary,
        "products": products,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(parse_args())

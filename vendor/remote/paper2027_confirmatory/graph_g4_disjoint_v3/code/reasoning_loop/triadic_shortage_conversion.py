from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from reasoning_loop.triadic_shortage import all_triples
from reasoning_loop.triadic_shortage_diagnostics import (
    collect_exhaustive_states,
    evaluate_hybrid_patch,
    evaluate_order_sweep,
    evaluate_reset,
    fourier_bucket_fractions,
    load_checkpoint,
)
from reasoning_loop.triadic_shortage_train import evaluate, pick_device


@torch.no_grad()
def analyze_conversion_checkpoint(
    checkpoint: Path,
    out_dir: Path,
    *,
    device: torch.device,
    sample_size: int = 1024,
    batch_size: int = 1024,
    seed: int = 101_001,
) -> dict[str, Any]:
    model, payload = load_checkpoint(checkpoint, device)
    operands, labels = all_triples(model.cfg.p, device=device)
    heldout_idx = payload["split"]["heldout_idx"].to(device)
    subset_idx = heldout_idx[: min(sample_size, heldout_idx.numel())]
    subset_operands = operands[subset_idx]
    subset_labels = labels[subset_idx]
    behavior = {
        condition: evaluate(
            model,
            operands,
            labels,
            heldout_idx,
            condition=condition,
            batch_size=batch_size,
            seed=seed + offset,
        )
        for offset, condition in enumerate(("full", "full_once", "sequential"))
    }
    order_sweep = evaluate_order_sweep(
        model,
        operands,
        labels,
        heldout_idx,
        batch_size=batch_size,
    )
    reset = evaluate_reset(
        model,
        subset_operands,
        subset_labels,
        condition="sequential",
        seed=seed + 10,
    )
    hybrid = evaluate_hybrid_patch(
        model,
        donor_operands=subset_operands,
        receiver_operands=subset_operands.roll(1, dims=0),
        condition="sequential",
        seed=seed + 11,
    )
    _, _, states = collect_exhaustive_states(
        model,
        condition="sequential",
        batch_size=batch_size,
        seed=seed + 12,
    )
    fourier = []
    for loop_index in range(model.cfg.loops):
        row: dict[str, Any] = {"loop": loop_index + 1}
        row.update(
            fourier_bucket_fractions(
                states[:, loop_index].numpy().astype(np.float64),
                model.cfg.p,
            )
        )
        fourier.append(row)
    hybrid_rows = hybrid["rows"]
    scorecard = {
        "checkpoint": str(checkpoint.resolve()),
        "run_name": checkpoint.parent.name,
        "trained_condition": payload["condition"],
        "seed": int(payload["seed"]),
        "step": int(payload["step"]),
        "full_accuracy": behavior["full"]["final_accuracy"],
        "full_once_accuracy": behavior["full_once"]["final_accuracy"],
        "sequential_accuracy": behavior["sequential"]["final_accuracy"],
        "order_mean_accuracy": order_sweep["mean_accuracy"],
        "order_min_accuracy": order_sweep["min_accuracy"],
        "order_max_accuracy": order_sweep["max_accuracy"],
        "reset_effect": reset["reset_effect"],
        "reset_accuracy": reset["reset_accuracy"],
        "hybrid_accuracy_mean": float(
            np.mean([row["hybrid_accuracy"] for row in hybrid_rows])
        ),
        "random_workspace_accuracy_mean": float(
            np.mean([row["random_workspace_accuracy"] for row in hybrid_rows])
        ),
        "loop6_final_sum_fraction": fourier[-1]["final_sum_fraction"],
        "loop3_single_total_fraction": fourier[min(2, len(fourier) - 1)][
            "single_total_fraction"
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("behavior.json", behavior),
        ("order_sweep.json", order_sweep),
        ("reset.json", reset),
        ("hybrid_patch.json", hybrid),
        ("fourier_trajectory.json", fourier),
        ("scorecard.json", scorecard),
    ):
        (out_dir / name).write_text(
            json.dumps(value, indent=2, allow_nan=False),
            encoding="utf-8",
        )
    return scorecard


def aggregate_conversion(analysis_root: Path, out_dir: Path) -> dict[str, Any]:
    cards = []
    for path in sorted(analysis_root.glob("*/scorecard.json")):
        card = json.loads(path.read_text(encoding="utf-8"))
        card["analysis_name"] = path.parent.name
        cards.append(card)
    source_names = {
        "full_to_sequential": "info_full_d64_L6_seed{seed}",
        "full_to_full": "info_full_d64_L6_seed{seed}",
        "sequential_to_full": "info_sequential_shuffled_d64_L6_seed{seed}",
        "sequential_to_sequential": "info_sequential_shuffled_d64_L6_seed{seed}",
    }
    trajectories: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for direction, source_template in source_names.items():
        trajectories[direction] = {}
        for seed in (0, 1, 2):
            source_name = source_template.format(seed=seed)
            source = [
                card
                for card in cards
                if card["run_name"] == source_name and int(card["step"]) == 4000
            ]
            switched = [
                card
                for card in cards
                if card["run_name"] == f"switch_{direction}_seed{seed}"
            ]
            trajectories[direction][str(seed)] = sorted(
                [*source, *switched],
                key=lambda card: int(card["step"]),
            )

    def final_cards(direction: str) -> list[dict[str, Any]]:
        output = []
        for seed in (0, 1, 2):
            rows = trajectories[direction][str(seed)]
            if rows:
                output.append(max(rows, key=lambda card: int(card["step"])))
        return output

    full_to_seq = final_cards("full_to_sequential")
    seq_to_full = final_cards("sequential_to_full")
    full_to_full = final_cards("full_to_full")
    seq_to_seq = final_cards("sequential_to_sequential")
    ledger = {
        "full_to_sequential": {
            "status": "causal_pass"
            if len(full_to_seq) == 3
            and all(
                card["order_mean_accuracy"] >= 0.95
                and card["reset_effect"] >= 0.80
                and card["hybrid_accuracy_mean"] >= 0.90
                for card in full_to_seq
            )
            else "open",
            "final_order_mean_accuracy": [
                card["order_mean_accuracy"] for card in full_to_seq
            ],
            "final_reset_effect": [card["reset_effect"] for card in full_to_seq],
            "final_hybrid_accuracy": [
                card["hybrid_accuracy_mean"] for card in full_to_seq
            ],
        },
        "sequential_to_full": {
            "status": "behavior_pass"
            if len(seq_to_full) == 3
            and all(card["full_accuracy"] >= 0.95 for card in seq_to_full)
            else "open",
            "final_full_accuracy": [card["full_accuracy"] for card in seq_to_full],
            "retained_order_mean_accuracy": [
                card["order_mean_accuracy"] for card in seq_to_full
            ],
            "retained_reset_effect": [card["reset_effect"] for card in seq_to_full],
        },
        "controls": {
            "full_to_full_accuracy": [card["full_accuracy"] for card in full_to_full],
            "sequential_to_sequential_order_accuracy": [
                card["order_mean_accuracy"] for card in seq_to_seq
            ],
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "trajectories.json").write_text(
        json.dumps(trajectories, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (out_dir / "conversion_ledger.json").write_text(
        json.dumps(ledger, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return {"scorecards": len(cards), "ledger": ledger}


def discover_conversion_checkpoints(run_root: Path) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    for seed in (0, 1, 2):
        for condition in ("full", "sequential_shuffled"):
            run_name = f"info_{condition}_d64_L6_seed{seed}"
            checkpoint = run_root / "runs" / run_name / "checkpoint_step_0004000.pt"
            if checkpoint.exists():
                found.append((f"source_{condition}_seed{seed}_step4000", checkpoint))
    for run_dir in sorted((run_root / "runs").glob("switch_*_seed*")):
        for step in (6000, 8000, 10000):
            checkpoint = run_dir / f"checkpoint_step_{step:07d}.pt"
            if checkpoint.exists():
                found.append((f"{run_dir.name}_step{step}", checkpoint))
    return found


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-schedule conversion diagnostics.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--aggregate-root", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sample-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=101_001)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.aggregate_root is not None:
        result = aggregate_conversion(args.aggregate_root, args.out_dir)
    else:
        result = analyze_conversion_checkpoint(
            args.checkpoint,
            args.out_dir,
            device=pick_device(args.device),
            sample_size=args.sample_size,
            batch_size=args.batch_size,
            seed=args.seed,
        )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Large matched multi-seed evaluation of one-step and compositional J behavior."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from reasoning_loop.graph_path_depth_circuit import load_checkpoint
from reasoning_loop.graph_path_loop import pick_device
from reasoning_loop.train_graph_path_age_specific_j_bank import AgeSpecificJBank
from reasoning_loop.train_graph_path_j_path_equivalence import evaluate_fixed_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bank", action="append", required=True, help="label=artifact.pt")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", default=(0, 1, 2, 3, 4))
    parser.add_argument("--seed-base", type=int, default=830001)
    parser.add_argument("--examples-per-seed", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pair-count", type=int, default=7)
    parser.add_argument("--back-counts", type=int, nargs="+", default=(2, 3, 5, 8, 12, 16, 24))
    parser.add_argument(
        "--path-generation-prefix-back-counts",
        type=int,
        nargs="*",
        default=(),
        help=(
            "Consume action-word RNG for these k values before --back-counts. "
            "This reproduces a slice of a larger ordered evaluation suite."
        ),
    )
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.02)
    parser.add_argument(
        "--allow-relocated-checkpoint",
        action="store_true",
        help=(
            "Allow a bank whose recorded backbone path differs from --checkpoint. "
            "Use only for a byte-preserved local copy; both paths are recorded."
        ),
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["bank"], row["family"], int(row["back_count"]))].append(row)
    output = []
    for (bank, family, back_count), values in groups.items():
        accuracies = np.asarray(
            [float(row[side]) for row in values for side in ("left_accuracy", "right_accuracy")]
        )
        agreements = np.asarray([float(row["prediction_agreement"]) for row in values])
        output.append(
            {
                "bank": bank,
                "family": family,
                "back_count": back_count,
                "evaluated_path_sides": len(accuracies),
                "examples_per_path_side": int(values[0]["examples"]),
                "accuracy_mean": float(accuracies.mean()),
                "accuracy_sem_across_paths": float(
                    accuracies.std(ddof=1) / np.sqrt(len(accuracies))
                    if len(accuracies) > 1 else 0.0
                ),
                "accuracy_min": float(accuracies.min()),
                "accuracy_max": float(accuracies.max()),
                "prediction_agreement_mean": float(agreements.mean()),
                "prediction_agreement_min": float(agreements.min()),
            }
        )
    return sorted(output, key=lambda row: (row["family"], row["bank"], row["back_count"]))


def plot(summary: list[dict[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=180)
    for bank in dict.fromkeys(row["bank"] for row in summary):
        selected = sorted(
            [row for row in summary if row["bank"] == bank and row["family"] == "equivalent_words"],
            key=lambda row: row["back_count"],
        )
        axes[0].errorbar(
            [row["back_count"] for row in selected],
            [row["accuracy_mean"] for row in selected],
            yerr=[row["accuracy_sem_across_paths"] for row in selected],
            marker="o", capsize=2, label=bank,
        )
        axes[1].plot(
            [row["back_count"] for row in selected],
            [row["prediction_agreement_mean"] for row in selected],
            marker="o", label=bank,
        )
    for axis, title in zip(
        axes,
        ("functional accuracy on unseen action words", "agreement of equivalent action words"),
        strict=True,
    ):
        axis.axhline(0.95, color="gray", linestyle=":", linewidth=0.9)
        axis.set_ylim(0, 1.03)
        axis.set_xlabel("J calls k")
        axis.set_title(title)
        axis.grid(alpha=0.22)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.examples_per_seed % args.batch_size:
        raise ValueError("examples-per-seed must be divisible by batch-size")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
        torch.cuda.reset_peak_memory_stats()
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    positions = tuple(range(cfg.seq_len))
    banks: dict[str, AgeSpecificJBank] = {}
    artifacts: dict[str, str] = {}
    recorded_bank_checkpoints: dict[str, str] = {}
    for item in args.bank:
        label, raw_path = item.split("=", 1)
        path = Path(raw_path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        recorded_checkpoint = str(payload.get("checkpoint"))
        recorded_bank_checkpoints[label] = recorded_checkpoint
        if (
            recorded_checkpoint != str(args.checkpoint)
            and not args.allow_relocated_checkpoint
        ):
            raise ValueError(f"{label} bank/backbone mismatch")
        bank = AgeSpecificJBank(
            dimension=cfg.d_model,
            rank=int(payload["rank"]),
            stage_rank=int(payload.get("stage_rank", payload["rank"])),
            map_architecture=str(payload["map_architecture"]),
        ).to(device)
        bank.load_state_dict(payload["state_dict"])
        banks[label] = bank.frozen()
        artifacts[label] = str(path)
    rows: list[dict[str, Any]] = []
    for seed_offset in args.evaluation_seeds:
        seed = args.seed_base + int(seed_offset)
        for label, bank in banks.items():
            current = evaluate_fixed_suite(
                model=model,
                cfg=cfg,
                bank=bank,
                positions=positions,
                device=device,
                seed=seed,
                examples=args.examples_per_seed,
                batch_size=args.batch_size,
                pair_count=args.pair_count,
                path_generation_prefix_back_counts=tuple(
                    args.path_generation_prefix_back_counts
                ),
                long_back_counts=tuple(args.back_counts),
                label=f"{label}_seed{seed}",
            )
            rows.extend({"bank": label, "evaluation_seed": seed, **row} for row in current)
    summary = aggregate(rows)
    write_csv(args.out_dir / "per_path.csv", rows)
    write_csv(args.out_dir / "aggregate.csv", summary)
    plot(summary, args.out_dir / "multiseed_composition.png")
    lifespan = {}
    for label in banks:
        selected = [
            row for row in summary
            if row["bank"] == label and row["family"] == "equivalent_words"
        ]
        passing = [row["back_count"] for row in selected if row["accuracy_mean"] >= 0.95]
        lifespan[label] = max(passing) if passing else 0
    result = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "checkpoint_step": checkpoint_payload.get("step"),
        "bank_artifacts": artifacts,
        "recorded_bank_checkpoints": recorded_bank_checkpoints,
        "relocated_checkpoint_override": bool(args.allow_relocated_checkpoint),
        "evaluation_seeds": [args.seed_base + int(value) for value in args.evaluation_seeds],
        "examples_per_seed": args.examples_per_seed,
        "pair_count_per_k_per_seed": args.pair_count,
        "path_generation_prefix_back_counts": list(
            args.path_generation_prefix_back_counts
        ),
        "action_words": "new deterministic samples disjoint from online training RNG streams",
        "functional_lifespan_at_mean_accuracy_0.95": lifespan,
        "summary": summary,
        "peak_cuda_allocated_mib": (
            torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

"""Cross-evaluate J banks trained under different composition laws."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from reasoning_loop.graph_path_depth_circuit import load_checkpoint
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.train_graph_path_age_specific_j_bank import (
    AgeSpecificJBank,
    evaluate_random_trajectories,
)


LAWS = (
    "product",
    "diag_gated_residual",
    "diag_left_gated_residual",
    "diag_product_u_sum",
    "pure_residual",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument(
        "--bank", action="append", required=True,
        help="Repeated label=/absolute/path/to/artifact.pt",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trajectories", type=int, default=56)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=820001)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.02)
    return parser.parse_args()


def load_bank(path: Path, *, dimension: int, device: torch.device) -> AgeSpecificJBank:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    bank = AgeSpecificJBank(
        dimension=dimension,
        rank=int(payload["rank"]),
        stage_rank=int(payload.get("stage_rank", payload["rank"])),
        map_architecture=payload["map_architecture"],
    )
    bank.load_state_dict(payload["state_dict"])
    return bank.to(device).frozen()


def specs() -> tuple[dict, ...]:
    return (
        dict(split="single_J", total=1, run=1, minimum=1, required=1, mandatory=True, offset=1),
        dict(split="focused_mixture_a", total=5, run=5, minimum=None, required=0, mandatory=True, offset=2),
        dict(split="focused_mixture_b", total=5, run=5, minimum=None, required=0, mandatory=True, offset=3),
        dict(split="hard_boundary_T05_R5", total=5, run=5, minimum=5, required=5, mandatory=False, offset=4),
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict], labels: list[str], path: Path) -> None:
    by = {(row["trained_law"], row["evaluation_law"], row["split"]): row["accuracy_mean"] for row in rows}
    figures = (
        ("single J", lambda train, law: by[(train, law, "single_J")]),
        (
            "mixed trajectories",
            lambda train, law: np.mean([
                by[(train, law, "focused_mixture_a")],
                by[(train, law, "focused_mixture_b")],
            ]),
        ),
        ("five consecutive J", lambda train, law: by[(train, law, "hard_boundary_T05_R5")]),
    )
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.7))
    for axis, (title, value) in zip(axes, figures, strict=True):
        matrix = np.array([[value(train, law) for law in LAWS] for train in labels])
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
        axis.set_xticks(range(len(LAWS)), LAWS, rotation=35, ha="right")
        axis.set_yticks(range(len(labels)), labels)
        axis.set_title(title)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                color = "white" if matrix[row, column] < 0.55 else "black"
                axis.text(column, row, f"{matrix[row, column]:.3f}", ha="center", va="center", color=color, fontsize=8)
    figure.suptitle("Composition-law cross evaluation")
    figure.subplots_adjust(left=0.08, right=0.91, bottom=0.28, top=0.84, wspace=0.35)
    color_axis = figure.add_axes((0.935, 0.24, 0.012, 0.56))
    figure.colorbar(image, cax=color_axis, label="accuracy")
    figure.savefig(path, dpi=190)
    plt.close(figure)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
        torch.cuda.reset_peak_memory_stats()
    set_seed(args.seed)
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    model.eval()
    phase = json.loads(args.phase_summary.read_text())
    phase_positions = [int(value) for value in phase["trajectory_positions_including_initial"]]
    positions = tuple(range(cfg.seq_len))
    banks = []
    for item in args.bank:
        label, raw_path = item.split("=", 1)
        banks.append((label, Path(raw_path)))
    rows = []
    for label, path in banks:
        bank = load_bank(path, dimension=cfg.d_model, device=device)
        for law in LAWS:
            for spec in specs():
                _, summary = evaluate_random_trajectories(
                    model=model,
                    cfg=cfg,
                    bank=bank,
                    phase_positions=phase_positions,
                    positions=positions,
                    device=device,
                    trajectories=args.trajectories,
                    batch_size=args.batch_size,
                    max_extra_backs=None,
                    max_total_backs=spec["total"],
                    max_consecutive_backs=spec["run"],
                    minimum_total_backs=spec["minimum"],
                    required_consecutive_backs=spec["required"],
                    schedule_mandatory_j=spec["mandatory"],
                    seed=args.seed + spec["offset"],
                    split=spec["split"],
                    conditions=("learned",),
                    rollback_composition=law,
                )
                row = dict(summary[0])
                row.update(trained_law=label, evaluation_law=law, bank_artifact=str(path))
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
        del bank
    write_csv(args.out_dir / "composition_cross_evaluation.csv", rows)
    plot(rows, [label for label, _ in banks], args.out_dir / "composition_cross_evaluation.png")
    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "trajectories_per_split": args.trajectories,
        "examples_per_trajectory": args.batch_size,
        "trained_laws": [label for label, _ in banks],
        "evaluation_laws": list(LAWS),
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

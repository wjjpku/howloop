"""Replay fixed-H1 curriculum batches and measure diagonal-only gradients."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_simple_one_step import _aligned_state_at_age
from reasoning_loop.train_graph_path_age_specific_j_bank import (
    AgeSpecificJBank,
    ROLLBACK_SOURCE_AGES,
    _run_mixed_trajectory,
    sample_bounded_bridge,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batches", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=932001)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.02)
    return parser.parse_args()


STAGES = (
    ("start_single", None, 1, 1),
    ("start_T02_R1", "single", 2, 1),
    ("start_T02_R2", "T02_R1", 2, 2),
    ("start_T03_R2", "T02_R2", 3, 2),
    ("start_T03_R3", "T03_R2", 3, 3),
    ("start_T04_R3", "T03_R3", 4, 3),
    ("start_T04_R4", "T04_R3", 4, 4),
    ("start_T05_R4", "T04_R4", 5, 4),
    ("start_T05_R5", "T05_R4", 5, 5),
    ("after_T05_R5", "T05_R5", 5, 5),
)


def artifact_path(root: Path, checkpoint_label: str) -> Path:
    if checkpoint_label == "single":
        return root.parent / "single_h1" / "age_specific_j_bank_after_T01_R1.pt"
    return root / f"age_specific_j_bank_after_{checkpoint_label}.pt"


def load_bank(
    *, root: Path, checkpoint_label: str | None, dimension: int, device: torch.device
) -> AgeSpecificJBank:
    if checkpoint_label is None:
        return AgeSpecificJBank(
            dimension=dimension,
            rank=48,
            stage_rank=16,
            map_architecture="shared_diagonal_stage_lora",
        ).to(device)
    payload = torch.load(
        artifact_path(root, checkpoint_label), map_location="cpu", weights_only=False
    )
    bank = AgeSpecificJBank(
        dimension=dimension,
        rank=int(payload["rank"]),
        stage_rank=int(payload["stage_rank"]),
        map_architecture=payload["map_architecture"],
    ).to(device)
    bank.load_state_dict(payload["state_dict"])
    return bank


def summarize(values: list[float], prefix: str) -> dict[str, float]:
    array = np.asarray(values)
    return {
        f"{prefix}_mean": float(array.mean()),
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_p90": float(np.quantile(array, 0.90)),
        f"{prefix}_max": float(array.max()),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.batches % 14:
        raise ValueError("batches must be a multiple of 14")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
    set_seed(args.seed)
    model, cfg, _ = load_checkpoint(args.checkpoint, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    phase_payload = json.loads(args.phase_summary.read_text(encoding="utf-8"))
    phase_positions = [
        int(value)
        for value in phase_payload["trajectory_positions_including_initial"]
    ]
    positions = tuple(range(cfg.seq_len))
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for stage_index, (label, checkpoint_label, total_cap, run_cap) in enumerate(STAGES):
        set_seed(args.seed + 100 * stage_index)
        rng = np.random.default_rng(args.seed + 100 * stage_index)
        bank = load_bank(
            root=args.bank_root,
            checkpoint_label=checkpoint_label,
            dimension=cfg.d_model,
            device=device,
        )
        bank.train()
        for batch_index in range(args.batches):
            hard = batch_index % 14 >= 7
            mandatory = None if hard else ROLLBACK_SOURCE_AGES[batch_index % 7]
            trajectory = sample_bounded_bridge(
                rng=rng,
                max_total_backs=total_cap,
                minimum_total_backs=total_cap if hard else None,
                mandatory_rollback_source=mandatory,
                max_consecutive_backs=run_cap,
                required_consecutive_backs=run_cap if hard else 0,
                fixed_start_age=1,
            )
            with torch.no_grad():
                _, path_targets, successors, _ = fixed_depth_batch(
                    cfg, args.batch_size, device, path_positions=cfg.max_depth
                )
                endpoint = path_targets[:, cfg.max_depth - 1]
                initial_state = _aligned_state_at_age(
                    model=model,
                    cfg=cfg,
                    successors=successors,
                    current=endpoint,
                    age=1,
                    phase_position=phase_positions[1],
                )
            bank.zero_grad(set_to_none=True)
            _, logits, target = _run_mixed_trajectory(
                model=model,
                cfg=cfg,
                bank=bank,
                positions=positions,
                successors=successors,
                endpoint=endpoint,
                initial_state=initial_state,
                trajectory=trajectory,
                condition="learned",
                phase_positions=phase_positions,
                rollback_composition="product",
            )
            loss = F.cross_entropy(logits.float(), target)
            loss.backward()
            gradient = bank.shared_diagonal_scale.grad.detach().float()
            all_gradients = [
                parameter.grad.detach().float().reshape(-1)
                for parameter in bank.parameters()
                if parameter.grad is not None
            ]
            total_gradient = torch.cat(all_gradients)
            total_gradient_l2 = float(total_gradient.norm())
            clip_scale = min(1.0, 1.0 / max(total_gradient_l2, 1e-12))
            rows.append(
                {
                    "stage": label,
                    "checkpoint_label": checkpoint_label or "identity",
                    "batch": batch_index,
                    "hard_boundary": hard,
                    "back_count": trajectory.back_count,
                    "max_rollback_run": trajectory.max_rollback_run,
                    "loss": float(loss.detach()),
                    "accuracy": float(logits.argmax(-1).eq(target).float().mean()),
                    "diagonal_grad_l2": float(gradient.norm()),
                    "diagonal_grad_rms": float(gradient.square().mean().sqrt()),
                    "diagonal_grad_mean_abs": float(gradient.abs().mean()),
                    "diagonal_grad_max_abs": float(gradient.abs().max()),
                    "diagonal_grad_signed_mean": float(gradient.mean()),
                    "all_parameter_grad_l2": total_gradient_l2,
                    "global_clip_scale_at_norm_1": clip_scale,
                    "clipped_diagonal_grad_l2": float(gradient.norm()) * clip_scale,
                    "clipped_diagonal_grad_max_abs": float(gradient.abs().max()) * clip_scale,
                }
            )
        selected = [row for row in rows if row["stage"] == label]
        summary: dict[str, Any] = {
            "stage": label,
            "checkpoint_label": checkpoint_label or "identity",
            "batches": len(selected),
            "examples": len(selected) * args.batch_size,
            "accuracy_mean": float(np.mean([row["accuracy"] for row in selected])),
            "loss_mean": float(np.mean([row["loss"] for row in selected])),
        }
        for key in (
            "diagonal_grad_l2",
            "diagonal_grad_rms",
            "diagonal_grad_mean_abs",
            "diagonal_grad_max_abs",
            "all_parameter_grad_l2",
            "global_clip_scale_at_norm_1",
            "clipped_diagonal_grad_l2",
            "clipped_diagonal_grad_max_abs",
        ):
            summary.update(summarize([row[key] for row in selected], key))
        summaries.append(summary)
        del bank
    write_csv(args.out_dir / "diagonal_gradient_batches.csv", rows)
    write_csv(args.out_dir / "diagonal_gradient_summary.csv", summaries)
    (args.out_dir / "summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "training_start_age": 1,
                "batches_per_checkpoint": args.batches,
                "examples_per_batch": args.batch_size,
                "summaries": summaries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()

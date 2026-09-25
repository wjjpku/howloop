#!/usr/bin/env python3
"""Select a held-out-data snapshot from a far-range Addition J run.

Selection is confined to logical lengths that were exposed during controller
training.  Longer lengths remain unopened for the final extrapolation audit.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from reasoning_loop.paper_length_telomere import (
    ControllerView,
    load_backbone,
    load_controller,
    pick_device,
)
from scripts.evaluate_addition_controller_endpoint_accuracy import evaluate_variant


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--controller-dir", type=Path, required=True)
    parser.add_argument("--validation-lengths", type=int, nargs="+", default=tuple(range(10, 21)))
    parser.add_argument("--id-gate-length", type=int, default=10)
    parser.add_argument("--id-gate-threshold", type=float, default=0.995)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=684001)
    parser.add_argument("--snapshot-stride", type=int, default=512)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cuda")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def snapshot_update(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("controller_"):
        raise ValueError(f"unexpected snapshot name: {path}")
    return int(stem.removeprefix("controller_"))


def snapshots(controller_dir: Path, stride: int) -> list[Path]:
    if stride < 1:
        raise ValueError("snapshot stride must be positive")
    paths = sorted(
        controller_dir.glob("checkpoints/controller_*.pt"),
        key=snapshot_update,
    )
    selected = [
        path
        for path in paths
        if snapshot_update(path) == 0 or snapshot_update(path) % stride == 0
    ]
    final_snapshot = max(paths, key=snapshot_update) if paths else None
    if final_snapshot is not None and final_snapshot not in selected:
        selected.append(final_snapshot)
    if not selected:
        raise FileNotFoundError(f"no controller snapshots under {controller_dir}")
    return sorted(selected, key=snapshot_update)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    if args.id_gate_length not in args.validation_lengths:
        raise ValueError("ID gate length must be included in validation lengths")
    device = pick_device(args.device)
    model, spec, backbone_payload = load_backbone(args.checkpoint, device=device)
    if spec.name != "addition" or not spec.addition_lsb_first:
        raise ValueError("snapshot selector requires the LSB-first Addition baseline")

    raw_by_length = {
        length: evaluate_variant(
            model=model,
            spec=spec,
            controller=None,
            controller_start_step=None,
            executor_off=False,
            logical_length=length,
            batch_size=args.batch_size,
            batches=args.batches,
            seed=args.seed,
            device=device,
        )
        for length in args.validation_lengths
    }

    flat_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    candidates: dict[int, Path] = {}
    for path in snapshots(args.controller_dir, args.snapshot_stride):
        update = snapshot_update(path)
        controller, payload = load_controller(path, device=device)
        if Path(payload["checkpoint"]) != args.checkpoint:
            raise ValueError(f"snapshot backbone mismatch: {path}")
        if payload.get("controller_post_final_j") is not False:
            raise ValueError(f"post-final J is not allowed: {path}")
        # ``ControllerView(mode="full")`` exposes the factorized controller's
        # complete D + AB + b map.  A dense controller already is the complete
        # affine map and does not have the factor-specific ``diagonal`` field.
        view = (
            ControllerView(controller, mode="full")
            if hasattr(controller, "diagonal")
            else controller
        ).to(device).eval()
        anchor_step = int(payload["anchor_step"])
        metrics_by_length: dict[int, dict[str, float]] = {}
        for length in args.validation_lengths:
            metrics = evaluate_variant(
                model=model,
                spec=spec,
                controller=view,
                controller_start_step=anchor_step,
                executor_off=False,
                logical_length=length,
                batch_size=args.batch_size,
                batches=args.batches,
                seed=args.seed,
                device=device,
            )
            metrics_by_length[length] = metrics
            raw = raw_by_length[length]
            flat_rows.append(
                {
                    "snapshot_update": update,
                    "length": length,
                    "examples": int(metrics["examples"]),
                    "raw_digit_em": float(raw["supervised_digit_exact_match"]),
                    "full_digit_em": float(metrics["supervised_digit_exact_match"]),
                    "delta_digit_em": float(
                        metrics["supervised_digit_exact_match"]
                        - raw["supervised_digit_exact_match"]
                    ),
                    "raw_bit_accuracy": float(raw["supervised_digit_bit_accuracy"]),
                    "full_bit_accuracy": float(metrics["supervised_digit_bit_accuracy"]),
                    "delta_bit_accuracy": float(
                        metrics["supervised_digit_bit_accuracy"]
                        - raw["supervised_digit_bit_accuracy"]
                    ),
                }
            )
        id_em = float(
            metrics_by_length[args.id_gate_length]["supervised_digit_exact_match"]
        )
        exact_values = np.asarray(
            [
                metrics_by_length[length]["supervised_digit_exact_match"]
                for length in args.validation_lengths
            ],
            dtype=float,
        )
        raw_exact_values = np.asarray(
            [
                raw_by_length[length]["supervised_digit_exact_match"]
                for length in args.validation_lengths
            ],
            dtype=float,
        )
        bit_values = np.asarray(
            [
                metrics_by_length[length]["supervised_digit_bit_accuracy"]
                for length in args.validation_lengths
            ],
            dtype=float,
        )
        aggregate_rows.append(
            {
                "snapshot_update": update,
                "id_gate_em": id_em,
                "id_gate_passed": id_em >= args.id_gate_threshold,
                "mean_digit_em": float(exact_values.mean()),
                "mean_raw_digit_em": float(raw_exact_values.mean()),
                "mean_delta_digit_em": float((exact_values - raw_exact_values).mean()),
                "mean_bit_accuracy": float(bit_values.mean()),
            }
        )
        candidates[update] = path
        del view, controller
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    eligible = [row for row in aggregate_rows if row["id_gate_passed"]]
    if not eligible:
        raise RuntimeError("no snapshot passed the ID-retention gate")
    best = max(
        eligible,
        key=lambda row: (
            float(row["mean_digit_em"]),
            float(row["mean_bit_accuracy"]),
            -int(row["snapshot_update"]),
        ),
    )
    best_update = int(best["snapshot_update"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "snapshot_length_metrics.csv", flat_rows)
    write_csv(args.out_dir / "snapshot_aggregate_metrics.csv", aggregate_rows)
    selected_path = args.out_dir / "selected_controller.pt"
    shutil.copy2(candidates[best_update], selected_path)
    result = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(backbone_payload["step"]),
        "controller_dir": str(args.controller_dir),
        "selected_source": str(candidates[best_update]),
        "selected_controller": str(selected_path),
        "selected_snapshot_update": best_update,
        "selection_metrics": best,
        "validation_lengths": list(args.validation_lengths),
        "validation_seed": int(args.seed),
        "examples_per_length": args.batch_size * args.batches,
        "id_gate": {
            "length": int(args.id_gate_length),
            "threshold": float(args.id_gate_threshold),
        },
        "unopened_test_lengths": list(range(max(args.validation_lengths) + 1, 31)),
        "selection_rule": (
            "among snapshots passing ID retention, maximize held-out-data mean "
            "supervised-digit exact match, then mean bit accuracy, then prefer "
            "the earlier update"
        ),
        "claim_boundary": (
            "snapshot selection uses held-out examples at controller-exposed "
            "lengths; lengths above the controller training maximum remain "
            "unopened until final evaluation"
        ),
    }
    (args.out_dir / "selection.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()

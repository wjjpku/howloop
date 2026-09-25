from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import torch

from reasoning_loop.paper_length_telomere import (
    _controlled_ce_batch,
    exact_match,
    generate_paper_batch,
    load_backbone,
    load_controller,
    masked_cross_entropy,
    pick_device,
)


def boundary_validation_lengths(minimum: int, maximum: int) -> tuple[int, ...]:
    if minimum < 1 or maximum < minimum:
        raise ValueError("boundary lengths require a positive ordered interval")
    width = maximum - minimum
    result = tuple(
        sorted({int(round(minimum + width * fraction / 4)) for fraction in range(5)})
    )
    if result[0] != minimum or result[-1] != maximum:
        raise AssertionError("boundary validation endpoints changed unexpectedly")
    return result


def validation_seed_for_length(base_seed: int, logical_length: int) -> int:
    if logical_length < 1:
        raise ValueError("validation logical length must be positive")
    return int(base_seed) + 1009 * int(logical_length)


def choose_checkpoint(
    rows: Sequence[Mapping[str, object]], *, retention_tolerance: float
) -> dict[str, object]:
    if retention_tolerance < 0:
        raise ValueError("retention tolerance cannot be negative")
    grouped: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["snapshot_update"])].append(row)
    if not grouped:
        raise ValueError("checkpoint selection requires validation rows")

    summaries: list[dict[str, object]] = []
    for update, update_rows in sorted(grouped.items()):
        retention = [row for row in update_rows if row["split"] == "retention"]
        boundary = [row for row in update_rows if row["split"] == "boundary"]
        if not retention or not boundary:
            raise ValueError(
                f"snapshot {update} requires both retention and boundary rows"
            )
        retention_deltas = [
            float(row["controller_exact_match"])
            - float(row["raw_exact_match"])
            for row in retention
        ]
        minimum_retention_delta = min(retention_deltas)
        summaries.append(
            {
                "selected_update": update,
                "retention_pass": minimum_retention_delta >= -retention_tolerance,
                "minimum_retention_delta": minimum_retention_delta,
                "boundary_mean_ce": sum(float(row["controller_ce"]) for row in boundary)
                / len(boundary),
                "boundary_mean_accuracy": sum(
                    float(row["controller_exact_match"]) for row in boundary
                )
                / len(boundary),
            }
        )

    eligible = [row for row in summaries if bool(row["retention_pass"])]
    if eligible:
        selected = min(
            eligible,
            key=lambda row: (
                float(row["boundary_mean_ce"]),
                -float(row["boundary_mean_accuracy"]),
                int(row["selected_update"]),
            ),
        )
        status = "retention_pass"
    else:
        selected = min(
            summaries,
            key=lambda row: (
                -float(row["minimum_retention_delta"]),
                float(row["boundary_mean_ce"]),
                int(row["selected_update"]),
            ),
        )
        status = "fallback_no_retention_pass"
    return {**selected, "selection_status": status}


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty validation table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def _evaluate_snapshot_at_length(
    *,
    model: torch.nn.Module,
    spec: object,
    controller: torch.nn.Module,
    logical_length: int,
    validation_seed: int,
    examples: int,
    batch_size: int,
    ce_temperature: float,
    anchor_step: int,
    device: torch.device,
) -> dict[str, float]:
    if examples < 1 or batch_size < 1:
        raise ValueError("validation examples and batch size must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(validation_seed_for_length(validation_seed, logical_length))
    raw_loss_sum = 0.0
    controlled_loss_sum = 0.0
    raw_accuracy_sum = 0.0
    controlled_accuracy_sum = 0.0
    completed = 0
    while completed < examples:
        live_batch_size = min(batch_size, examples - completed)
        batch = generate_paper_batch(
            spec,
            batch_size=live_batch_size,
            min_length=logical_length,
            max_length=logical_length,
            fixed_length=logical_length,
            generator=generator,
        ).to(device)
        state = torch.zeros(
            live_batch_size,
            batch.inputs.shape[1],
            model.config.d_model,
            device=device,
        )
        for step_index in range(1, logical_length + spec.step_offset + 1):
            embedded = model.input_embeddings(batch.inputs, step_index=step_index)
            state = model.recurrent_step(state, embedded)
        raw_logits = model.decode(state).float()
        raw_loss = masked_cross_entropy(raw_logits / ce_temperature, batch)
        raw_accuracy = exact_match(raw_logits, batch)
        controlled_loss, controlled_accuracy = _controlled_ce_batch(
            model=model,
            controller=controller,
            batch=batch,
            anchor_step=anchor_step,
            controlled_steps=logical_length + spec.step_offset - anchor_step,
            ce_temperature=ce_temperature,
            post_final_controller=False,
        )
        raw_loss_sum += float(raw_loss) * live_batch_size
        controlled_loss_sum += float(controlled_loss) * live_batch_size
        raw_accuracy_sum += float(raw_accuracy) * live_batch_size
        controlled_accuracy_sum += float(controlled_accuracy) * live_batch_size
        completed += live_batch_size
    return {
        "raw_ce": raw_loss_sum / examples,
        "controller_ce": controlled_loss_sum / examples,
        "raw_exact_match": raw_accuracy_sum / examples,
        "controller_exact_match": controlled_accuracy_sum / examples,
    }


def validate_controller_checkpoints(args: argparse.Namespace) -> dict[str, object]:
    if args.logical_min_length > args.logical_max_length:
        raise ValueError("logical validation interval is reversed")
    device = pick_device(args.device)
    model, spec, backbone_payload = load_backbone(args.checkpoint, device=device)
    snapshots = sorted((args.controller_dir / "checkpoints").glob("controller_*.pt"))
    if not snapshots:
        raise FileNotFoundError("controller directory has no checkpoint snapshots")
    retention_lengths = (10, 20)
    boundary_lengths = boundary_validation_lengths(
        args.logical_min_length, args.logical_max_length
    )
    rows: list[dict[str, object]] = []
    for snapshot in snapshots:
        controller, payload = load_controller(snapshot, device=device)
        if str(payload["checkpoint"]) != str(args.checkpoint):
            raise ValueError("controller snapshot belongs to a different backbone")
        if int(payload["anchor_step"]) != 1:
            raise ValueError("boundary selection requires controller anchor 1")
        update = int(payload["snapshot_update"])
        ce_temperature = float(payload["controller_ce_temperature"])
        for split, lengths in (
            ("retention", retention_lengths),
            ("boundary", boundary_lengths),
        ):
            for logical_length in lengths:
                metrics = _evaluate_snapshot_at_length(
                    model=model,
                    spec=spec,
                    controller=controller,
                    logical_length=logical_length,
                    validation_seed=args.validation_seed,
                    examples=args.examples_per_length,
                    batch_size=args.batch_size,
                    ce_temperature=ce_temperature,
                    anchor_step=1,
                    device=device,
                )
                row: dict[str, object] = {
                    "snapshot_update": update,
                    "snapshot": str(snapshot),
                    "split": split,
                    "length": logical_length,
                    "validation_seed": validation_seed_for_length(
                        args.validation_seed, logical_length
                    ),
                    "examples": args.examples_per_length,
                    "ce_temperature": ce_temperature,
                    **metrics,
                }
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
    selection = choose_checkpoint(
        rows, retention_tolerance=args.retention_tolerance
    )
    selected_update = int(selection["selected_update"])
    selected_rows = [
        row for row in rows if int(row["snapshot_update"]) == selected_update
    ]
    selected_snapshot = Path(str(selected_rows[0]["snapshot"]))
    output_dir = args.out_dir or args.controller_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "checkpoint_validation.csv", rows)
    best_controller = output_dir / "best_controller.pt"
    shutil.copy2(selected_snapshot, best_controller)
    result: dict[str, object] = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "backbone_step": int(backbone_payload["step"]),
        "controller_dir": str(args.controller_dir),
        "logical_training_range": [
            args.logical_min_length,
            args.logical_max_length,
        ],
        "controller_ood_definition": f"n>{args.logical_max_length}",
        "retention_lengths": list(retention_lengths),
        "boundary_validation_lengths": list(boundary_lengths),
        "validation_seed": args.validation_seed,
        "examples_per_length": args.examples_per_length,
        "retention_tolerance": args.retention_tolerance,
        "snapshot_count": len(snapshots),
        "selected_snapshot": str(selected_snapshot),
        "best_controller": str(best_controller),
        **selection,
    }
    (output_dir / "selection.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a Parity boundary J checkpoint without using OOD lengths."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--controller-dir", type=Path, required=True)
    parser.add_argument("--logical-min-length", type=int, required=True)
    parser.add_argument("--logical-max-length", type=int, required=True)
    parser.add_argument("--validation-seed", type=int, required=True)
    parser.add_argument("--examples-per-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--retention-tolerance", type=float, default=0.005)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    validate_controller_checkpoints(parse_args(argv))


if __name__ == "__main__":
    main()


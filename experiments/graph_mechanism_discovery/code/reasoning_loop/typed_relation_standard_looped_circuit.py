from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Sequence

import torch

from reasoning_loop.typed_relation_circuit import (
    cell_schedule_table,
    enumerate_component_circuits,
    load_checkpoint,
    phase_circuit_table,
)
from reasoning_loop.typed_relation_composition import (
    RelationBatch,
    TypedRelationConfig,
    TypedRelationModel,
    make_relation_batch,
    relation_visibility,
)
from reasoning_loop.typed_relation_train import pick_device


def _mask_signature(row: dict[str, Any]) -> tuple[bool, ...]:
    bits: list[bool] = []
    head_count = int(row["n_heads"])
    for heads, keep_mlp in zip(
        row["kept_heads"], row["kept_mlps"], strict=True
    ):
        head_set = set(heads)
        bits.extend(head in head_set for head in range(head_count))
        bits.append(bool(keep_mlp))
    return tuple(bits)


def _component_set(row: dict[str, Any]) -> set[str]:
    result: set[str] = set()
    for loop_index, (heads, keep_mlp) in enumerate(
        zip(row["kept_heads"], row["kept_mlps"], strict=True),
        start=1,
    ):
        result.update(f"loop{loop_index}.head{head}" for head in heads)
        if keep_mlp:
            result.add(f"loop{loop_index}.mlp")
    return result


def summarize_circuits(
    rows: list[dict[str, Any]],
    *,
    node_metric: str,
    accuracy_threshold: float = 0.99,
) -> dict[str, Any]:
    if not rows or node_metric not in {"effective_nodes", "parameter_nodes"}:
        raise ValueError("rows and a valid node metric are required")
    qualifying = [
        row for row in rows if row["endpoint_accuracy"] >= accuracy_threshold
    ]
    baseline = max(rows, key=lambda row: row["effective_nodes"])
    result: dict[str, Any] = {
        "evaluated_circuits": len(rows),
        "accuracy_threshold": accuracy_threshold,
        "baseline_accuracy": baseline["endpoint_accuracy"],
        "baseline_margin": baseline["mean_logit_margin"],
        "faithful_circuit_count": len(qualifying),
    }
    if not qualifying:
        result["minimum_nodes"] = None
        result["minimal_circuits"] = []
        return result
    minimum_nodes = min(row[node_metric] for row in qualifying)
    minimal = [row for row in qualifying if row[node_metric] == minimum_nodes]
    same_size = [row for row in rows if row[node_metric] == minimum_nodes]
    all_signatures = {_mask_signature(row): row for row in rows}
    minimal_rows: list[dict[str, Any]] = []
    for row in minimal:
        signature = _mask_signature(row)
        complement = all_signatures.get(tuple(not bit for bit in signature))
        minimal_rows.append(
            {
                "kept_heads": row["kept_heads"],
                "kept_mlps": row["kept_mlps"],
                "effective_nodes": row["effective_nodes"],
                "parameter_nodes": row["parameter_nodes"],
                "circuit_only_accuracy": row["endpoint_accuracy"],
                "circuit_only_margin": row["mean_logit_margin"],
                "complement_only_accuracy": (
                    None if complement is None else complement["endpoint_accuracy"]
                ),
                "complement_only_margin": (
                    None if complement is None else complement["mean_logit_margin"]
                ),
            }
        )
    component_sets = [_component_set(row) for row in minimal]
    jaccards: list[float] = []
    for left_index, left in enumerate(component_sets):
        for right in component_sets[left_index + 1 :]:
            union = left | right
            jaccards.append(len(left & right) / len(union) if union else 1.0)
    result.update(
        {
            "minimum_nodes": minimum_nodes,
            "minimal_circuit_count": len(minimal),
            "minimal_circuits": minimal_rows,
            "same_size_control_count": len(same_size),
            "same_size_accuracy_mean": statistics.mean(
                row["endpoint_accuracy"] for row in same_size
            ),
            "same_size_accuracy_max": max(
                row["endpoint_accuracy"] for row in same_size
            ),
            "same_size_faithful_count": sum(
                row["endpoint_accuracy"] >= accuracy_threshold
                for row in same_size
            ),
            "minimal_pairwise_jaccard_mean": (
                statistics.mean(jaccards) if jaccards else 1.0
            ),
            "minimal_pairwise_jaccard_min": min(jaccards) if jaccards else 1.0,
        }
    )
    return result


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, list) else value
                    for key, value in row.items()
                }
            )


def _expand_looped_depth(
    model: TypedRelationModel,
    payload: dict[str, Any],
    *,
    max_loops: int,
    device: torch.device,
) -> TypedRelationModel:
    if model.shared_cell is None or model.cfg.loops >= max_loops:
        return model
    config = dict(payload["config"])
    config["loops"] = max_loops
    expanded = TypedRelationModel(TypedRelationConfig.from_dict(config)).to(device)
    expanded.load_state_dict(model.state_dict())
    expanded.eval()
    return expanded


@torch.no_grad()
def _depth_accuracy_table(
    model: TypedRelationModel,
    batch: RelationBatch,
    visibility: torch.Tensor,
) -> list[dict[str, Any]]:
    logits, _ = model(batch, visibility)
    target = batch.targets[:, 1]
    return [
        {
            "loop_count": loop_index + 1,
            "endpoint_accuracy": float(
                (logits[:, loop_index].argmax(dim=-1) == target).float().mean().item()
            ),
        }
        for loop_index in range(model.cfg.loops)
    ]


@torch.no_grad()
def analyze_checkpoint_deep(
    checkpoint: Path,
    out_dir: Path,
    *,
    device: torch.device,
    examples: int,
    seed: int,
) -> dict[str, Any]:
    model, payload = load_checkpoint(checkpoint, device)
    model = _expand_looped_depth(model, payload, max_loops=4, device=device)
    train_loops = int(payload["train_loops"])
    if train_loops != 2:
        raise ValueError("standard-versus-looped comparison requires two trained loops")
    composition_order = payload.get("composition_order", "f_then_g")
    generator = torch.Generator(device=device).manual_seed(seed)
    donor = make_relation_batch(
        batch_size=examples,
        node_count=model.cfg.node_count,
        device=device,
        generator=generator,
        composition_order=composition_order,
    )
    receiver = make_relation_batch(
        batch_size=examples,
        node_count=model.cfg.node_count,
        device=device,
        generator=generator,
        composition_order=composition_order,
    )
    visibility = relation_visibility(
        "full",
        batch_size=examples,
        loops=model.cfg.loops,
        node_count=model.cfg.node_count,
        device=device,
        composition_order=composition_order,
    )
    effective_rows = enumerate_component_circuits(
        model,
        donor,
        visibility,
        train_loops=train_loops,
        tie_masks_across_loops=False,
    )
    tied_rows: list[dict[str, Any]] | None = None
    if model.shared_cell is not None:
        tied_rows = enumerate_component_circuits(
            model,
            donor,
            visibility,
            train_loops=train_loops,
            tie_masks_across_loops=True,
        )
    result: dict[str, Any] = {
        "checkpoint": str(checkpoint.resolve()),
        "architecture": model.cfg.architecture,
        "model_seed": int(payload["seed"]),
        "analysis_seed": seed,
        "examples": examples,
        "composition_order": composition_order,
        "depth_accuracy": _depth_accuracy_table(model, donor, visibility),
        "phase": phase_circuit_table(
            model,
            donor,
            receiver,
            train_loops=train_loops,
        ),
        "cell_schedules": cell_schedule_table(
            model,
            donor,
            visibility,
            train_loops=train_loops,
        ),
        "effective_search": summarize_circuits(
            effective_rows,
            node_metric="effective_nodes",
        ),
        "tied_parameter_search": (
            None
            if tied_rows is None
            else summarize_circuits(tied_rows, node_metric="parameter_nodes")
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(out_dir / "effective_circuits.csv", effective_rows)
    if tied_rows is not None:
        _write_rows(out_dir / "tied_parameter_circuits.csv", tied_rows)
    (out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=92_021)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = analyze_checkpoint_deep(
        args.checkpoint,
        args.out_dir,
        device=pick_device(args.device),
        examples=args.examples,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "architecture": result["architecture"],
                "effective_search": result["effective_search"],
                "tied_parameter_search": result["tied_parameter_search"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
